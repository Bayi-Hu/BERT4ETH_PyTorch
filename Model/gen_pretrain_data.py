import torch
import torch.utils.data as data_utils

import numpy as np
import pickle as pkl
from tqdm import tqdm
import collections
import functools
import random
import argparse

import six
import time
import math
from vocab import FreqVocab

random_seed = 12345
rng = random.Random(random_seed)

parser = argparse.ArgumentParser(description="Data Processing with PyTorch")
parser.add_argument('--pool_size', type=int, default=10, help='multiprocesses pool size.')
parser.add_argument('--max_seq_length', type=int, default=100, help='max sequence length.')
parser.add_argument('--masked_lm_prob', type=float, default=0.8, help='Masked LM probability.')
parser.add_argument('--mask_prob', type=float, default=1.0, help='mask probability')
parser.add_argument('--do_eval', action='store_true', help='Whether to do evaluation.')
parser.add_argument('--do_embed', action='store_true', default=True, help='Whether to do embedding.')
parser.add_argument('--dupe_factor', type=int, default=10, help='Number of times to duplicate the input data (with different masks).')
parser.add_argument('--data_dir', type=str, default='./inter_data/', help='data dir.')
parser.add_argument('--vocab_filename', type=str, default='vocab', help='vocab filename')
parser.add_argument('--bizdate', type=str, default=None, help='the signature of running experiments')

args = parser.parse_args()

if args.bizdate is None:
    raise ValueError("bizdate is required.")

HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(
    ",")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

MAX_PREDICTIONS_PER_SEQ = math.ceil(args.max_seq_length * args.masked_lm_prob)
SLIDING_STEP = round(args.max_seq_length * 0.6)

print("MAX_SEQUENCE_LENGTH:", args.max_seq_length)
print("MAX_PREDICTIONS_PER_SEQ:", MAX_PREDICTIONS_PER_SEQ)
print("SLIDING_STEP:", SLIDING_STEP)

class BertTrainDataset(data_utils.Dataset):
    def __init__(self, seq_list, mask_prob, mask_token, max_predictions_per_seq):
        self.seq_list = seq_list
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.max_predictions_per_seq = max_predictions_per_seq
        self.rng = rng
    def __getitem__(self, index):

        # only one index as input
        tranxs = self.seq_list[index]
        address = tranxs[0][0]
        cand_indexes = []
        for (i, token) in enumerate(tranxs):
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)
        num_to_predict = min(self.max_predictions_per_seq,
                         max(1, int(len(tranxs) * self.masked_lm_prob)))

        masked_lms = []
        covered_indexes = set()
        labels = [0 for i in range(len(tranxs))] # labels = 0 denotes not masked.

        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)
            labels[index] = tranxs[index][0]
            tranxs[index][0] = "[MASK]"

        # MAP discrete feature to int

        address = [address]
        tokens = list(map(lambda x: x[0], tranxs))
        block_timestamps = list(map(lambda x: x[2], tranxs))
        values = list(map(lambda x: x[3], tranxs))

        def map_io_flag(tranxs):
            flag = tranxs[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        io_flags = list(map(map_io_flag, tranxs))
        cnts = list(map(lambda x: x[5], tranxs))

        return torch.LongTensor(tokens), \
               torch.LongTensor(block_timestamps), \
               torch.LongTensor(values), \
               torch.LongTensor(io_flags), \
               torch.LongTensor(cnts), \
               torch.LongTensor(labels)


def gen_samples(sequences,
                dupe_factor,
                masked_lm_prob,
                max_predictions_per_seq,
                pool_size,
                rng):
    instances = []
    # create train
    for step in range(dupe_factor):
        start = time.time()
        for tokens in sequences:
            (address, tokens, masked_lm_positions,
             masked_lm_labels) = create_masked_lm_predictions(
                tokens, masked_lm_prob, max_predictions_per_seq, rng)
            instance = TrainingInstance(
                address=address,
                tokens=tokens,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            instances.append(instance)
        end = time.time()
        cost = end - start
        print("step=%d, time=%.2f" % (step, cost))
    print("=======Finish========")
    return instances

def create_embedding_predictions(tokens):
    """Creates the predictions for the masked LM objective."""
    address = tokens[0][0]
    output_tokens = tokens
    masked_lm_positions = []
    masked_lm_labels = []
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def gen_embedding_samples(sequences):
    instances = []
    # create train
    start = time.time()
    for tokens in sequences:
        (address, tokens, masked_lm_positions,
         masked_lm_labels) = create_embedding_predictions(tokens)
        instance = TrainingInstance(
            address=address,
            tokens=tokens,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances


def convert_timestamp_to_position(block_timestamps):
    position = [0]
    if len(block_timestamps) <= 1:
        return position
    last_ts = block_timestamps[1]
    idx = 1
    for b_ts in block_timestamps[1:]:
        if b_ts != last_ts:
            last_ts = b_ts
            idx += 1
        position.append(idx)
    return position


def cmp_udf_reverse(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0


def main():
    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open(args.data_dir + "eoa2seq_" + args.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))
    vocab.update(eoa2seq)
    # generate mapping
    vocab.generate_vocab()

    # save vocab
    print("token_size:{}".format(len(vocab.vocab_words)))
    vocab_file_name = args.data_dir + args.vocab_filename + "." + args.bizdate
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    print("===========Original===========")
    length_list = []
    for eoa in eoa2seq.keys():
        seq = eoa2seq[eoa]
        length_list.append(len(seq))

    length_list = np.array(length_list)
    print("Median:", np.median(length_list))
    print("Mean:", np.mean(length_list))
    print("Seq num:", len(length_list))

    # clip
    max_num_tokens = args.max_seq_length - 1
    seqs = []
    idx = 0
    for eoa, seq in eoa2seq.items():
        if len(seq) <= max_num_tokens:
            seqs.append([[eoa, 0, 0, 0, 0, 0]])
            seqs[idx] += seq
            idx += 1
        elif len(seq) > max_num_tokens:
            beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * SLIDING_STEP))
            beg_idx.append(0)

            if len(beg_idx) > 500:
                beg_idx = list(np.random.permutation(beg_idx)[:500])
                for i in beg_idx:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

            else:
                for i in beg_idx[::-1]:
                    seqs.append([[eoa, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1

    seqs = np.random.permutation(seqs)

    # start from here






    print("========Generate Training Samples========")
    normal_instances = gen_samples(seqs,
                                   dupe_factor=args.dupe_factor,
                                   masked_lm_prob=args.masked_lm_prob,
                                   max_predictions_per_seq=MAX_PREDICTIONS_PER_SEQ,
                                   pool_size=args.pool_size,
                                   rng=rng)

    write_instance = normal_instances
    rng.shuffle(write_instance)

    output_filename = args.data_dir + "train.tfrecord" + "." + args.bizdate

    return

if __name__ == '__main__':
    main()

