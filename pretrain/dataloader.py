from abc import *
from pathlib import Path
import numpy as np
from tqdm import trange
from collections import Counter
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import random
import torch.utils.data as data_utils


def map_io_flag(tranxs):
    flag = tranxs[4]
    if flag == "OUT":
        return 1
    elif flag == "IN":
        return 2
    else:
        return 0

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


class BERT4ETHDataloader:

    def __init__(self, args, vocab, eoa2seq):
        self.args = args
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.eoa2seq = eoa2seq
        self.vocab = vocab
        self.seq_list = self.preprocess(eoa2seq)

    def preprocess(self, eoa2seq):
        self.masked_lm_prob = self.args.masked_lm_prob
        self.rng = random.Random(self.args.dataloader_random_seed)
        self.sliding_step = round(self.args.max_seq_length * 0.6)

        # preprocess
        length_list = []
        for eoa in eoa2seq.keys():
            seq = eoa2seq[eoa]
            length_list.append(len(seq))

        length_list = np.array(length_list)
        print("Median:", np.median(length_list))
        print("Mean:", np.mean(length_list))
        print("Seq num:", len(length_list))

        # clip
        max_num_tokens = self.args.max_seq_length - 1
        seqs = []
        idx = 0
        for eoa, seq in eoa2seq.items():
            if len(seq) <= max_num_tokens:
                seqs.append([[eoa, 0, 0, 0, 0, 0]])
                seqs[idx] += seq
                idx += 1
            elif len(seq) > max_num_tokens:
                beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * self.sliding_step))
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

        self.rng.shuffle(seqs)
        return seqs

    def get_train_loader(self):
        dataset = BERT4ETHTrainDataset(self.args, self.vocab, self.seq_list)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.train_batch_size,
                                           shuffle=True, pin_memory=True)
        return dataloader

    def get_eval_loader(self):
        dataset = BERT4ETHEvalDataset(self.args, self.vocab, self.seq_list)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.args.eval_batch_size,
                                           shuffle=False, pin_memory=True)

        return dataloader


class BERT4ETHTrainDataset(data_utils.Dataset):

    def __init__(self, args, vocab, seq_list):
        # mask_prob, mask_token, max_predictions_per_seq):
        self.args = args
        self.seq_list = seq_list
        self.vocab = vocab
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
        self.max_predictions_per_seq = math.ceil(self.args.max_seq_length * self.args.masked_lm_prob)

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):

        # only one index as input
        tranxs = self.seq_list[index]
        address = tranxs[0][0]
        cand_indexes = []
        for (i, token) in enumerate(tranxs):
            cand_indexes.append(i)

        self.rng.shuffle(cand_indexes)
        num_to_predict = min(self.max_predictions_per_seq,
                         max(1, int(len(tranxs) * self.args.masked_lm_prob)))

        num_masked = 0
        covered_indexes = set()
        labels = [-1 for i in range(len(tranxs))] # labels = -1 denotes not masked.

        for index in cand_indexes:
            if num_masked >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)
            labels[index] = self.vocab.convert_tokens_to_ids([tranxs[index][0]])[0]
            tranxs[index][0] = "[MASK]"
            num_masked += 1

        # MAP discrete feature to int
        address_id = self.vocab.convert_tokens_to_ids([address])
        tokens = list(map(lambda x: x[0], tranxs))
        input_ids = self.vocab.convert_tokens_to_ids(tokens)

        block_timestamps = list(map(lambda x: x[2], tranxs))
        values = list(map(lambda x: x[3], tranxs))
        io_flags = list(map(map_io_flag, tranxs))
        counts = list(map(lambda x: x[5], tranxs))
        positions = convert_timestamp_to_position(block_timestamps)
        input_mask = [1] * len(input_ids)

        max_seq_length = self.args.max_seq_length

        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))
        labels += [-1] * (max_seq_length - len(labels))

        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(labels) == max_seq_length


        return torch.LongTensor(address_id), \
               torch.LongTensor(input_ids), \
               torch.LongTensor(counts), \
               torch.LongTensor(values), \
               torch.LongTensor(io_flags), \
               torch.LongTensor(positions), \
               torch.LongTensor(input_mask), \
               torch.LongTensor(labels)


class BERT4ETHEvalDataset(data_utils.Dataset):

    def __init__(self, args, vocab, seq_list):
        # mask_prob, mask_token, max_predictions_per_seq):
        self.args = args
        self.seq_list = seq_list
        self.vocab = vocab
        seed = args.dataloader_random_seed
        self.rng = random.Random(seed)
    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):

        # only one index as input
        tranxs = self.seq_list[index]
        address = tranxs[0][0]

        # MAP discrete feature to int
        address_id = self.vocab.convert_tokens_to_ids([address])
        tokens = list(map(lambda x: x[0], tranxs))
        input_ids = self.vocab.convert_tokens_to_ids(tokens)

        block_timestamps = list(map(lambda x: x[2], tranxs))
        values = list(map(lambda x: x[3], tranxs))
        io_flags = list(map(map_io_flag, tranxs))
        counts = list(map(lambda x: x[5], tranxs))
        positions = convert_timestamp_to_position(block_timestamps)
        input_mask = [1] * len(input_ids)

        max_seq_length = self.args.max_seq_length

        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length

        return torch.LongTensor(address_id), \
               torch.LongTensor(input_ids), \
               torch.LongTensor(counts), \
               torch.LongTensor(values), \
               torch.LongTensor(io_flags), \
               torch.LongTensor(positions), \
               torch.LongTensor(input_mask), \

