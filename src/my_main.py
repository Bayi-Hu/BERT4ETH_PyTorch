from my_options import args
from dataloader import BERT4ETHDataloader
from modeling import BERT4ETH
from trainer import BERT4ETHTrainer
from utils import *
import argparse
import collections
import math
import pickle as pkl
from vocab import FreqVocab


random_seed = 12345
rng = random.Random(random_seed)

HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(
    ",")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

MAX_PREDICTIONS_PER_SEQ = math.ceil(args.max_seq_length * args.masked_lm_prob)
SLIDING_STEP = round(args.max_seq_length * 0.6)

print("MAX_SEQUENCE_LENGTH:", args.max_seq_length)
print("MAX_PREDICTIONS_PER_SEQ:", MAX_PREDICTIONS_PER_SEQ)
print("SLIDING_STEP:", SLIDING_STEP)

def train():

    # prepare dataset
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

    # read model configuration
    config_file = "bert4eth_config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    # dataloader
    dataloader = BERT4ETHDataloader(args, vocab, eoa2seq)
    train_loader = dataloader.get_pytorch_dataloaders()

    # model
    model = BERT4ETH(args, config)

    # tranier
    trainer = BERT4ETHTrainer(args, model, train_loader)
    trainer.train()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
