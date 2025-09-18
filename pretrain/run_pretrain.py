from config import args
from dataloader import BERT4ETHDataloader
from models.model import BERT4ETH
from trainer import BERT4ETHTrainer
import pickle as pkl
from vocab import FreqVocab

args.bizdate= 'gas'

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

    # dataloader
    dataloader = BERT4ETHDataloader(args, vocab, eoa2seq)
    train_loader = dataloader.get_train_loader()

    # model
    model = BERT4ETH(args)

    # trainer
    trainer = BERT4ETHTrainer(args, vocab, model, train_loader)
    trainer.train()


if __name__ == '__main__':
    train()

