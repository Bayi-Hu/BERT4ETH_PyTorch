from config import args
from dataloader import BERT4ETHDataloader
from modeling import BERT4ETH
from trainer import BERT4ETHTrainer
import pickle as pkl
import numpy as np
import os

def infer_embed():

    # prepare dataset
    print("===========Load Sequence===========")
    with open(args.data_dir + "eoa2seq_" + args.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    # load vocab
    with open(args.data_dir + args.vocab_filename + "." + args.bizdate, "rb") as f:
        vocab = pkl.load(f)

    # dataloader
    dataloader = BERT4ETHDataloader(args, vocab, eoa2seq)
    eval_loader = dataloader.get_eval_loader()

    # model
    model = BERT4ETH(args)

    # tranier
    trainer = BERT4ETHTrainer(args, vocab, model, eval_loader)
    trainer.load("1130/epoch_10.pth")

    # tranier
    for name, param in model.named_parameters():
        if name == "embedding.token_embed.weight":
            print(param[200000,:])
            print(param[800000, :])

    # print("Finish..")

if __name__ == '__main__':
    infer_embed()
