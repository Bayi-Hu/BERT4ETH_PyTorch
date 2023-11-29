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
    trainer.load(args.init_checkpoint)
    address_array, seq_embedding_array = trainer.infer_embedding()

    # save embedding
    checkpoint_name = args.init_checkpoint.split("/")[0]
    model_index = str(args.init_checkpoint.split("/")[-1].split(".pth")[0])

    output_dir = args.data_dir + checkpoint_name + "_" + model_index
    os.makedirs(output_dir, exist_ok=True)

    embed_output_dir = os.path.join(output_dir, "embedding.npy")
    address_output_dir = os.path.join(output_dir, "address.npy")

    np.save(embed_output_dir, seq_embedding_array)
    np.save(address_output_dir, address_array)

    print("Finish..")

if __name__ == '__main__':
    infer_embed()
