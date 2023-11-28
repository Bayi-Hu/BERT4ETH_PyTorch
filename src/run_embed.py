from options import args
from dataloader import BERT4ETHDataloader
from modeling import BERT4ETH
from trainer import BERT4ETHTrainer
import pickle as pkl
import numpy as np
from vocab import FreqVocab


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
    trainer.load(args.ckpt_dir)
    address_array, seq_embedding_array = trainer.infer_embedding()
    # save embedding
    #
    checkpoint_name = args.init_checkpoint.split("/")[0]
    model_index = str(args.init_checkpoint.split("/")[-1].split("_")[1])
    embed_output_file = args.data_dir + "embedding" + checkpoint_name + "_" + model_index + ".npy"

    # How about save as a dict ?
    print(embed_output_file)
    np.save(embed_output_file, seq_embedding_array)
    address_output_file = args.data_dir + "address" + checkpoint_name + "_" + model_index + ".npy"
    print(address_output_file)
    np.save(address_output_file, address_array)

    print("Finish..")

if __name__ == '__main__':
    infer_embed()
