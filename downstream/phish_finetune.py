from config import args
from pretrain.dataloader import FineTuneLoader
from models.model import FineTuneModel
from pretrain.trainer import PhishAccountTrainer
import pickle as pkl

args.bizdate = 'gas'
args.num_epochs = 2
args.lr = 3e-4

def train():

    vocab_file_name = args.data_dir + args.vocab_filename + "." + args.bizdate

    with open(vocab_file_name, "rb") as vocab_file:
        vocab = pkl.load(vocab_file)
    with open(args.data_dir + "eoa2seq_" + args.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)
    # dataloader
    dataloader = FineTuneLoader(args, vocab, eoa2seq)
    train_loader = dataloader.get_train_loader()

    # model
    model = FineTuneModel(args)

    # tranier
    trainer = PhishAccountTrainer(args, vocab, model, train_loader)
    trainer.train()

if __name__ == '__main__':
    train()

