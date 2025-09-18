from config import args
from pretrain.dataloader import FineTuneLoader
from models.model import FineTuneModel
from pretrain.trainer import PhishAccountTrainer
import pickle as pkl
import numpy as np
from sklearn.metrics import classification_report
import  os
import torch

args.bizdate= 'gas'

def eval(args):
    vocab_file_name = args.data_dir + args.vocab_filename + "." + args.bizdate

    with open(vocab_file_name, "rb") as vocab_file:
        vocab = pkl.load(vocab_file)
    with open(args.data_dir + "eoa2seq_" + args.bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)
    # dataloader
    dataloader = FineTuneLoader(args, vocab, eoa2seq)

    # model
    model = FineTuneModel(args)

    # tranier
    test_loader = dataloader.get_eval_loader()
    trainer = PhishAccountTrainer(args, vocab, model, test_loader)

    # load ckpt
    ckpt_dir = args.ckpt_dir + "_phish"
    content = os.listdir(ckpt_dir)
    full_path = [os.path.join(ckpt_dir, x)  for x in content]
    dir_content = sorted(full_path, key=lambda t: os.stat(t).st_mtime)
    if not len(dir_content):
        raise FileNotFoundError("CKPT file for testing needed")

    ckpt_dir = dir_content[-1]
    print(f"load ckpt at: {ckpt_dir}")

    trainer.model.load_state_dict(torch.load(ckpt_dir))

    final_output, original_data = trainer.predict_proba(test_loader)
    y_test_proba = np.concatenate(final_output)
    y_test = np.concatenate(original_data)

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        print("threshold =", threshold)
        y_pred = np.zeros_like(y_test_proba)
        y_pred[np.where(np.array(y_test_proba) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(y_test, y_pred, digits=4))

if __name__ == '__main__':
    eval(args)

