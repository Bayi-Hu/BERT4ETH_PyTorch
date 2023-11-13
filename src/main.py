from options import args
from dataset import ML1MDataset
from dataloader import BertDataloader
from modeling import BERT
from trainer import BERTTrainer
from utils import *


def train():
    export_root = setup_train(args)

    # dataset
    dataset = ML1MDataset(args)

    # dataloader
    dataloader = BertDataloader(args, dataset)
    train_loader = dataloader.get_pytorch_dataloaders()

    # model
    model = BERT(args)

    # tranier
    trainer = BERTTrainer(args, model, train_loader)
    trainer.train()


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
