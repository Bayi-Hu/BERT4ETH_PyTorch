import numpy as np
import os

import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser("phishing_detection")
parser.add_argument("--input_dir", type=str, default="../outputs/1130_epoch_50", help="the input directory of address and embedding list")
parser.add_argument("--train_batch_size", type=int, default=256, help="the input directory of address and embedding list")

args = parser.parse_args()


class MLP(nn.Module):
    def __init__(self, dataloader):
        super(MLP, self).__init__()
        self.dataloader = dataloader
        self.input_dim = 64
        self.hidden_dim = 256
        self.num_epochs = 2
        self.lr = 5e-4
        self.device = "cuda"
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim).to(self.device)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
        self.out_layer = nn.Linear(self.hidden_dim, 2).to(self.device)
        self.optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)

    def forward(self, x):
        dnn1 = F.relu(self.fc1(x))
        dnn2 = F.relu(self.fc2(dnn1))
        logits = self.out_layer(dnn1+dnn2)
        return logits

    def fit(self):
        self.train()
        accum_iter = 0
        for epoch in range(self.num_epochs):
            # for each epoch
            tqdm_dataloader = tqdm(self.dataloader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x for x in batch]

                X_batch = batch[0]
                y_batch = batch[1]
                X_batch = torch.tensor(X_batch).to(self.device)
                y_batch = torch.tensor(y_batch).to(self.device)

                self.optimizer.zero_grad()
                logits = self.forward(X_batch)
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y_batch.view(-1))
                loss.backward()
                self.optimizer.step()
                tqdm_dataloader.set_description(
                    'Epoch {}, loss {:.3f} '.format(epoch, loss.item())
                )
                batch_size = X_batch.shape[0]
                accum_iter += batch_size

        return
    def predict_proba(self, X_test):
        X_test = torch.tensor(X_test).to(self.device)
        logits = self.forward(X_test)
        y_test = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()

        return y_test


class TrainDataset(data_utils.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    def __getitem__(self, index):
        X_batch = self.X[index]
        y_batch = self.y[index]
        return X_batch, y_batch


def main():

    phisher_account_set = set()
    with open("../data/phisher_account.txt", "r") as f:
        for line in f.readlines():
            phisher_account_set.add(line[:-1])

    address_input_dir = os.path.join(args.input_dir, "address.npy")
    embed_input_dir = os.path.join(args.input_dir, "embedding.npy")

    address_list = np.load(address_input_dir)
    embedding_list = np.load(embed_input_dir)

    y = []
    for addr in address_list:
        if addr in phisher_account_set:
            y.append(1)
        else:
            y.append(0)

    X_train, X_test, y_train, y_test = train_test_split(embedding_list, y, test_size=0.3, random_state=42)

    print(X_train.shape)
    print(X_test.shape)
    # define dataset
    dataset = TrainDataset(X_train, y_train)
    dataloader = data_utils.DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True)

    # model
    model = MLP(dataloader)
    model.fit()

    y_test_proba = model.predict_proba(X_test)
    print(y_test_proba.shape)

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        print("threshold =", threshold)
        y_pred = np.zeros_like(y_test_proba)
        y_pred[np.where(np.array(y_test_proba) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    main()