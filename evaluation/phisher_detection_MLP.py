import numpy as np
import os

import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from torch import nn
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser("phishing_detection")
parser.add_argument("--input_dir", type=str, default=None, help="the input directory of address and embedding list")
args = parser.parse_args()


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.input_dim = 64
        self.hidden_dim = 256
        self.lr = 5e-4
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_layer = nn.Linear(self.hidden_dim, 2)
        self.optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)

    def forward(self, x):
        dnn1 = F.relu(self.fc1(x))
        dnn2 = F.relu(self.fc2(dnn1))
        logits = self.out_layer(dnn1+dnn2)
        return logits

    def fit(self, X, y):
        X = torch.tensor(X)
        y = torch.tensor(y)
        self.optimizer.zero_grad()
        logits = self.forward(X)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss.backward()
        self.optimizer.step()
        return





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

    model = MLP()
    model.fit(X_train, y_train)

    y_test_proba = model.predict_proba(X_test)[:, 1]

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        print("threshold =", threshold)
        y_pred = np.zeros_like(y_test_proba)
        y_pred[np.where(np.array(y_test_proba) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    main()