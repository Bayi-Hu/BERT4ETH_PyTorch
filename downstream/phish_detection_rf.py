import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_curve, auc, precision_recall_curve

import argparse

parser = argparse.ArgumentParser("phishing_detection")
parser.add_argument("--input_dir", type=str, default="../inter_data/bert4eth_exp_embed", help="the input directory of address and embedding list")
args = parser.parse_args()


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

    model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    model.fit(X_train, y_train)

    y_test_proba = model.predict_proba(X_test)[:, 1]

    print("=============Precision-Recall Curve=============")
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_proba)
    plt.figure("P-R Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall, precision)
    plt.show()

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        print("threshold =", threshold)
        y_pred = np.zeros_like(y_test_proba)
        y_pred[np.where(np.array(y_test_proba) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(y_test, y_pred, digits=4))


if __name__ == '__main__':
    main()