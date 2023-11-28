import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_curve, auc, precision_recall_curve


flags.DEFINE_bool("visual", False, "whether to do visualization or not")
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")

def load_embedding():

    # must have checkpoint
    if FLAGS.init_checkpoint == None:
        raise ValueError("Must need a checkpoint for evaluation")

    checkpoint_name = FLAGS.init_checkpoint.split("/")[0]
    model_index = str(FLAGS.init_checkpoint.split("/")[-1].split("_")[1])
    embeddings = np.load("./inter_data/embedding_" + checkpoint_name + "_" + model_index + ".npy")
    address_for_embedding = np.load("./inter_data/address_" + checkpoint_name + "_" + model_index + ".npy")

    # group by embedding according to address
    address_to_embedding = {}

    for i in range(len(address_for_embedding)):
        address = address_for_embedding[i]
        embedding = embeddings[i]
        try:
            address_to_embedding[address].append(embedding)
        except:
            address_to_embedding[address] = [embedding]

    # group to one
    address_list = []
    embedding_list = []

    for addr, embeds in address_to_embedding.items():
        address_list.append(addr)
        if len(embeds) > 1:
            embedding_list.append(np.mean(embeds, axis=0))
        else:
            embedding_list.append(embeds[0])

    # final embedding table
    X = np.array(np.squeeze(embedding_list))

    return X, address_list

def main():

    phisher_account = pd.read_csv("../Data/phisher_account.txt", names=["account"])
    phisher_account_set = set(phisher_account.account.values)

    X, address_list = load_embedding()

    y = []
    for addr in address_list:
        if addr in phisher_account_set:
            y.append(1)
        else:
            y.append(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    model.fit(X_train, y_train)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    # rf_output = np.array(list(zip(y_test_proba, y_test)))
    # if FLAGS.algo in ["sage", "gcn", "gat", "gin"]:
    #     np.save("dgl_gnn/data/" + FLAGS.algo + "_phisher_account_output_" + FLAGS.dataset + ".npy", rf_output)
    # else:
    #     np.save(FLAGS.algo + "/data/phisher_account_output_"+ FLAGS.dataset +".npy", rf_output)

    print("=============Precision-Recall Curve=============")
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_proba)
    plt.figure("P-R Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall, precision)
    plt.show()

    # print("================ROC Curve====================")
    model_index = str(FLAGS.init_checkpoint.split("/")[-1].split("_")[1])
    print("model_index:", model_index)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba, pos_label=1)
    print("AUC=", auc(fpr, tpr))

    plt.figure("ROC Curve")
    plt.title("ROC Curve")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.plot(fpr, tpr)
    plt.show()

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]:
        print("threshold =", threshold)
        y_pred = np.zeros_like(y_test_proba)
        y_pred[np.where(np.array(y_test_proba) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(y_test, y_pred, digits=4))

    if FLAGS.visual:
        # visualization
        y = np.array(y)
        p_idx = np.where(y == 1)[0]
        n_idx = np.where(y == 0)[0]
        X_phisher = X[p_idx]
        X_normal = X[n_idx]

        permutation = np.random.permutation(len(X_normal))
        X_normal_sample = X_normal[permutation[:10000]]
        X4tsne = np.concatenate([X_normal_sample, X_phisher], axis=0)
        tsne = TSNE(n_components=2, init="random")
        X_tsne = tsne.fit_transform(X4tsne)

        # plot
        plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(x=X_tsne[:10000, 0], y=X_tsne[:10000, 1], marker=".")
        plt.scatter(x=X_tsne[10000:, 0], y=X_tsne[10000:, 1], marker=".", color="orange")
        plt.show()

        plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(x=X_tsne[:10000, 0], y=X_tsne[:10000, 1], marker=".")
        plt.show()


if __name__ == '__main__':
    main()