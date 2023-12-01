import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

phisher_account_set = set()
with open("../data/phisher_account.txt", "r") as f:
    for line in f.readlines():
        phisher_account_set.add(line[:-1])

input_dir = "../outputs/1130_epoch_20"

address_input_dir = os.path.join(input_dir, "address.npy")
embed_input_dir = os.path.join(input_dir, "embedding.npy")

address_list = np.load(address_input_dir)
X = np.load(embed_input_dir)

y = []
for addr in address_list:
    if addr in phisher_account_set:
        y.append(1)
    else:
        y.append(0)

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

print("pause")