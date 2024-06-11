import numpy as np
from sklearn.datasets import make_blobs

datafolder = "data/"

datasets = [
  # (n,    k, d, r)
    #(1000, 4, 2, 10),
    (1000, 4, 4, 11),
    #(1000, 4, 6, 12),
    #(1000, 4, 8, 13),
    #(1000, 6, 2, 14),
    #(1000, 6, 4, 15),
    #(1000, 6, 6, 16),
    #(1000, 6, 8, 17),
    #(1000, 8, 2, 18),
    #(1000, 8, 4, 19),
    #(1000, 8, 6, 20),
    #(1000, 8, 8, 21),
]

def print_dataset_info():
    print("{0}\t{1}\t{2}".format("ID", "Dim.", "Cat."))

    for i in range(len(datasets)):
        n, k, d, r = datasets[i]
        name = "gen_{0}_{1}_{2}".format(n, d, k)
        print("{0}\t{1}\t{2}\t{3}".format(i+1, name, d, k))

def generate_blobs():
    for dataset_info in datasets:
        n, k, d, r = dataset_info
        X, y = make_blobs(n_samples=n, centers=k, n_features=d, center_box=(-6, 6), random_state=r)
        np.savetxt(datafolder + "gen_{0}_{1}_{2}.csv".format(n, d, k), X, delimiter=";")

def make_my_blog(n, k, d):
    X, y = make_blobs(n_samples=n, centers=k, n_features=d, center_box=(-6, 6), random_state=42)
    return normalize_01(X)

def read_dataset(filename):
    return np.loadtxt(filename, delimiter=";")

def normalize_01(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized
def get_dataset(index):
    ds = datasets[index]
    n, k, d, r = ds
    file = "gen_{0}_{1}_{2}".format(n, d, k)
    X = read_dataset(datafolder + file + ".csv")
    X = normalize_01(X)
    return X, k, file

print_dataset_info()

#generate_blobs()