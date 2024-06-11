import numpy as np
from ucimlrepo import fetch_ucirepo, list_available_datasets
import pandas as pd

datafolder = "data/"

datasets = [
     ("fertility.csv", "Fertility", 244, 2),
     ("iris.csv", "Iris", 53, 3),
     # ("zoo.csv", "Zoo", 111, 7),                 # same zera i jedynki
     ("ecoli.csv", "Ecoli", 39, 8),
     ("glass.csv", "Glass Identification", 42, 6),
     #("leaf.csv", "Leaf", -1, 30),
     #("led.csv", "LED Display Domain", -1, 10),
     ("parkinson.csv", "Parkinson Dataset", -1, 2),
     ("speaker.csv", "Speaker accent", -1, 6),
     ("german.csv", "Statlog (German Credit Data)", 144, 2),
     ("banknote.csv", "Banknote Authentication", 267, 2),
     #("wine.csv", "Wine Quality", 186, 7),
]

k_tab = [2, 3, 4, 5, 6, 7, 8]
a_tab = [0, 1, 2, 3, 4]
d_tab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
#
k_tab = [2, 3]
#a_tab = [0, 1, 2, 3, 4]
#d_tab = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

datasets_med = [
    ("data/_BC_Coimbra_iloraz.csv", 1, "Breast Cancer Coimbra", "BCC", ",", ["Classification"], k_tab, d_tab, a_tab),
    ("data/_BC_Wisconsin_iloraz.csv", 2, "Breast Cancer Wisconsin", "BCW", ",", ["Diagnosis"], k_tab, d_tab, a_tab),
    ("data/_hcv_iloraz.csv", 3, "HCV", "HCV", ",", ["Category"], k_tab, d_tab, a_tab),
    ("data/_ILPD_iloraz.csv", 4, "Indian Liver Patient", "ILP", ",", ["Selector"], k_tab, d_tab, a_tab),
    ("data/_ChKDfull_iloraz_normal3.csv", 5, "Chronic Kidney Disease", "CKD", ",", ["class"], k_tab, d_tab, a_tab),
]

# selected

def get_dataset_med(i):
    file, id, name, short_name, delim, decision_col, k, dist, agg = datasets_med[i]
    X = pd.read_csv(file, sep=delim)
    decision = X[decision_col].to_numpy()
    decision = decision.flatten()
    # decision = pd.factorize(decision)

    X = X.drop(decision_col, axis=1)
    X = X.to_numpy()

    #X = X.iloc[:, :-1].values  # Convert all columns except the last to a NumPy array
    #y = X.iloc[:, -1].values  # Convert the last column to a NumPy array (labels)
    X = normalize_01(X)
    return X, id, name, short_name, k, dist, agg, decision

def print_dataset_info():
    print("{0}\t{1}\t{2}\t{3}".format("Dataset name", "Cnt.", "Dim.", "Cat."))

    for id in range(len(datasets)):
        X, k, name = get_dataset(id)
        print("{0}\t{1}\t{2}\t{3}".format(name,X.shape[0], X.shape[1], k))

def save_dataset(arr, filename):
    np.savetxt(filename, arr, delimiter=";")

def read_dataset(filename):
    return np.loadtxt(filename, delimiter=";")

def get_dataset(index):
    dataset = datasets[index]
    file, name, id, k = dataset
    X = read_dataset(datafolder + file)
    X = normalize_01(X)
    return X, k, name

def normalize_01(X):
    X_min = X.min(axis=0)  # Find the minimum value in each column
    X_max = X.max(axis=0)  # Find the maximum value in each column
    X_normalized = (X - X_min) / (X_max - X_min)
    return X_normalized
