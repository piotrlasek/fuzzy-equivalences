import numpy as np
from sklearn.datasets import make_blobs

arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

u = np.unique(arr, axis=0)

print(u)

X, y = make_blobs(n_samples=n, centers=k, n_features=2, center_box=(-7.0, 7.0), random_state=33)

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X_normalized = (X - X_min) / (X_max - X_min)
X = X_normalized