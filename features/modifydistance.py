import numpy as np

def modify_distance_with_cannot_link(D, cannot_links):
    # D is the initial distance matrix
    # cannot_links is a list of tuples (p, q)
    n = D.shape[0]
    D_modified = D.copy()
    for p, q in cannot_links:
        for i in range(n):
            for j in range(i + 1, n):
                vi = (D[i, q] - D[i, p]) / (D[i, q] + D[i, p])
                vj = (D[j, q] - D[j, p]) / (D[j, q] + D[j, p])
                D_modified[i, j] = D_modified[j, i] = np.abs(vi - vj)
    return D_modified

def modify_distance_with_relative(D, relatives):
    # D is the initial distance matrix
    # relatives is a list of tuples (a, b, c)
    n = D.shape[0]
    D_modified = D.copy()
    for a, b, c in relatives:
        for i in range(n):
            for j in range(i + 1, n):
                vi = (min(D[i, a], D[i, b]) - D[i, c]) / (min(D[i, a], D[i, b]) + D[i, c])
                vj = (min(D[j, a], D[j, b]) - D[j, c]) / (min(D[j, a], D[j, b]) + D[j, c])
                D_modified[i, j] = D_modified[j, i] = np.abs(vi - vj)
    return D_modified