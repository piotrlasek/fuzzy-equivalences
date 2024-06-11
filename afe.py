import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
def d_E_LK(u, v):
    differences = 1 - np.abs(u - v)
    distance = 1 - np.mean(differences)
    return distance


def d_E_GG(x, y):
    result = np.where(x == y, 1, np.where(x < y, x / y, y / x))
    print("result    " + str(result))
    result = np.mean(result)
    print("mean      " + str(result))

    result = 1 - np.mean(result)
    print("1 - mean  " + str(result))

    return result

def d_E_FD(x, y):
    result = np.where(x == y, 1, np.where(x < y, np.maximum(1-y, x), np.maximum(1-x, y)))
    result = 1 - np.mean(result)
    return result

def d_R_E_FD(x, y):
    # Calculate E_FD first
    e_fd = np.where(x == y, 1, np.where(x < y, np.maximum(1-y, x), np.maximum(1-x, y)))
    # Then use it to calculate E_R_FD
    result = np.where(e_fd >= 0.5, 1 - 2 * (1 - e_fd)**2, 2 * e_fd**2)
    result = 1 - np.mean(result)
    return result

# Example usage:
# Define two sample rows from a numpy array
row_x = np.array([1, 2, 3])
row_y = np.array([3, 2, 1])


X = np.array([[0, 0],  # First sample
             [1.0, 1.0],  # Second sample
             [3.0, 1.0],  # Second sample
             [5.0, 6.0],
              [10, 10]

              ]) # Third sample

X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min)

print(X)

print("0: " + str(X[0]))
print("1: " + str(X[1]))
print("2: " + str(X[2]))

euk = euclidean_distance(X[1], X[2])
elk = d_E_GG(X[1], X[2])
egg = d_E_LK(X[1], X[2])

efd = d_E_FD(X[1], X[2])
refd = d_R_E_FD(X[1], X[2])

print("------------")

print(efd)
print(refd)


print("------------")