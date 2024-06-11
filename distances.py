import numpy as np
import warnings

warnings.simplefilter("ignore", category=RuntimeWarning)

def E(x1, x2, agg):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def d_E_LK(u, v, agg):
    differences = 1 - np.abs(u - v)
    distance = 1 - agg(differences)
    return distance

def d_R_E_LK(u, v, agg):
    differences = 1 - np.abs(u - v)
    differences = differences * differences
    distance = 1 - agg(differences)
    return distance

def d_E_GG(x, y, agg):
    result = np.where(x == y, 1, np.where(x < y, x / y, y / x))
    result = 1 - agg(result)
    return result

def d_E_GD(x, y, agg):
    result = np.where(x == y, 1, np.where(x < y, x, y))
    result = 1 - agg(result)
    return result

def d_R_E_GD(x, y, agg):
    result = np.where(x == y, 1, np.where(x < y, x / y, y / x))
    result = result * (2 - result)
    result = 1 - agg(result)
    return result

def d_E_FD(x, y, agg):
    result = np.where(x == y, 1, np.where(x < y, np.maximum(1-y, x), np.maximum(1-x, y)))
    result = 1 - agg(result)
    return result

def d_R_E_FD(x, y, agg):
    # Calculate E_FD first
    e_fd = np.where(x == y, 1, np.where(x < y, np.maximum(1-y, x), np.maximum(1-x, y)))
    # Then use it to calculate E_R_FD
    result = np.where(e_fd >= 0.5, 1 - 2 * (1 - e_fd)**2, 2 * e_fd**2)
    result = 1 - agg(result)
    return result

def d_E_3(x, y, agg):
    result = np.where(x + y == 0, 0, (2 * np.minimum(x, y)) / (x + y))
    result = 1 - agg(result)
    return result

def d_E_4(x, y, agg):
    result =  np.where(x + y == 0, 0, (2 * x * y) / (x**2 + y**2))
    result = 1 - agg(result)
    return result

def d_E_5(x, y, agg):
    result =  np.where(x + y == 0, 0, (2 * np.minimum(x**2, y**2)) / (x**2 + y**2))
    result = 1 - agg(result)
    return result

def d_E_6(x, y, agg):
    result =  (1 - np.abs(x - y)) / (1 + np.abs(x - y))
    result = 1 - agg(result)
    return result

# E_LK, E^R_LK, E_GD, E^R_GD, E3 - E6
# 1, 2, 4, 5, 8, 9, 10, 11
distances = [
    (E, "$Euclid.$"),               # 0
    (d_E_LK, "$E_{LK}$"),           # 1
    (d_R_E_LK, "$E_{LK}^{R}$"),     # 2
    (d_E_GG, "$E_{GG}$"),           # 3
    (d_E_GD, "$E_{GD}$"),           # 4
    (d_R_E_GD, "$E_{GD}^{R}$"),     # 5
    (d_E_FD, "$E_{FD}$"),           # 6
    (d_R_E_FD, "$E_{FD}^R$"),       # 7
    (d_E_3, "$E3$"),                # 8
    (d_E_4, "$E4$"),                # 9
    (d_E_5, "$E5$"),                # 10
    (d_E_6, "$E6$"),                # 11
]


def power_root(input, p):
    return np.power(np.mean(np.power(input, p)), 1 / p)
def A2_01(input):
    p = 0.1
    return power_root(input, p)

def A2_1(input):
    p = 1
    return power_root(input, p)

def A2_2(input):
    p = 2
    return power_root(input, p)

def A2_4(input):
    p = 4
    return power_root(input, p)


'''
A1      arithmetic mean
A2      power-root mean, where p > 0
A3      minimum
A4      maximum
'''


# A2_0.1, A2_1, A2_4, A3, A4
# 0, 1, 2, 3, 4


aggregations = [
    ("$2_{0.1}$", A2_01),                   # 0
    ("$2_1$",     np.mean), # arithmetic mean  # 1
    ("$2_{4}$",   A2_4),                     # 2
    ("$3$",       np.min),                   # 3
    ("$4$",       np.max),                   # 4
    ("$2_2$",     A2_2),                     # 5
]