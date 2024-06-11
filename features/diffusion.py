import numpy as np
import MDAnalysis.analysis.diffusionmap as diffusionmap

from sklearn.metrics import pairwise_distances

import numpy as np
from pydiffmap import diffusion_map as dm
from sklearn.metrics import pairwise_distances


def compute_diffusion_distance_matrix2(X, epsilon='bgh', alpha=0.5, n_evecs=2):
    # Initialize the Diffusion Map object
    mydmap = dm.DiffusionMap.from_sklearn(n_evecs=n_evecs, epsilon=epsilon, alpha=alpha)
    # Fit to data and get the diffusion map embedding
    diffusion_map_embedding = mydmap.fit_transform(X)

    # Compute pairwise distances between diffusion map embeddings
    diffusion_distance_matrix = pairwise_distances(diffusion_map_embedding)

    return diffusion_distance_matrix


def modify_distance_matrix_with_cannot_link2(X, cannot_links):
    # X - data matrix
    # cannot_links - list of tuples (p, q) representing cannot-link constraints
    # diffusion_distance_matrix - precomputed diffusion distance matrix

    diffusion_distance_matrix = compute_diffusion_distance_matrix2(X)

    M = X.shape[0]  # Number of data points
    N = len(cannot_links)  # Number of cannot-link constraints

    # Initialize new feature space matrix
    V = np.zeros((M, N))

    # Compute new feature values based on cannot-link constraints
    for c, (p, q) in enumerate(cannot_links):
        for i in range(M):
            if i == p:
                V[i, c] = -1
            elif i == q:
                V[i, c] = 1
            else:
                V[i, c] = (diffusion_distance_matrix[i, q] - diffusion_distance_matrix[i, p]) / \
                          (diffusion_distance_matrix[i, q] + diffusion_distance_matrix[i, p])

    # Compute the new distance matrix using the modified feature space
    modified_distance_matrix = pairwise_distances(V)

    return modified_distance_matrix






# Example usage:
# X = np.array(...)  # Your data matrix
# cannot_links = [(0, 1), (2, 3)]  # Example cannot-link constraints
# modified_distance_matrix = modify_distance_matrix_with_cannot_link(X, cannot_links, epsilon='bgh')


def compute_diffusion_matrix(D, sigma=1.0):
    # Oblicz macierz dyfuzji z macierzy odległości i parametru sigma
    # D - macierz odległości między punktami
    # sigma - parametr rozmycia jądra Gaussa
    return np.exp(-(D ** 2) / (2 * sigma ** 2))

def calculate_diffusion_matrix(X, sigma):
    # X - macierz punktów
    # sigma - waga funkcji jądra gaussowskiego

    # Obliczenie funkcji jądra gaussowskiego dla wszystkich par punktów
    A = np.exp(-pairwise_distances(X) ** 2 / sigma ** 2)

    # Obliczenie normalizacji di dla każdego punktu
    d = A.sum(axis=1)

    # Obliczenie macierzy przejścia P
    P = A / d[:, np.newaxis]

    return P

def modify_distance_matrix_with_cannot_link(X, cannot_links, D):
    # X - macierz punktów
    # cannot_links - lista par indeksów punktów, między którymi istnieje ograniczenie cannot-link
    # D - macierz odległości między punktami

    # Ilość ograniczeń cannot-link
    N = len(cannot_links)
    # Ilość punktów
    M = X.shape[0]

    # Oblicz macierz dyfuzji z macierzy odległości
    Dt = compute_diffusion_matrix(D, 1)
    #Dt = calculate_diffusion_matrix(D, 1)

    # Nowa macierz cech, która będzie zawierać oryginalne cechy rozszerzone o nowe
    V = np.zeros((M, N))

    # Dla każdego ograniczenia cannot-link dodajemy nową cechę
    for c, (p, q) in enumerate(cannot_links):
        # Obliczenie nowej cechy dla punktów p i q
        V[p, c] = 0
        V[q, c] = 1
        # Obliczenie nowej cechy dla pozostałych punktów
        for i in range(M):
            if i != p and i != q:
                V[i, c] = (Dt[i, q] - Dt[i, p]) / (Dt[i, q] + Dt[i, p])

    # Obliczenie zmodyfikowanej macierzy odległości
    D_modified = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            D_modified[i, j] = np.linalg.norm(V[i] - V[j])

    return D_modified


import numpy as np


def modify_distance_matrix_with_relative_constraints(X, relative_constraints, Din):
    # X - macierz punktów
    # relative_constraints - lista trójek indeksów punktów (a, b, c), gdzie a i b są bliżej siebie niż do c
    # D - macierz odległości między punktami

    # Ilość ograniczeń relacyjnych
    N = len(relative_constraints)
    # Ilość punktów
    M = X.shape[0]

    # Nowa macierz cech, która będzie zawierać oryginalne cechy rozszerzone o nowe
    V = np.zeros((M, N))

    D = compute_diffusion_matrix(Din, 1)
    #D = calculate_diffusion_matrix(Din, 3)

    # Obliczenie największej wartości odległości w macierzy D
    alpha = np.max(D)

    # Dla każdego ograniczenia relacyjnego dodajemy nową cechę
    for r, (a, b, c) in enumerate(relative_constraints):
        # Obliczenie nowej cechy dla każdego punktu X
        for i in range(M):
            V[i, r] = (min(D[i, a], D[i, b]) - D[i, c]) / (min(D[i, a], D[i, b]) + D[i, c])

    # Obliczenie macierzy odległości D(r) dla każdego ograniczenia relacyjnego
    D_r = np.zeros((M, M, N))
    for r in range(N):
        for i in range(M):
            for j in range(M):
                D_r[i, j, r] = np.abs(V[i, r] - V[j, r])

    # Obliczenie końcowej macierzy odległości Df
    Df = D.copy()
    for r in range(N):
        Df += (alpha * D_r[:, :, r]) ** 2

    return Df





def diffusion_distance(P, t):
    # P - macierz przejścia
    # t - liczba kroków

    # Obliczenie macierzy P do potęgi t
    Pt = np.linalg.matrix_power(P, t)

    # Obliczenie kwadratu odległości dyfuzyjnej
    D_square_diffusion = np.sum((Pt - Pt.T) ** 2, axis=1)

    # Obliczenie odległości dyfuzyjnej
    D_diffusion = np.sqrt(D_square_diffusion)

    return D_diffusion

# Przykład użycia:
# X - macierz twoich punktów danych
# sigma - waga funkcji jądra gaussowskiego
# t - liczba kroków do obliczenia odległości dyfuzyjnej

# P = calculate_diffusion_matrix(X, sigma)
# D_diffusion = diffusion_distance(P, t)

