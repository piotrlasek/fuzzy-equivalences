import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

def plot_matrices(m1, m2, m3):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Original distance matrix
    ax[0].imshow(m1, cmap='viridis')
    ax[0].set_title('m1')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Index')

    # Inverted distance matrix
    ax[1].imshow(m2, cmap='viridis')
    ax[1].set_title('m2')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Index')

    # Squared distance matrix
    ax[2].imshow(m3, cmap='viridis')
    ax[2].set_title('m3')
    ax[2].set_xlabel('Index')
    ax[2].set_ylabel('Index')

    plt.tight_layout()
    plt.show()


def plot_clusters(X, cannot_links, relative_constraints, kmedoids1, kmedoids2, kmedoids3, s1, s2, s3):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

    # Original distance matrix
    axs[0].scatter(X[:, 0], X[:, 1], c=kmedoids1.labels_, cmap='viridis', edgecolor='k', alpha=0.7)
    #axs[0].scatter(X[kmedoids1.medoid_indices_, 0], X[kmedoids1.medoid_indices_, 1], c='red', marker='o', s=100, label='Medoid Centers')
    #axs[0].set_title('KMedoids')
    axs[0].text(0, 0, 's = ' + str(round(s1, 3)), fontsize=10, verticalalignment='top')

    axs[1].scatter(X[:, 0], X[:, 1], c=kmedoids2.labels_, cmap='viridis', edgecolor='k', alpha=0.7)
    #axs[1].scatter(X[kmedoids2.medoid_indices_, 0], X[kmedoids2.medoid_indices_, 1], c='red', marker='o', s=100,
    #               label='Medoid Centers')
    for (i, j) in cannot_links:
        axs[1].plot([X[i, 0], X[j, 0]], [X[i, 1], X[j, 1]], 'r--')
    #axs[1].set_title('KMedoids with Cannot-Link Constraints')
    axs[1].text(0, 0, 's = ' + str(round(s2, 3)), fontsize=10, verticalalignment='top')

    axs[2].scatter(X[:, 0], X[:, 1], c=kmedoids3.labels_, cmap='viridis', edgecolor='k', alpha=0.7)
    #axs[2].scatter(X[kmedoids3.medoid_indices_, 0], X[kmedoids3.medoid_indices_, 1], c='red', marker='o', s=100, label='Medoid Centers')
    for (a, b, c) in relative_constraints:
        #axs[2].plot(X[[a, b, c, a], 0], X[[a, b, c, a], 1], 'k-', alpha=0.5)  # Draw a triangle for each constraint
        axs[2].plot([X[a, 0], X[b, 0]], [X[a, 1], X[b, 1]], 'g--')
        axs[2].plot([X[a, 0], X[c, 0]], [X[a, 1], X[c, 1]], 'b--')
    #axs[2].set_title('KMedoids with Relative Constraints')
    axs[2].text(0, 0, 's = ' + str(round(s3, 3)), fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()
