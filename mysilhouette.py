import numpy as np
from scipy.spatial.distance import pdist, squareform

import numpy as np
from scipy.spatial.distance import pdist, squareform

def silhouette_score_custom_distance(X, labels, distance_function):
    # Calculating the distance matrix
    distance_matrix = squareform(pdist(X, metric=distance_function))

    silhouette_scores = []
    for i in range(len(X)):
        # Distances from points in the same cluster
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude the point itself

        if np.any(same_cluster):  # Ensure there are other points in the cluster
            a = np.mean(distance_matrix[i, same_cluster])
        else:  # If there are no other points in the cluster, handle appropriately
            a = 0

        # Distances from points in other clusters
        other_cluster_scores = []
        for cluster in set(labels):
            other_cluster = labels == cluster
            if cluster != labels[i] and np.any(other_cluster):  # Ensure the cluster has points and is not the same cluster
                other_cluster_score = np.mean(distance_matrix[i, other_cluster])
                if not np.isnan(other_cluster_score):  # Exclude NaN values
                    other_cluster_scores.append(other_cluster_score)

        # Filter out NaN values from other_cluster_scores
        filtered_scores = [score for score in other_cluster_scores if not np.isnan(score)]

        if filtered_scores:  # Proceed only if there are other clusters with points
            b = min(filtered_scores)
            # Calculating the silhouette score for a point, ensuring no division by zero
            s = (b - a) / max(a, b) if max(a, b) > 0 else 0
            silhouette_scores.append(s)
        else:  # Handle cases where a point is in its own unique cluster or only NaN values were produced
            silhouette_scores.append(0)

    # Averaging silhouette scores for all points
    overall_silhouette_score = np.mean(silhouette_scores)
    return overall_silhouette_score


# Example usage
# X = np.array([...])  # Your data
# labels = np.array([...])  # Cluster labels for each point in X
# distance_function = 'euclidean'  # or any custom distance function compatible with pdist
# print(silhouette_score_custom_distance(X, labels, distance_function))


'''```
def silhouette_score_custom_distance(X, labels, distance_function):

    # Calculating the distance matrix
    distance_matrix = squareform(pdist(X, metric=distance_function))

    silhouette_scores = []
    for i in range(len(X)):
        # Distances from points in the same cluster
        same_cluster = labels == labels[i]
        same_cluster[i] = False
        a = np.mean(distance_matrix[i, same_cluster])

        # Distances from points in other clusters
        other_cluster_scores = []
        for cluster in set(labels):
            if cluster != labels[i]:
                other_cluster = labels == cluster
                other_cluster_score = np.mean(distance_matrix[i, other_cluster])
                other_cluster_scores.append(other_cluster_score)
        b = min(other_cluster_scores)

        # Calculating the silhouette coefficient for a point
        s = (b - a) / max(a, b)
        silhouette_scores.append(s)

    # Averaging silhouette coefficients for all points
    overall_silhouette_score = np.mean(silhouette_scores)
    print(">>>>>> " + str(overall_silhouette_score))
    return overall_silhouette_score

'''

'''def silhouette_score_custom_distance(X, labels, distance_function):
    # Calculating the distance matrix
    distance_matrix = squareform(pdist(X, metric=distance_function))

    silhouette_scores = []
    for i in range(len(X)):
        # Distances from points in the same cluster
        same_cluster = labels == labels[i]
        same_cluster[i] = False  # Exclude the point itself

        if np.sum(same_cluster) == 0:  # If there are no other points in the cluster
            a = 0
        else:
            a = np.mean(distance_matrix[i, same_cluster])

        # Distances from points in other clusters
        other_cluster_scores = []
        for cluster in set(labels):
            other_cluster = labels == cluster
            if cluster != labels[i] and np.sum(other_cluster) > 0:
                other_cluster_score = np.mean(distance_matrix[i, other_cluster])
                other_cluster_scores.append(other_cluster_score)

        if other_cluster_scores:  # Only proceed if there are other clusters with points
            b = min(other_cluster_scores)
            # Calculating the silhouette coefficient for a point
            s = (b - a) / max(a, b) if a > 0 else 0  # Handle division by zero if a is 0
            silhouette_scores.append(s)
        else:  # If this point is the only point across all clusters, silhouette score is 0
            silhouette_scores.append(0)

    # Averaging silhouette coefficients for all points
    overall_silhouette_score = np.mean(silhouette_scores)
    return overall_silhouette_score
'''