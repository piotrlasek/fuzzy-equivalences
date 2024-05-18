import numpy as np

def generate_idx(t, k):
    if k == 0:
        raise ValueError("k must be greater than 0")

    index_step = len(t) / k  # Calculate the step between each index based on the table size and k
    return [int(index_step * i) for i in range(k)]

def my_kmeans(X, k, distance_func, max_iters=100):
    # Initialize centroids randomly from the dataset
    indices = np.random.choice(len(X), k, replace=False)
    #indices = generate_idx(X, k)
    #print(indices)

    centroids = X[indices]

    for iteration in range(max_iters):
        # Assign clusters based on the custom distance function
        clusters = np.array([np.argmin([distance_func(x, centroid) for centroid in centroids]) for x in X])
        # Update centroids
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters, centroids



def my_kmedoids(X, distance_matrix, k, max_iters=10):
    num_points = distance_matrix.shape[0]

    # Initialize medoids randomly from the dataset
    #indices = np.random.choice(num_points, k, replace=False)
    indices = generate_idx(distance_matrix[:, 0], k)

    for iteration in range(max_iters):
        # Assign clusters based on the minimum distance to the medoids
        clusters = np.array([np.argmin([distance_matrix[i, idx] for idx in indices]) for i in range(num_points)])

        new_indices = []
        for i in range(k):
            cluster_points = np.where(clusters == i)[0]
            if len(cluster_points) > 0:
                # Compute the sum of distances within the cluster for each point
                intra_cluster_distances = distance_matrix[cluster_points][:, cluster_points].sum(axis=1)
                # Choose the point with the minimum sum of distances as the new medoid
                new_medoid_idx = cluster_points[np.argmin(intra_cluster_distances)]
                new_indices.append(new_medoid_idx)
            else:
                # If a cluster is empty, choose a random new point as a medoid
                new_indices.append(np.random.choice(num_points))

        # Check for convergence (if medoids do not change)
        if np.array_equal(indices, new_indices):
            break
        indices = np.array(new_indices)

    # Return the cluster assignments and the coordinates of the medoids
    medoid_coordinates = X[indices]
    return clusters, medoid_coordinates


def my_kmeans_dist(distance_matrix, k, max_iters=10):
    # Number of points based on the distance matrix
    num_points = distance_matrix.shape[0]

    # Initialize centroids randomly from the dataset
    #indices = np.random.choice(num_points, k, replace=False)
    indices = generate_idx(distance_matrix[:,0], k)

    for iteration in range(max_iters):
        # Assign clusters based on the minimum distance in the distance matrix
        clusters = np.array([np.argmin([distance_matrix[i, idx] for idx in indices]) for i in range(num_points)])

        # Since we cannot directly calculate the mean of points in the original space,
        # we update centroids by selecting a new point from the cluster that minimizes
        # the sum of distances to all other points in the cluster (a medoid).
        new_indices = []
        for i in range(k):
            cluster_points = np.where(clusters == i)[0]
            if len(cluster_points) > 0:
                # Compute the sum of distances within the cluster for each point
                intra_cluster_distances = distance_matrix[cluster_points][:, cluster_points].sum(axis=1)
                # Choose the point with the minimum sum of distances as the new centroid
                new_centroid_idx = cluster_points[np.argmin(intra_cluster_distances)]
                new_indices.append(new_centroid_idx)
            else:
                # If a cluster is empty, choose a random new point as a centroid
                new_indices.append(np.random.choice(num_points))

        # Check for convergence (if centroids do not change)
        if np.array_equal(indices, new_indices):
            break
        indices = np.array(new_indices)

    return clusters, indices


def my_kmeans_sep(X, k, distance_func, max_iters=100):
    # Initialize centroids by randomly selecting within the bounds of the data-xx-old
    min_bounds = X.min(axis=0)
    max_bounds = X.max(axis=0)
    centroids = np.array([np.random.uniform(low=min_bounds, high=max_bounds) for _ in range(k)])

    for iteration in range(max_iters):
        # Assign clusters based on the custom distance function
        clusters = np.array([np.argmin([distance_func(x, centroid) for centroid in centroids]) for x in X])

        # Compute new centroids as the mean of the assigned data-xx-old points, not as the data-xx-old points themselves
        new_centroids = np.array([X[clusters == i].mean(axis=0) if len(X[clusters == i]) > 0 else centroids[i] for i in range(k)])

        # Check for convergence (using an allclose comparison rather than equality to account for floating point precision)
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    return clusters, centroids
