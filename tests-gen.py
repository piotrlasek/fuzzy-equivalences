import numpy as np
from sklearn.datasets import make_blobs

import distances
import generator
import kmeans as kmeans
import mysilhouette as silhouette
import distances as dist
import data as data
import pandas as pd
from datetime import datetime
from functools import partial
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import pdist, squareform

#k = 3
#n = 50
max_iters = 5

dataset_res = {}

distance_names = []
aggregation_names = []

for aggregation in dist.aggregations:
    agg_name, agg = aggregation

    for distance in dist.distances:
        dist_fun, dist_name = distance

        distance_names.append(dist_name)
        aggregation_names.append(agg_name)

print(distance_names)

dataset_res["Agg."] = aggregation_names #[[nd ] for fd, nd in distances.distances]
dataset_res["Dist."] = distance_names #[[nd ] for fd, nd in distances.distances]

for i in range(len(generator.datasets)):
    X, k, dataset_name = generator.get_dataset(i)

    print("---------- " + dataset_name + " (" + str(k) + ") ---------- ")

    dist_res = []

    for aggregation in dist.aggregations:

        agg_name, agg = aggregation

        #if agg_name ==  "$4$":

        for distance_tuple in dist.distances:

            distance, distance_name = distance_tuple

            #if distance_name == "$E_{LK}$":

            distance_with_agg = partial(distance, agg=agg)

            #labels, centroids = kmeans.my_kmeans(X, k, distance_with_agg, max_iters=max_iters)
            #sc = silhouette.silhouette_score_custom_distance(X, labels, distance_with_agg)

            dist_matrix = squareform(pdist(X, metric=lambda u, v: distance_with_agg(u, v)))

            labels, centroid = kmeans.my_kmedoids(X, dist_matrix, k, 5)

            if len(np.unique(labels)) > 1:
                try:
                    sc = silhouette_score(dist_matrix, labels, metric='precomputed')
                except:
                    sc = -2
            else:
                sc = -1

            print("\t{0}\t\t{1}\t\t{2}".format(agg_name, distance_name, round(sc, 3)))
            sc = round(sc, 3)

            dist_res.append(sc)

        dataset_res[dataset_name] = dist_res

df = pd.DataFrame(dataset_res)
print(df)

currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%H%M%S")

res_file = "results/gen-" + currentTime + ".xlsx"

print("Writing to {0}".format(res_file))

df.to_excel(res_file)