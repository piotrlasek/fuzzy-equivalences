import os
import traceback

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn_extra.cluster import KMedoids
import generator
import kmeans
import matplotlib as mpl


import numpy as np
from sklearn.datasets import make_blobs

import distances
import kmeans as kmeans
import mysilhouette as silhouette
import distances as dist
import data as data
import pandas as pd
from datetime import datetime
from functools import partial
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score


mpl.rcParams['font.family'] = 'Calibri'

#k = 3
#n = 50
max_iters = 10

dataset_res = {}
dataset_chi = {}
dataset_pva = {}
dataset_ran = {}

distance_names = []
aggregation_names = []

for aggregation in np.array(dist.aggregations)[data.a_tab]:
    agg_name, agg = aggregation

    for distance in np.array(dist.distances)[data.d_tab]:
        dist_fun, dist_name = distance

        distance_names.append(dist_name)
        aggregation_names.append(agg_name)

print(distance_names)

dataset_res["Agg."] = aggregation_names #[[nd ] for fd, nd in distances.distances]
dataset_res["Dist."] = distance_names #[[nd ] for fd, nd in distances.distances]
dataset_chi["Agg."] = aggregation_names #[[nd ] for fd, nd in distances.distances]
dataset_chi["Dist."] = distance_names #[[nd ] for fd, nd in distances.distances]
dataset_pva["Agg."] = aggregation_names #[[nd ] for fd, nd in distances.distances]
dataset_pva["Dist."] = distance_names #[[nd ] for fd, nd in distances.distances]
dataset_ran["Agg."] = aggregation_names #[[nd ] for fd, nd in distances.distances]
dataset_ran["Dist."] = distance_names #[[nd ] for fd, nd in distances.distances]


for i in range(len(data.datasets_med)):
    X, dataset_id, dataset_name, short_name, k_list, dist_list, agg_list, decision = data.get_dataset_med(i)

    print("---------- " + dataset_name + "---------- ")

    dist_res = []

    k_res = {}
    k_chi = {}
    k_pva = {}
    k_ran = {}
    for x in data.k_tab:
        k_res[x] = []
        k_chi[x] = []
        k_pva[x] = []
        k_ran[x] = []

    for agg_id in agg_list:
        agg_name, agg = dist.aggregations[agg_id]
        for dist_id in dist_list:
            distance, distance_name = dist.distances[dist_id]
            distance_with_agg = partial(distance, agg=agg)

            for k in k_list:

                suffix = dataset_name + "_" + str(agg_id) + "_" + str(dist_id) + "_" + str(k)
                file_path_labels = "serialized/" + suffix + "_labels"  + ".xlsx"
                file_path_matrix = "serialized/" + suffix + "_matrix" + ".npy"


                # read or save results in file
                if os.path.exists(file_path_matrix):
                    dist_matrix = np.load(file_path_matrix)
                else:
                    dist_matrix = squareform(pdist(X, metric=lambda u, v: distance_with_agg(u, v)))
                    np.save(file_path_matrix, dist_matrix)

                if os.path.exists(file_path_labels):
                    #print(file_path_labels)
                    labels = pd.read_excel(file_path_labels, engine='openpyxl')
                    labels = labels['Cluster_Labels']
                else:
                    labels, centroid = kmeans.my_kmedoids(X, dist_matrix, k, 5)
                    df_labels = pd.DataFrame(labels, columns=['Cluster_Labels'])
                    df_labels.to_excel(file_path_labels, index=False, engine='openpyxl')

                d = {
                    'decision': decision,
                    'cluster' : labels
                }

                decision_cluster = pd.DataFrame(d)

                # chi-square
                #contingency_table = pd.crosstab(decision_cluster['decision'], decision_cluster['cluster'])
                #chi2, pval, dof, expected = chi2_contingency(contingency_table)

                # person

                ari_score = adjusted_rand_score(decision, labels)
                ari_score = round(ari_score, 2)

                if len(np.unique(labels)) > 1:
                    try:
                        sc = silhouette_score(dist_matrix, labels, metric='precomputed')
                    except Exception:
                        sc = -2
                        traceback.print_exc()
                else:
                    sc = -1

                sc = round(sc, 2)

                cluster_averages = pd.DataFrame(X).groupby(labels).mean()
                sns.heatmap(cluster_averages, cmap='viridis', cbar=False)
                plt.title(short_name + ", " + agg_name + ", " + distance_name + ", $k = " + str(k) + "$, $s = " + str(round(sc, 2)) + "$" , fontsize=18)
                plt.ylabel("")
                #plt.show()
                file_name = "fig/" + str(dataset_id) + "." + short_name.replace("$", "") + "_" + agg_name.replace("$", "") + "_" + distance_name.replace("$", "") + "_" + str(k) + ".png"
                plt.savefig(file_name)

                # print("\t{0}\t\t{1}\t\t{2}\t\t{3}\t\t{4}\t\t{5}".format(agg_name, distance_name, k, round(sc, 2), round(chi2, 1), round(pval, 2)))
                print("\t{0: <10}\t{1: <10}\t{2}\t{3}\t{4}".format(agg_name, distance_name, k, round(sc, 2), ari_score))

                sc = round(sc, 2)
                #chi2 = round(chi2, 2)
                #pval = round(pval, 2)
                k_res[k].append(sc)
                #k_chi[k].append(chi2)
                #k_pva[k].append(pval)
                k_ran[k].append(ari_score)


    for k in k_res.keys():
        dataset_res[str(dataset_id) + "." + str(k)] = k_res[k]
        #dataset_chi[str(dataset_id) + "." + str(k)] = k_chi[k]
        #dataset_pva[str(dataset_id) + "." + str(k)] = k_pva[k]
        dataset_ran[str(dataset_id) + "." + str(k)] = k_ran[k]

df_sil = pd.DataFrame(dataset_res)
#df_chi = pd.DataFrame(dataset_chi)
#df_pva = pd.DataFrame(dataset_pva)
df_ran = pd.DataFrame(dataset_ran)

currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%m.%d_%H-%M-%S")

res_file_sil = "results/vis-" + currentTime + "-sil.xlsx"
#res_file_chi = "results/vis-" + currentTime + "-chi.xlsx"
#res_file_pva = "results/vis-" + currentTime + "-pva.xlsx"
res_file_ran = "results/vis-" + currentTime + "-ran.xlsx"

print("Writing to {0}".format(res_file_sil))

df_sil.to_excel(res_file_sil)
#df_chi.to_excel(res_file_chi)
#df_pva.to_excel(res_file_pva)
df_ran.to_excel(res_file_ran)