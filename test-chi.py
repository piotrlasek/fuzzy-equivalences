# import pandas as pd
# from scipy.stats import chi2_contingency
#
# d = {
#     'decision': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2],
#     'cluster' : [3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 1,
#        1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 4, 1, 1, 1, 1, 0, 1, 1, 2, 0, 2,
#        2, 2, 2, 0, 2, 0, 2, 0, 3, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 3, 3, 3, 4, 3, 0, 3, 1, 2, 1, 4, 4, 1, 2, 4, 0, 1, 4, 1, 4,
#        4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 0, 1, 0, 2, 2, 1, 2, 1, 1, 2, 2,
#        2, 1, 3, 2, 3, 2]
# }
#
# d = pd.DataFrame(d)
#
# pearson = d['decision'].corr(d['cluster']) # = df['x'].corr(df['y'])
#
# print(pearson)


from sklearn.metrics import adjusted_rand_score

# Assume 'labels_true' and 'labels_pred' are your true labels and predicted cluster labels, respectively.
labels_true = ["10", 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
labels_pred = [  20, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1]

# Calculating ARI
ari_score = adjusted_rand_score(labels_true, labels_pred)
print("Adjusted Rand Index:", ari_score)
