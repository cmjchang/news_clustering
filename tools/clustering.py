import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV


def km_clustering_selection(df, matrix):
    KM = KMeans(random_state=42)
    classifier = GridSearchCV(KM,
                              {"n_clusters": np.logspace(np.log10(df.shape[0] / 100),
                                                         np.log10(df.shape[0] / 1.5), 5).astype(int)},
                              n_jobs=-1)
    classifier.fit(matrix)
    df['labels'] = classifier.best_estimator_.labels_
    df['distance_point_to_cluster'] = classifier.best_estimator_.transform(matrix).min(axis=1)
    print("inertia {}".format(classifier.best_estimator_.inertia_))
    print("Number of Cluster {}".format(df['labels'].nunique()))


def km_clustering_selection_sil_score(df, matrix):
    list_cluster_trial = np.logspace(np.log10(df.shape[0] / 100), np.log10(df.shape[0] / 1.5), 100).astype(int)
    list_cluster_trial = range(50, 201)

    sil_score_list = []
    for ncluster in list_cluster_trial:
        kmean = KMeans(n_clusters=ncluster, n_jobs=-1, random_state=42)
        kmean.fit(matrix)

        sil_score = silhouette_score(matrix, kmean.labels_)
        print('number of clusters: {} and corresponding silhouette score: {:.3f}'.format(ncluster, sil_score))
        sil_score_list.append(sil_score)

    sil_score_df = pd.DataFrame()
    sil_score_df['number_cluster'] = list_cluster_trial
    sil_score_df['silhouette_score'] = sil_score_list
    return sil_score_df


def km_clustering(df, matrix, ncluster):
    KM = KMeans(n_clusters=ncluster, n_jobs=-1, random_state=42)
    KM.fit(matrix)
    df['labels'] = KM.labels_
    df['distance_point_to_cluster'] = KM.transform(matrix).min(axis=1)


def clustering(df, matrix):
    from sklearn.cluster import AffinityPropagation
    from sklearn import metrics
    AP = AffinityPropagation()
    AP.fit(matrix)
    df['labels'] = AP.labels_


def gini(v):
    bins = np.linspace(0., 100., 11)
    total = float(np.sum(v))
    yvals = []
    for b in bins:
        bin_vals = v[v <= np.percentile(v, b)]
        bin_fraction = (np.sum(bin_vals) / total) * 100.0
        yvals.append(bin_fraction)
    # perfect equality area
    pe_area = np.trapz(bins, x=bins)
    # lorenz area
    lorenz_area = np.trapz(yvals, x=bins)
    gini_val = (pe_area - lorenz_area) / float(pe_area)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(bins, yvals, label="observed")
    plt.plot(bins, bins, '--', label="perfect eq.")
    plt.xlabel("fraction of population")
    plt.ylabel("fraction of wealth")
    plt.title("GINI: %.4f" % (gini_val))
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.hist(v, bins=20)
    return bins, yvals, gini_val