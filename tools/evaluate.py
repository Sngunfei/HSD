# -*- coding:utf-8 -*-

"""
Evluate the performance of embedding via different methods.
"""

import math

import numpy as np
from sklearn import metrics
from sklearn import utils as sktools
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

def cluster_evaluate(embeddings, labels, n_class, metric="euclidean"):
    """
        Unsupervised setting: We assess the ability of each method to embed close together nodes
        with the same ground-truth structural role. We use agglomerative clustering (with single linkage)
        to cluster embeddings learned by each method and evaluate the clustering quality via:
            (1) homogeneity, conditional entropy of ground-truth structural roles given the predicted clustering;
            (2) completeness, a measure of how many nodes with the same ground-truth structural role are assigned to the same cluster;
            (3) silhouette score, a measure of intra-cluster distance vs. inter-cluster distance.

        Supervised setting: We assess the performance of learned embeddings for node classifcation.
        Using 10-fold cross validation, we predict the structural role (label) of each node in the test set
        based on its 4-nearest neighbors in the training set as determined by the embedding space.
        The reported score is then the average accuracy and F1-score over 25 trials.
    """
    clusters = AgglomerativeClustering(n_clusters=n_class, linkage='single', affinity=metric).fit_predict(embeddings)
    h, c, v = metrics.homogeneity_completeness_v_measure(labels, clusters)
    s = metrics.silhouette_score(embeddings, clusters)
    acc = accuracy_score(labels, clusters)
    macro_f1 = f1_score(labels, clusters, average="macro")
    print("cluster:", clusters, "labels:", labels)
    print("accuracy: ", acc)
    print("macro_score: ", macro_f1)
    print("homogeneity: ", h)
    print("completeness: ", c)
    print("v-score: ", v)
    print("silhouette: ", s)

    return h, c, v, s


def LR_evaluate(data, labels, cv=5):
    """
    Evaluate embedding effect using Logistic Regression. Mode = One vs Rest (OVR)
    """
    data, labels = sktools.shuffle(data, labels)
    lr = LogisticRegression(solver="lbfgs", penalty='l2', max_iter=1000, multi_class='ovr')
    test_scores = cross_val_score(lr, data, y=labels, cv=cv)
    print(f"LR: test scores={test_scores}, mean_score={np.mean(test_scores)}\n")
    return np.mean(test_scores)


def KNN_evaluate(data, labels, metric="minkowski", cv=5, n_neighbor=10):
    """
    基于节点的相似度进行KNN分类，在嵌入之前进行，为了验证通过层次化相似度的优良特性。
    """
    data, labels = sktools.shuffle(data, labels)
    knn = KNeighborsClassifier(weights='uniform', algorithm="auto", n_neighbors=n_neighbor, metric=metric, p=2)
    test_scores = cross_val_score(knn, data, y=labels, cv=cv, scoring="accuracy")
    print(f"KNN: test scores:{test_scores}, mean_score={np.mean(test_scores)}\n")
    return np.mean(test_scores)


def evalute_results(labels: list, preds: list):
    accuracy = accuracy_score(labels, preds)
    balanced_accuracy = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="micro")
    recall = recall_score(labels, preds, average="micro")
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")
    report = classification_report(labels, preds, digits=7)

    res = { "accuracy": accuracy,
            "balanced accuracy": balanced_accuracy,
            "micro precision": precision,
            "micro recall": recall,
            "macro f1": macro_f1,
            "micro f1": micro_f1,
            "report": report
           }

    print(res)
    return res


def spectral_cluster_evaluate(data, labels, n_cluster, affinity="rbf"):
    """

    :param data: 相似度矩阵 or 嵌入向量
    :param n_cluster:
    :param affinity: precomputed || rbf
    :return:
    """
    metric = "euclidean"
    if affinity == "precomputed":
        # sklearn指导，如果data是距离矩阵而不是相似度矩阵，则可以用下面的rbf转换一下
        distance_mat = data
        delta = math.sqrt(2)
        data = np.exp(-distance_mat ** 2 / (2. * delta ** 2))
        metric = affinity

    clustering = SpectralClustering(n_clusters=n_cluster, affinity=affinity, n_init=50, random_state=42)
    preds = clustering.fit_predict(data)
    h, c, v = metrics.homogeneity_completeness_v_measure(labels, preds)
    s1 = metrics.silhouette_score(embeddings, labels, metric=metric)
    s2 = metrics.silhouette_score(embeddings, preds, metric=metric)

    print(f"homogenetiy: {h}, completeness: {c}, v_measure: {v}, silhouette_score label: {s1}, silhouette_score pred: {s2}\n")
