# -*- coding:utf-8 -*-

"""
evluate the performance of embedding via different methods.
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def cluster_evaluate(embeddings=None, labels=None, class_num=None):
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
    clusters = AgglomerativeClustering(n_clusters=class_num, linkage='single').fit_predict(embeddings)
    h, c, v = metrics.homogeneity_completeness_v_measure(labels, clusters)
    s = metrics.silhouette_score(embeddings, clusters)
    print("cluster:", clusters, "labels:", labels)
    print("homogeneity: ", h)
    print("completeness: ", c)
    print("v-score: ", v)
    print("silhouette: ", s)

    return h, c, v, s


def evaluate_LR_accuracy(embeddings=None, labels=None, random_state=42):
    """
    Evaluate embedding effect using Logistic Regression. Mode = One vs Rest (OVR)

    :param embeddings: learned representation vectors. shape=(n_samples, n_dim)
    :param labels: nodes' label for classification.
    :param random_state: random seed.
    :return: Accuracy score, float.
    """
    from sklearn.linear_model import LogisticRegressionCV
    #from sklearn.multiclass import OneVsRestClassifier

    xtrain, xtest, ytrain, ytest = train_test_split(embeddings, labels, test_size=0.3,
                                                    random_state=random_state, shuffle=True)

    lrc = LogisticRegressionCV(cv=25, solver="lbfgs", penalty='l2', max_iter=1000, verbose=0, multi_class='ovr')
    lrc.fit(xtrain, ytrain)
    preds = lrc.predict(xtest)
    score = accuracy_score(preds, ytest)
    balanced_score = balanced_accuracy_score(ytest, preds)
    report = classification_report(ytest, preds)
    print("logistic regression(ovr) accuracy score:{}.".format(score))
    print("logistic regression(ovr) balanced accuracy score:{}.".format(balanced_score))

    print("classification report: ")
    print(report)

    return score


def evaluate_SVC_accuracy(embeddings=None, labels=None, random_state=42):
    """
    Evaluate embedding effect using support vector classifier. Mode = One vs Rest (OVR)

    :param embeddings: learned representation vectors. shape=(n_samples, n_dim)
    :param labels: nodes' label for classification.
    :param random_state: random seed.
    :return: Accuracy score, float.
    """
    from sklearn import svm

    xtrain, xtest, ytrain, ytest = train_test_split(embeddings, labels, test_size=0.2,
                                                    random_state=random_state, shuffle=True)

    model = svm.SVC(decision_function_shape="ovr", C=0.5)
    model.fit(xtrain, ytrain)
    preds = model.predict(xtest)
    score = accuracy_score(ytest, preds)
    balanced_score = balanced_accuracy_score(ytest, preds)
    report = classification_report(ytest, preds)

    print("SVC(ovr) accuracy score:{}.".format(score))
    print("SVC(ovr) balanced accuracy score:{}.".format(balanced_score))
    print("classification report: ")
    print(report)

    return score


def evaluate_KNN_accuracy(embeddings=None, labels=None, random_state=42):
    """
    基于节点的相似度进行KNN分类，在嵌入之前进行，为了验证通过层次化相似度的优良特性。
    :return:
    """
    xtrain, xtest, ytrain, ytest = train_test_split(embeddings, labels, test_size=0.3,
                                                    random_state=random_state, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=20, weights="uniform", algorithm="auto", n_jobs=-1)
    knn.fit(xtrain, ytrain)
    preds = knn.predict(xtest)

    score = accuracy_score(ytest, preds)
    balanced_score = balanced_accuracy_score(ytest, preds)
    report = classification_report(ytest, preds)

    print("KNN accuracy score:{}.".format(score))
    print("KNN balanced accuracy score:{}.".format(balanced_score))
    print("classification report: ")
    print(report)

    return score


if __name__ == '__main__':
    from utils.util import dataloader
    from model.LaplacianEigenmaps import LaplacianEigenmaps

    graph, label_dict, n_class = dataloader("europe", auto_label=True, directed=False, similarity=True, scale=10, metric="l1")
    model = LaplacianEigenmaps(graph, dim=32)
    embedding_dict = model.create_embedding(percentile=0.85)
    nodes = []
    labels = []
    embeddings = []
    for node, embedding in embedding_dict.items():
        nodes.append(node)
        embeddings.append(embedding)
        labels.append(label_dict.get(node, str(n_class)))

    evaluate_KNN_accuracy(embeddings=embeddings, labels=labels, n_neighbors=10, random_state=42)



