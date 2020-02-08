# -*- coding:utf-8 -*-

"""
evluate the performance of embedding via different methods.
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score, precision_score, recall_score
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


def evaluate_LR_accuracy(embeddings, labels, random_state=42):
    """
    Evaluate embedding effect using Logistic Regression. Mode = One vs Rest (OVR)

    :param embeddings: learned representation vectors. shape=(n_samples, n_dim)
    :param labels: nodes' label for classification.
    :param random_state: random seed.
    :return: Accuracy score, float.
    """
    from sklearn.linear_model import LogisticRegressionCV

    xtrain, xtest, ytrain, ytest = train_test_split(embeddings, labels, test_size=0.3,
                                                    random_state=random_state, shuffle=True)

    lrc = LogisticRegressionCV(cv=5, solver="lbfgs", penalty='l2', max_iter=500, verbose=0, multi_class='ovr')
    lrc.fit(xtrain, ytrain)
    preds = lrc.predict(xtest)

    print("------------------------------ LR --------------------------------")
    accuracy, balanced_accuracy, precision, recall, macro_f1, micro_f1 = evalute_results(ytest, preds)

    return accuracy, balanced_accuracy, precision, recall, macro_f1, micro_f1


def evaluate_SVC_accuracy(embeddings, labels, random_state=42):
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


def evaluate_KNN_accuracy(X, labels, metric, n_neighbor=10, random_state=42):
    """
    基于节点的相似度进行KNN分类，在嵌入之前进行，为了验证通过层次化相似度的优良特性。
    :return:
    """
    knn = KNeighborsClassifier(weights='uniform', algorithm="auto", n_neighbors=n_neighbor, metric=metric)
    #knngs = GridSearchCV(knn, param_grid={"n_neighbors": [n_neighbor], "metric": [metric]}, cv=10)
    preds = cross_val_predict(knn, X, labels, cv=10)

    print("------------------------------ KNN --------------------------------")
    accuracy, balanced_accuracy, precision, recall, macro_f1, micro_f1 = evalute_results(labels, preds)
    return accuracy, balanced_accuracy, precision, recall, macro_f1, micro_f1


def evalute_results(labels: list, preds: list):
    accuracy = accuracy_score(labels, preds)
    balanced_accuracy = balanced_accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="micro")
    recall = recall_score(labels, preds, average="micro")
    macro_f1 = f1_score(labels, preds, average="macro")
    micro_f1 = f1_score(labels, preds, average="micro")

    print("accuracy: ", accuracy)
    print("balanced accuracy: ", balanced_accuracy)
    print("micro precision: ", precision)
    print("micro recall: ", recall)
    print("macro f1: ", macro_f1)
    print("micro f1: ", micro_f1)

    report = classification_report(labels, preds, digits=7)

    print(report)

    return accuracy, balanced_accuracy, precision, recall, macro_f1, micro_f1







