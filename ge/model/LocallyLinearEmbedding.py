# -*- coding:utf-8 -*-
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize
from time import time
import pdb
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding as LLE


class LocallyLinearEmbedding:
    """
        实现了局部线性嵌入算法，和Sklearn中的LLE对比使用。
    """
    def __init__(self, graph):
        self.graph = graph
        self.n_nodes = nx.number_of_nodes(graph)
        self.nodes = list(nx.nodes(graph))
        self.adj = nx.to_scipy_sparse_matrix(self.graph)
        self.embeddings = {}


    def create_embedding(self, d):
        """

        :param d: 目标嵌入维度
        :return:
        """
        A = self.adj
        A = normalize(A, norm='l1', axis=1, copy=False)
        I_n = sp.eye(self.n_nodes)
        I_min_A = I_n - A
        u, s, vt = lg.svds(I_min_A, d+1, which='SM')
        self._X = vt.T
        self._X = self._X[:, 1:]
        self.embeddings = {}
        for idx, node in enumerate(self.nodes):
            self.embeddings[node] = self._X[idx, :]
        return self.embeddings


    def sklearn_lle(self, n_neighbors, dim, random_state):
        A = self.adj
        A = A.todense()
        model = LLE(n_neighbors=n_neighbors, n_components=dim, random_state=random_state)
        X = np.array(model.fit_transform(A))
        self.embeddings = {}
        for idx, node in enumerate(self.nodes):
            self.embeddings[node] = X[idx, :]
        return self.embeddings



if __name__ == '__main__':

    from ge.utils.visualize import plot_embeddings, plot_subway_embedding
    from ge.utils.util import cluster_evaluate, dataloader, evaluate_LR_accuracy, evaluate_SVC_accuracy

    dataset = 'subway'
    scale = 10
    metric = 'L1'

    graph, label_dict, n_class = dataloader(name=dataset, similarity=True, scale=scale, metric=metric)
    model = LocallyLinearEmbedding(graph)
    embeddings_dict = model.create_embedding(16)

    embeddings = []
    nodes = []
    labels = []
    for node, embedd in embeddings_dict.items():
        embeddings.append(embedd)
        nodes.append(node)
        labels.append(label_dict[node])
    evaluate_LR_accuracy(embeddings, labels, random_state=42)
    evaluate_SVC_accuracy(embeddings, labels, random_state=42)
    #cluster_evaluate(embeddings, labels, class_num=n_class, perplexity=5)
    #plot_embeddings(nodes, embedd, labels=labels, method="tsne", perplexity=5)
    #plot_subway_embedding(nodes, embeddings, labels, perplexity=5)
