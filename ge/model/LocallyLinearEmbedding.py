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
    from ge.utils.util import read_label, cluster_evaluate

    graph = nx.read_edgelist(path="../../similarity/subway_10_L1.csv", create_using=nx.Graph, edgetype=float, data=[('weight', float)])
    labels = read_label(path='../../data/subway.label')

    model = LocallyLinearEmbedding(graph)
    embeddings = model.create_embedding(32)
    #embeddings = model.sklearn_lle(10, 32, 666)
    nodes = model.nodes
    embedd = []
    L = []
    for node in nodes:
        embedd.append(embeddings[node])
        L.append(labels[node])
    embedd = np.array(embedd)
    #cluster_evaluate(embedd, L, class_num=12, perplexity=5)
    #plot_embeddings(nodes, embedd, labels=labels, method="tsne", perplexity=5)
    plot_subway_embedding(nodes, embedd, L, perplexity=10)
