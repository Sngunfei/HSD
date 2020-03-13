# -*- coding:utf-8 -*-
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from sklearn.preprocessing import normalize
from sklearn.manifold import LocallyLinearEmbedding as LLE


class LocallyLinearEmbedding:
    """
        实现了局部线性嵌入算法，和Sklearn中的LLE对比使用。
    """
    def __init__(self, graph, n_neighbors = 10, dim=16):
        self.graph = graph
        self.n_neighbors = n_neighbors
        self.n_nodes = nx.number_of_nodes(graph)
        self.nodes = list(nx.nodes(graph))
        self.adj = nx.to_scipy_sparse_matrix(self.graph)
        self.dim = dim
        self.embeddings = {}


    def create_embedding(self):
        A = self.adj
        A = normalize(A, norm='l1', axis=1, copy=False)
        I_n = sp.eye(self.n_nodes)
        I_min_A = I_n - A
        u, s, vt = lg.svds(I_min_A, self.dim + 1, which='SM')
        self._X = vt.T
        self._X = self._X[:, 1:]
        self.embeddings = {}
        for idx, node in enumerate(self.nodes):
            self.embeddings[node] = self._X[idx, :]
        return self.embeddings


    def sklearn_lle(self, random_state=42):
        A = self.adj
        A = A.todense()
        model = LLE(n_neighbors=self.n_neighbors, n_components=self.dim, random_state=random_state)
        X = np.array(model.fit_transform(A))
        self.embeddings = {}
        for idx, node in enumerate(self.nodes):
            self.embeddings[node] = X[idx, :]
        return self.embeddings

