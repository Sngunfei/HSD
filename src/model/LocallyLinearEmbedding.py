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
from utils.plt import plot_embeddings, plot_embedding2D



class LocallyLinearEmbedding:

    def __init__(self, graph):
        self.graph = graph
        self.n_nodes = nx.number_of_nodes(graph)
        self.nodes = list(nx.nodes(graph))


    def create_embedding(self, d):
        A = nx.to_scipy_sparse_matrix(self.graph)
        normalize(A, norm='l1', axis=1, copy=False)
        I_n = sp.eye(self.n_nodes)
        I_min_A = I_n - A
        u, s, vt = lg.svds(I_min_A, d+1, which='SM')
        self._X = vt.T
        self._X = self._X[:, 1:]
        return self._X

    def get_edge_weight(self, i, j):
        return np.exp(
            -np.power(np.linalg.norm(self._X[i, :] - self._X[j, :]), 2)
        )

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self._X = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


def sklearn_LLE(data_path):
    from sklearn.manifold import LocallyLinearEmbedding

    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, edgetype=float, data=[('weight', float)])
    LLE = LocallyLinearEmbedding(n_neighbors=10, n_components=64, random_state=42)
    A = nx.to_scipy_sparse_matrix(graph)
    #normalize(A, norm='l1', axis=1, copy=False)
    A = A.todense()
    X_transformed = LLE.fit_transform(A)
    plot_embeddings(list(nx.nodes(graph)), X_transformed, method="pca")
    plt.show()


if __name__ == '__main__':
    data_path = 'G:\pyworkspace\graph-embedding\out\\fb1_5_L1.txt'
    #sklearn_LLE(data_path)

    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, edgetype=float, data=[('weight', float)])
    model = LocallyLinearEmbedding(graph)
    embeddings = np.array(model.create_embedding(5))
    plot_embeddings(model.nodes, embeddings, method="tsne")
    plt.show()


