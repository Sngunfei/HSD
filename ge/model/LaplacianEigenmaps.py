# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from utils.visualize import plot_embeddings
import pandas as pd


class LaplacianEigenmaps:

    def __init__(self, graph):
        self.graph = graph
        self.n_nodes = graph.number_of_nodes()
        self.nodes = list(graph.nodes())
        self.embeddings = {}


    def create_embedding(self, d):
        self.d = d
        L_sym = nx.normalized_laplacian_matrix(self.graph)
        w, v = lg.eigs(L_sym, k=d + 1, which='SM')
        self._X = v[:, 1:]
        p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
        eig_err = np.linalg.norm(p_d_p_t - L_sym)
        print('Laplacian matrix recon. error (low rank): %f' % eig_err)

        for idx, node in enumerate(self.nodes):
            embedding = self._X[idx, :]
            self.embeddings[node] = np.real(embedding)
        return self.embeddings


    def save_embedding(self, filename):
        embeddings = np.array([embedding for embedding in self.embeddings.values()])
        df = pd.DataFrame(data=embeddings, index=self.nodes)
        df.to_csv(filename, mode='w+', encoding='utf8', header=[x for x in range(self.d)])


    def get_embedding(self):
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
