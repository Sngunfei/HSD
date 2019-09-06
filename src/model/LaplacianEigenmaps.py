
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import scipy.sparse.linalg as lg
from utils.visualize import plot_embeddings


class LaplacianEigenmaps:

    def __init__(self, graph):
        self.graph = graph
        self.n_node = nx.number_of_nodes(graph)
        self.nodes = nx.nodes(graph)


    def create_embedding(self, d):
        L_sym = nx.normalized_laplacian_matrix(graph)

        w, v = lg.eigs(L_sym, k=d + 1, which='SM')
        self._X = v[:, 1:]

        p_d_p_t = np.dot(v, np.dot(np.diag(w), v.T))
        eig_err = np.linalg.norm(p_d_p_t - L_sym)
        print('Laplacian matrix recon. error (low rank): %f' % eig_err)
        return self._X

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


if __name__ == '__main__':
    data_path = 'G:\pyworkspace\graph-embedding\out\\subway_dist_L1.txt'
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, edgetype=float, data=[('weight', float)])
    model = LaplacianEigenmaps(graph)
    embeddings = np.array(model.create_embedding(5))
    print(embeddings.shape)
    plot_embeddings(model.nodes, embeddings, method="pca")
    plt.show()