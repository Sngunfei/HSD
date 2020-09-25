# -*- encoding:utf-8 -*-

import networkx as nx
from scipy import sparse
from scipy.linalg import eigvals
import numpy as np
from model.HSD import HSD
import time

def func():
    graph = nx.read_edgelist("../data/graph/test1.edgelist", create_using=nx.Graph, edgetype=float, data=[('weight', float)])
    laplacian = nx.laplacian_matrix(graph).todense()
    laplacian = -1 * laplacian
    print(laplacian)
    cur = laplacian
    for i in range(5):
        cur = np.matmul(cur, laplacian)
        print(cur)


def chebyshev_approx_test():
    graph = nx.read_edgelist("../data/graph/bio_grid_human_new.edgelist", create_using=nx.Graph, edgetype=float,
                             data=[('weight', float)])
    hsd = HSD(graph, "bio_grid_human", scale=1, hop=5, metric="wasserstein")
    startTime = time.time()
    wavelets1 = hsd.calculate_wavelets(scale=1, approx=True)
    wavelets2 = hsd.calculate_wavelets(scale=1, approx=False)
    diff = wavelets1 - wavelets2
    print("max diff:", np.max(np.abs(diff)))
    print("min diff:", np.min(np.abs(diff)))
    print("sum diff:", np.sum(np.abs(diff)))
    print("avg diff:", np.mean(np.abs(diff)))
    print(time.time() - startTime)


if __name__ == '__main__':
    scales1 = np.exp(np.linspace(np.log(0.01), np.log(10), 20))
    scales2 = np.linspace(0.01, 10, 20)
    print(scales1)
    print(scales2)

