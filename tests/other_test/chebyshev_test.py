# -*- encoding: utf-8 -*-

import time

import networkx as nx
import numpy as np
import pygsp

def precision_test(graph, scale, order):
    G = pygsp.graphs.Graph(nx.adjacency_matrix(graph))

    start = time.time()
    # approx mode
    G.estimate_lmax()
    heat_filter = pygsp.filters.Heat(G, tau=[scale * G._lmax])
    chebyshev = pygsp.filters.approximations.compute_cheby_coeff(heat_filter, m=order)
    approx_wavelets = []
    n_node = nx.number_of_nodes(graph)
    for idx in range(n_node):
        impulse = np.zeros(n_node, dtype=np.float)
        impulse[idx] = 1.0
        coeff = pygsp.filters.approximations.cheby_op(G, chebyshev, impulse)
        approx_wavelets.append(coeff)
    print(time.time() - start)

    # precise mode
    eigenvalues, eigenvectors = np.linalg.eigh(nx.laplacian_matrix(graph).todense())
    precise_wavelets = np.dot(np.dot(eigenvectors, np.diag(np.exp(-1 * scale * eigenvalues))),
                              np.transpose(eigenvectors))
    print(time.time() - start)

    diff = approx_wavelets - precise_wavelets
    print("max diff:", np.max(np.abs(diff)))
    print("min diff:", np.min(np.abs(diff)))
    print("sum diff:", np.sum(np.abs(diff)))
    print("avg diff:", np.mean(np.abs(diff)))

    print("-" * 80)

if __name__ == '__main__':
    g = nx.read_edgelist(f"../../data/graph/cox2.edgelist", create_using=nx.Graph, edgetype=float,
                         data=[('weight', float)])
    for scale in np.linspace(0.5, 50, 10):
        precision_test(g, scale, 50)
