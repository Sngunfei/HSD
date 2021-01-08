# -*- encoding: utf-8 -*-

# 热核函数的性质


import numpy as np
import networkx as nx
import math
from tools.util import build_node_idx_map

def main():
    graph = nx.read_edgelist("../../data/graph/mkarate.edgelist", create_using=nx.Graph, nodetype=int, edgetype=float, data=[('weight', float)])
    A = nx.adjacency_matrix(graph).todense()
    L = nx.laplacian_matrix(graph).todense()
    es, vs = np.linalg.eigh(L)
    #print(es)
    #print(vs[1])

    idx2node, node2idx = build_node_idx_map(graph)

    n = nx.number_of_nodes(graph)
    i, j = 0, 1
    alpha = 0.5
    beta = 100
    coeff = 0.0

    wavelets = np.dot(np.dot(vs, np.diag(np.exp(-1 * beta * es))), np.transpose(vs))

    W = np.zeros(shape=(n, n))
    for edge in nx.edges(graph):
        u, v = edge[0], edge[1]
        i, j = node2idx[u], node2idx[v]
        W[i, j] = 1 / nx.degree(graph, v)
        W[j, i] = 1 / nx.degree(graph, u)
    print(W)

    station = (1 - alpha) * np.dot((1 / n) * np.ones(shape=(n, 1)), np.ones(shape=(1, n)))
    P = station + alpha*W
    print(P)

    def recursive(lamb, P, depth):
        if depth == 20:
            return
        ans = lamb * P + (1 - lamb) * recursive()


if __name__ == '__main__':
    main()