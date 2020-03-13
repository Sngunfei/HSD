# -*- coding:utf-8 -*-

"""
Test robustness for edge removel.
"""

import networkx as nx
import random


def random_remove_edges(graph=None, prob=0.9):
    random.seed()
    edges = nx.edges(graph)
    del_edges = []
    for edge in edges:
        u, v = edge[0], edge[1]
        if random.random() > prob:
            del_edges.append((u, v))
    graph.remove_edges_from(del_edges)
    return graph
