# -*- coding:utf-8 -*-

"""
Test robustness for edge removel.
"""

import networkx as nx
import random


def random_remove_edges(graph, ratio):
    """
    在图内加入微小噪声
    :param graph:
    :param ratio: 噪声比例
    :return:
    """
    random.seed()
    g = nx.Graph(graph)
    edges = list(nx.edges(g))
    n = int(len(edges) * ratio)
    del_edges = []
    del_idx = []
    cnt = 0
    while cnt < n:
        idx = random.randint(0, len(edges)-1)
        if idx not in del_idx:
            del_edges.append(edges[idx])
            del_idx.append(idx)
            cnt += 1

    g.remove_edges_from(del_edges)
    return g


def random_add_edges(graph, ratio: float):
    """
    添加噪声，加入随机边
    :param graph:
    :param ratio:
    :return:
    """
    random.seed()
    g = nx.Graph(graph)
    nodes = list(nx.nodes(g))
    random.shuffle(nodes)
    n = int(nx.number_of_edges(g) * ratio)
    for _ in range(n):
        node1 = random.choice(nodes)
        node2 = random.choice(nodes)
        if node1 == node2:
            continue
        g.add_edge(node1, node2)

    return g


def add_noise(graph, ratio:float):
    """

    :param graph:
    :param ratio:
    :return:
    """
    g = random_add_edges(graph, ratio)
    g = random_remove_edges(g, ratio)
    return g