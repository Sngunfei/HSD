# -*- encoding: utf-8 -*-

"""
得到一张图的层级结构表示
"""

import networkx as nx

from utils.util import build_node_idx_map


def get_node_layers(graph: nx.graph, node: str, max_hop=5) -> list:
    """
    得到某个点的层级结构表示。
    :param graph:
    :param node:
    :param max_hop: 最大hop数，包括自身，比如说5层，那么就是0,1,2,3,4
    :return:  list[list]
    """
    vised = set()
    rings = []
    queue = [node]

    vised.add(node)
    rings.append(queue)
    for hop in range(max_hop):
        size = len(queue)
        for _ in range(size):
            cur = queue.pop(0)
            for neibor in nx.neighbors(graph, cur):
                if neibor not in vised:
                    queue.append(neibor)
                    vised.add(neibor)
        rings.append(queue)

    return rings


def get_graph_layers(graph: nx.graph, max_hop=5) -> dict:
    """
    得到图的层级表示
    :param graph:
    :param max_hop:
    :return: dict[node: rings]
    """
    res = dict()
    for node in nx.nodes(graph):
        res[node] = get_node_layers(graph, node, max_hop)

    return res


def wavelet_layers(graph: nx.graph, coeff_mat, max_hop=5) -> dict:
    """
    小波系数的分层表示
    :param graph:
    :param coeff_mat:
    :param max_hop:
    :return:
    """
    idx2node, node2idx = build_node_idx_map(graph)
    layers = get_graph_layers(graph, max_hop)
    coeffs_dict = dict()

    for idx, node in idx2node.items():
        coeffs = []
        rings = layers[node]
        for _, ring in enumerate(rings):
            coeffs.append([coeff_mat[idx, node2idx[neighbor]] for neighbor in ring])
        coeffs_dict[node] = coeffs

    return coeffs_dict




