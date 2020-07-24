# -*- encoding: utf-8 -*-

"""
研究图本身的一些性质，工具函数放在这，比如对连通分量规模等等
"""

import networkx as nx

#todo

def node_ring_scale(graph: nx.Graph):
    """
    研究非连通图的节点，在其连通分量中的环状结构性质，比如bio_grid_human
    :param graph:
    :return:
    """
    nodes = nx.nodes(graph)
    n_components = nx.number_connected_components(graph)
    print(f"Number of components: {n_components}")

    for node in nodes:
        pass

