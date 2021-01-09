# -*- encoding: utf-8 -*-

"""
PRD：对图的层级划分不应该内嵌到具体的模型中，而是抽象出一套公共方法
能够实现对图的层级处理，比如io操作，省去每次模型run的时候都去重构一套
"""

import os
import platform
import networkx as nx
from tqdm import tqdm

from tools import const


def get_hierarchical_representation(graph: nx.Graph, maxHop):
    hierarchy = {}
    for node in nx.nodes(graph):
        hierarchy[node] = get_node_hierarchical_structure(graph, node, maxHop)

    print(f"done, number of nodes: {len(hierarchy)}")
    return hierarchy


def get_node_hierarchical_structure(graph: nx.Graph, node: str, maxHop: int):
    layers = [[node]]
    curLayer = {node}
    visited = {node}
    for i in range(1, maxHop+1):
        nextLayer = set()
        for neighbor in curLayer:
            for next_hop_neighbor in nx.neighbors(graph, neighbor):
                if next_hop_neighbor not in visited:
                    nextLayer.add(next_hop_neighbor)
                    visited.add(next_hop_neighbor)
        curLayer = nextLayer
        layers.append(list(nextLayer))
    return layers


def save_hierarchical_representation(graph: nx.Graph, file_path: str, hop=7):
    """
    explore & save hierarchy of graph
    hierarchy file format:
        node#neighbor,...,neighbor#neighbor,...,neighbor#
        ...
    where `#` denote increasing hop
    """
    with open(file_path, encoding="utf-8", mode="w+") as fout:
        nodes = nx.nodes(graph)
        for node in tqdm(nodes):
            record = ""
            layers = get_node_hierarchical_structure(graph, node, hop)
            for level in layers:
                if len(level) == 0:
                    break
                for idx, neighbor in enumerate(level):
                    if idx == len(level) - 1:
                        record += str(neighbor) + '#'
                    else:
                        record += str(neighbor) + ','
            fout.write(record + '\n')
            fout.flush()


def read_hierarchical_representation(graphName: str, maxHop=3) -> dict:
    cur_system = platform.system()
    if cur_system == "Windows":
        PathTemplate = const.HierarchyWindowsPathTemplate
    else:
        PathTemplate = const.HierarchyLiunxPathTemplate
    file_path = PathTemplate.format(graphName)
    return read_hierarchy(file_path, maxHop)


def read_hierarchy(file_path: str, maxHop: int) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"path:{file_path}, hierarchy file not exist")

    hierarchy = {}
    with open(file_path, mode="r", encoding="utf-8") as fin:
        while True:
            line = fin.readline().strip()
            if not line:
                break
            layers = []
            neighbor_layer = line.split("#")
            for hop in range(0, maxHop + 1):
                if hop >= len(neighbor_layer):
                    layer = []
                else:
                    layer = neighbor_layer[hop].strip().split(",")

                layers.append(layer)

            hierarchy[layers[0][0]] = layers

    print(f"done, number of nodes: {len(hierarchy)}")
    return hierarchy


if __name__ == '__main__':
    graph = "zxr_2"
    G = nx.read_edgelist(f"../data/graph/{graph}.edgelist", create_using=nx.Graph,
                         nodetype=str, edgetype=float, data=[("weight", float)])
    save_hierarchical_representation(G, f"../data/hierarchy/{graph}.layers", hop=5)