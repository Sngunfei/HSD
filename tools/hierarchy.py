# -*- encoding: utf-8 -*-

"""
PRD：对图的层级划分不应该内嵌到具体的模型中，而是抽象出一套公共方法
能够实现对图的层级处理，比如io操作，省去每次模型run的时候都去重构一套
"""

import copy
import os
import networkx as nx
from tqdm import tqdm


PathTemplate = "data/hierarchy/{}.layers"
MaxHop = 5


def get_hierarchical_representation(graph: nx.Graph, maxHop):
    hierarchy = {}
    for node in nx.nodes(graph):
        hierarchy[node] = get_node_hierarchical_structure(graph, node, maxHop)
    return hierarchy


def get_node_hierarchical_structure(graph: nx.Graph, node: str, hop: int):
    """
    explore hierarchical neighborhoods of node
    """
    layers = [[node]]
    curLayer = {node}
    visited = {node}
    for _ in range(hop):
        if len(curLayer) == 0:
            break
        nextLayer = set()
        for neighbor in curLayer:
            for next_hop_neighbor in nx.neighbors(graph, neighbor):
                if next_hop_neighbor not in visited:
                    nextLayer.add(next_hop_neighbor)
                    visited.add(next_hop_neighbor)
        curLayer = nextLayer
        layers.append(list(nextLayer))
    return layers


def save_hierarchical_representation(graphName: str, graph: nx.Graph):
    """
    explore & save hierarchy of graph
    hierarchy file format:

    node#neighbor,...,neighbor#neighbor,...,neighbor#
    .
    .
    .

    where `#` denote increasing hop
    """
    file_path = PathTemplate.format(graphName)
    with open(file_path, encoding="utf-8", mode="w+") as fout:
        nodes = nx.nodes(graph)
        for node in tqdm(nodes):
            record = ""
            layers = get_node_hierarchical_structure(graph, node, MaxHop)
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

    print(f"save {graphName} hierarchical representation done\n")


def read_hierarchical_representation(graphName: str, maxHop=3) -> dict:
    path = PathTemplate.format(graphName)
    if not os.path.exists(path):
        raise FileNotFoundError(f"graph:{graphName}, hierarchy file not exist")

    hierarchy = {}
    with open(path, mode="r", encoding="utf-8") as fin:
        while True:
            line = fin.readline().strip()
            if not line:
                break
            layers = []
            for idx, level in enumerate(line.split("#")):
                if idx >= maxHop:
                    break
                neighbors = level.strip().split(",")
                layers.append(neighbors)

            hierarchy[layers[0][0]] = layers

    print(f"load {graphName} hierarchy done. number of nodes: {len(hierarchy)}")
    return hierarchy


if __name__ == '__main__':
    graphs = ["barbell", "mkarate", "europe", "usa"]
    for graphName in graphs:
        graph = nx.read_edgelist(path=f"data/graph/{graphName}.edgelist", create_using=nx.Graph,
                                 edgetype=float, data=[('weight', float)])
        save_hierarchical_representation(graphName, graph)
