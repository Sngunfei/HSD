# -*- encoding: utf-8 -*-

import networkx as nx


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


def print_different_hop_neighbors(graph: nx.Graph):
    nodes = list(nx.nodes(graph))
    for node in range(1, len(nodes)+1):
        layers = get_node_hierarchical_structure(graph, node, maxHop=2)
        print(node, len(layers[1]), len(layers[2]))


if __name__ == '__main__':
    G = nx.read_edgelist("../../data/graph/karate.edgelist", create_using=nx.Graph,
                         nodetype=int, edgetype=float, data=[("weight", float)])
    print_different_hop_neighbors(G)
