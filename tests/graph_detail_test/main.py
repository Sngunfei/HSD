# -*- encoding: utf-8 -*-

import networkx as nx
import numpy as np
from tqdm import tqdm


# 分析图上一些必要信息
def graph_infos(G: nx.Graph):

    nodes = list(nx.nodes(G))
    edges = list(nx.edges(G))
    print(f"number of nodes: {len(nodes)}")
    print(f"number of edges: {len(edges)}")

    degrees = nx.degree(G)
    degrees = sorted([node_degree[1] for node_degree in degrees])
    print(f"average degree: {np.mean(degrees)}, standard deviation: {np.std(degrees)}")
    print(f"max degree: {degrees[-1]}, min degree: {degrees[0]}")
    print(f"number of outlier nodes: {len(degrees) - np.count_nonzero(degrees)}")

    connected = nx.is_connected(G)
    print(f"connected: {connected}")
    number_connected_compoents = nx.number_connected_components(G)
    print(f"number of connected compoents: {number_connected_compoents}")

    if connected:
        print(f"diameter: {nx.diameter(G)}")
    else:
        print(f"longest shortest path length: {longest_shortest_path(G)}")


def longest_shortest_path(G: nx.Graph):
    if nx.is_connected(G):
        return nx.diameter(G)

    diameter = 0
    for nodes_set in tqdm(nx.connected_components(G)):
        longest_shortest = 0
        nodes = list(nodes_set)
        for idx1, node1 in enumerate(nodes):
            for idx2 in range(idx1+1, len(nodes)):
                node2 = nodes[idx2]
                length = -1
                try:
                    length = nx.shortest_path_length(G, node1, node2)
                except nx.NetworkXNoPath:
                    pass
                finally:
                    longest_shortest = max(longest_shortest, length)
        print(f"nodes:{len(nodes)}, diameter: {longest_shortest}")
        diameter = max(diameter, longest_shortest)

    return diameter


if __name__ == '__main__':
    graph = nx.read_edgelist(f"../../data/graph/bio_dmela.edgelist", create_using=nx.Graph,
                             nodetype=str, edgetype=float, data=[('weight', float)])
    graph_infos(graph)
    # node_length = []
    # for node in nx.nodes(graph):
    #     if int(node)+1 >= 37:
    #         continue
    #     max_length = max(list(nx.shortest_path_length(graph, node).values()))
    #     node_length.append((int(node)+1, max_length))
    # node_length = sorted(node_length, key=lambda x: x[1])
    # for node, length in node_length:
    #     print(node, length)

