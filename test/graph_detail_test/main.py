# -*- encoding: utf-8 -*-

import networkx as nx

def graph_node_info(graphName: str):
    # 分析图中有哪些独立的节点，度数如何
    graph = nx.read_edgelist(f"../../data/graph/{graphName}.edgelist", create_using=nx.Graph, edgetype=float,
                             data=[('weight', float)])

    outlier_cnt = 0
    for node in nx.nodes(graph):
        if nx.degree(graph, node) == 0:
            outlier_cnt += 1

    print(f"number of outliers is {outlier_cnt}")

if __name__ == '__main__':
    graph_node_info("cora")
