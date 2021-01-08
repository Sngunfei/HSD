# -*- encoding: utf-8 -*-

import networkx as nx
from tools import const


def centriality(graph: nx.Graph, label_type: str) -> dict:
    if label_type == const.PAGERANK:
        scores = nx.pagerank(graph, max_iter=1000)
    elif label_type == const.EIGEN_CENTRALITY:
        scores = nx.eigenvector_centrality(graph, max_iter=1000)
    else:
        raise NotImplementedError(f"{label_type} not supported yet")
    return scores


def write_label(score_dict: dict, number_class: int, label_path: str):
    scores = sorted(score_dict.items(), key=lambda kv: kv[1])
    label_file = open(label_path, mode="w+", encoding="utf-8")
    cur_label = 0
    class_size = len(scores) // number_class + 1
    count = 0
    for node, _ in scores:
        label_file.write(f"{node} {cur_label}\n")
        count += 1
        if count == class_size:
            cur_label += 1
            count = 0
            
    label_file.flush()
    label_file.close()


if __name__ == '__main__':
    graph_name = "facebook"
    label_type = const.EIGEN_CENTRALITY
    number_class = 5
    
    G = nx.read_edgelist(f"../data/graph/{graph_name}.edgelist", create_using=nx.Graph,
                         nodetype=str, edgetype=float, data=[("weight", float)])
    score_dict = centriality(G, label_type)
    write_label(score_dict, number_class, f"../data/label/{graph_name}_{label_type}.label")
