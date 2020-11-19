# -*- encoding: utf-8 -*-

import networkx as nx

def pageRank_label(graphName: str, n_class: int) -> dict:
    graph1 = nx.read_edgelist(f"../data/graph/{graphName}.edgelist", create_using=nx.Graph, edgetype=float,
                              data=[('weight', float)])
    ranks = nx.pagerank(graph1, max_iter=1000)
    scores = sorted(ranks.items(), key=lambda x: x[1])

    interval = len(scores) / n_class
    labels = {}
    label = 1
    for idx, (node, score) in enumerate(scores):
        labels[node] = label
        if idx >= interval * label:
            label += 1

    return labels

if __name__ == '__main__':
    graphName = "cora"
    label_dict = pageRank_label(graphName, 5)
    with open(f"../data/label/{graphName}_PageRank.label", mode="w+", encoding="utf-8") as fout:
        for node, label in label_dict.items():
            fout.write(f"{node} {label}\n")
