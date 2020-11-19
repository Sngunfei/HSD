# -*- encoding:utf-8 -*-

"""
节点分类实验
"""

import networkx as nx

from model.multiscale_HSD import MultiHSD
from tools import dataloader

def multi_HSD(graphName, hop, n_scales) -> (list, list):
    graph = nx.read_edgelist(f"../../data/graph/{graphName}.edgelist", create_using=nx.Graph, edgetype=float,
                             data=[('weight', float)])
    label_dict = dataloader.read_label(f"../../data/label/{graphName}.label")

    model = MultiHSD(graph, graphName, hop, n_scales)
    model.init()
    embedding_dict = model.parallel_embed(n_workers=3)

    embeddings, labels = [], []
    for node, vector in embedding_dict.items():
        embeddings.append(vector)
        labels.append(label_dict[node])

    return embeddings, labels

if __name__ == '__main__':
    pass
