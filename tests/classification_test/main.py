# -*- encoding:utf-8 -*-

"""
节点分类实验
"""

import sys
sys.path.append("/home/data/users/master/2019/songyunfei/workspace/py/HSD")

from collections import defaultdict

import networkx as nx
import numpy as np
from model import MultiHSD, HSD
from tools import dataloader, const
from tools.evaluate import KNN_evaluate


# 输出距离矩阵
def multiHSD_dist(graph, graphName, hop, n_scales) -> np.ndarray:
    model = MultiHSD(graph, graphName, hop, n_scales)
    dist_mat = model.parallel_calculate_structural_distance(n_workers=3)
    return dist_mat


# 输出嵌入向量
def multiHSD_embed(graph, graphName, hop, n_scales) -> np.ndarray:
    model = MultiHSD(graph, graphName, hop, n_scales)
    embedding_dict = model.parallel_embed(n_workers=3)
    embedding_list = []
    for node in nx.nodes(graph):
        embedding_list.append(embedding_dict[node])
    return np.array(embedding_list)


# 节点各阶层度数分布
def hierarchical_degrees(graph, grapuName, hop) -> np.ndarray:
    hsd = HSD(graph, grapuName, 1.0, hop, "euclidean")
    hsd.init()
    degree_dict = hsd.get_nodes_hierarchical_degree()
    degree_list = []
    for node in nx.nodes(graph):
        degree_list.append(degree_dict[node])
    return np.array(degree_list)


# PCA降维展示
def dimension_reduction(features):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, whiten=False, random_state=42)
    results = pca.fit_transform(features)
    return results


def run():
    graphName = "europe"
    hop = 3
    n_scales = 100
    label_type = const.EIGEN_CENTRALITY

    graph, label_dict = dataloader.load_data(graphName, label_type)
    label_list = []
    for node in nx.nodes(graph):
        label_list.append(label_dict[node])

    data = multiHSD_embed(graph, graphName, hop, n_scales)
    #data = hierarchical_degrees(graph, graphName, hop)
    KNN_evaluate(data, label_list, cv=5, n_neighbor=10)


if __name__ == '__main__':
    run()
