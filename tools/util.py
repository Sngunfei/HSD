# -*- coding:utf-8 -*-

import math

import networkx as nx
import numpy as np
from scipy import sparse

from tools import rw

def build_node_idx_map(graph) -> (dict, dict):
    """
    建立图节点与标号之间的映射关系，方便采样。
    :param graph:
    :return:
    """
    node2idx = {}
    idx2node = {}
    node_size = 0
    for node in nx.nodes(graph):
        node2idx[node] = node_size
        idx2node[node_size] = node
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def compute_chebshev_coeff_basis(scale, order):
    """
    GraphWave: Calculate the chebshev coeff.
    :param scale:
    :param order:
    :return:
    """
    xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * math.pi)
                   for i in range(1, order + 1)])
    basis = [np.ones((1, order)), np.array(xx)]
    for k in range(order + 1 - 2):
        basis.append(2 * np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    products = np.einsum("j,ij->ij", f, basis)
    coeffs = 2.0 / order * products.sum(1)
    coeffs[0] = coeffs[0] / 2
    return list(coeffs)


def sparse_graph(graph: nx.Graph, threshold=None, percentile=None) -> nx.Graph:
    """
    将邻接矩阵稀疏化
    :param graph:
    :param threshold: 权重低于threshold的边将会被删掉
    :param percentile: 按照百分比删边
    :return:
    """
    del_edges = []
    edges = nx.edges(graph)
    sparsed_graph = nx.Graph(graph, sparse="true")
    if threshold:
        for edge in edges:
            u, v = edge
            if graph[u][v]['weight'] < threshold:
                del_edges.append((u, v))
    elif percentile:
        _weights = []
        for edge in edges:
            u, v = edge
            _weights.append(graph[u][v]['weight'])
        # 保留权重较小的边，表示距离相近
        threshold = np.percentile(_weights, (1 - percentile) * 100)
        """
        print("sparse: min: {}, max: {}".format(min(_weights), max(_weights)))
        print("sparse: mean: ", np.mean(_weights))
        print("sparse: median: ", np.median(_weights))
        print("sparse: thereshold: ", threshold)
        """
        for edge in edges:
            u, v = edge
            if graph[u][v]['weight'] > threshold:
                del_edges.append((u, v))

    sparsed_graph.remove_edges_from(del_edges)
    return sparsed_graph

def recommend_scale_range(eignvalues: list) -> (float, float):
    eignvalues = sorted(eignvalues)
    e1, en = eignvalues[0], eignvalues[-1]
    for e in eignvalues:
        if e > 0.001:
            e1 = e
            break
    scale_min, scale_max = scale_boundary(e1, en)
    return scale_min, scale_max


# normalized laplacian
def normalize_laplacian(adj):
    n, _ = adj.shape
    posinv = np.vectorize(lambda x: float(1.0) / np.sqrt(x) if x > 1e-10 else 0.0)
    diag = sparse.diags(np.array(posinv(adj.sum(0))).reshape([-1, ]), 0)
    lap = sparse.eye(n) - diag.dot(adj.dot(diag))
    return lap


# calculate the scale range discussed in GraphWave
def scale_boundary(e1, eN, eta=0.85, gamma=0.95):
    t = np.sqrt(e1 * eN)
    sMax = - np.log(eta) / t
    sMin = - np.log(gamma) / t
    return sMin, sMax


def compare_labels_difference():
    # 同一张图可能有多种标签，比如SIR随时间变化的标签，这个函数用来对比这些标签的不同
    raise NotImplementedError


def filter_edgelist(edgelist: list, save_path: str, ratio=0.05) -> list:
    """
    过滤边，只保留权重前5%大的边，其他边的权重按0处理。
    :return:
    """
    edgelist.sort(key=lambda x: x[2], reverse=True)
    edgelist = edgelist[:int(len(edgelist) * ratio) + 1]

    if save_path is not None:
        rw.save_edgelist(save_path, edgelist)

    return edgelist


def filter_distance_matrix(dist_mat: np.ndarray, nodes: list, save_path: str, ratio=0.05) -> list:
    # 对距离矩阵进行过滤，返回过滤后的边集
    assert dist_mat.shape[0] == dist_mat.shape[1], "距离矩阵必须是方阵"
    assert dist_mat.shape[0] == len(nodes), "距离矩阵的宽度必须和节点数量一致"

    edgelist = []
    for idx1, node1 in enumerate(nodes):
        for idx2 in range(idx1 + 1, len(nodes)):
            node2 = nodes[idx2]
            distance = float(dist_mat[idx1, idx2])
            edgelist.append((node1, node2, distance))

    return filter_edgelist(edgelist, save_path, ratio)


# 将具有相同key的dict，取其value，放在列表的相同位置上对齐。
# 必须有完全相同的键，个数也必须相同。
def merge_dicts_to_lists(*args) -> list:
    n_input = len(args)
    if n_input == 0:
        return []
    dicts = []
    for obj in args:
        if not isinstance(obj, dict):
            raise TypeError("must be dict type")
        dicts.append(obj)

    lists = [[] for _ in range(n_input)]
    keys = dicts[0].keys()
    for k in keys:
        for idx, dic in enumerate(dicts):
            if k not in dic:
                raise KeyError(f"key:{k} not in dict: {dic}")
            v = dic[k]
            lists[idx].append(v)

    for idx in range(n_input):
        if len(lists[idx]) != len(dicts[idx]):
            raise ValueError("size of list and dict not equal")

    return lists


