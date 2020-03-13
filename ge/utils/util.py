# -*- coding:utf-8 -*-

"""
一些常用的工具函数
"""

import numpy as np
import networkx as nx
from tqdm import tqdm
import math


def write_subway_label(data_path):
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, nodetype=str, edgetype=float, data=[('weight', float)])
    fout = open("G:\pyworkspace\graph-embedding\out\subway_label_2.txt", mode="w+", encoding="utf-8")
    nodes = list(nx.nodes(graph))
    rings = dict()
    for node1 in nodes:
        hop1, hop2 = 0, 0
        for node2 in nodes:
            length = nx.dijkstra_path_length(graph, node1, node2)
            if length > 2:
                continue
            elif length == 1:
                hop1 += 1
            elif length == 2:
                hop2 += 1
        rings[node1] = [hop1, hop2]

    for node, hop in rings.items():
        hop1 = min(hop[0], 4)
        hop2 = min(hop[1], 6) // 3 + 1
        label = (hop1 - 1) * 3 + hop2
        fout.write("{} {}\n".format(node, label))

    fout.close()


def write_label(name="", max_hop=10, hops_weight=None, percentiles=None):
    """
    给节点贴标签
    :param name:
    :param max_hop: 最多考虑多少层
    :param hops_weight: 每层的权重
    :param percentiles: 贴标签的百分位数
    :return:
    """
    graph, _, _ = dataloader(name=name)
    if not hops_weight:
        hops_weight = np.array([1.0 / hop for hop in range(1, max_hop + 2)])
    if not percentiles:
        percentiles = np.array([20, 40, 60, 80], dtype=np.float)

    idx2node, node2idx = build_node_idx_map(graph)
    scores = np.zeros_like(idx2node, dtype=np.float)

    for idx, node in tqdm(enumerate(idx2node)):
        degrees = np.zeros(max_hop + 1)
        queue = [node]
        visited = [node]
        hop = 0
        while queue and hop <= max_hop:
            n_cur_nodes = len(queue)
            for _ in range(n_cur_nodes):
                _node = queue.pop(0)
                degrees[hop] += nx.degree(graph, _node)
                next_hop_neighbors = list(nx.neighbors(graph, _node))
                for _neighbor in next_hop_neighbors:
                    if _neighbor not in visited:
                        queue.append(_neighbor)
                        visited.append(_neighbor)
            hop += 1
        score = np.dot(degrees, hops_weight)
        scores[idx] = score
    labels = np.zeros_like(scores)
    percentiles_value = np.percentile(scores, percentiles)
    print(percentiles_value)
    n_class = len(percentiles)
    labels[scores < percentiles_value[0]] = 0
    labels[scores > percentiles_value[-1]] = n_class
    for i in range(1, n_class):
        idxs1 = scores >= percentiles_value[i-1]
        idxs2 = scores < percentiles_value[i]
        idxs = np.bitwise_and(idxs1, idxs2)
        labels[idxs] = i
    labels = labels.astype(np.int)
    with open("../../data/{}_auto.label".format(name), mode='w+', encoding='utf8') as f:
        for idx, node in enumerate(idx2node):
            f.write("{} {}\n".format(node, labels[idx]))
    return labels


def add_inverse_edges(graph: nx.DiGraph):
    edges = graph.edges()
    inv_edges = []
    for edge in edges:
        u, v = edge[0], edge[1]
        inv_edges.append((v, u))
    graph.add_edges_from(inv_edges)
    return graph


def build_node_idx_map(graph):
    """
    建立图节点与标号之间的映射关系，方便采样。
    :param graph:
    :return:
    """
    node2idx = {}
    idx2node = {}
    node_size = 0
    for node in graph.nodes():
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
    if threshold:
        for edge in edges:
            u, v = edge
            if graph[u][v]['weight'] < threshold:
                del_edges.append((u, v))
    elif percentile:
        _weights = []
        for edge in edges:
            u, v = edge
            # if one edge's weight very big, then would delete many valuable edges.
            #thres += 1.0 / n * (self.graph[u][v]['weight'] * percentile - thres)
            _weights.append(graph[u][v]['weight'])
        threshold = np.percentile(_weights, percentile * 100)

        for edge in edges:
            u, v = edge
            if graph[u][v]['weight'] <= threshold:
                del_edges.append((u, v))

    graph.remove_edges_from(del_edges)
    return graph


def connect_graph(graph: nx.Graph) -> nx.Graph:
    """
    一个图可能有多个连通分量，为了使其之间能够有路径到达，在不同连通分量之间添加一条边，使其连通
    :param graph: 原图
    :return: 连通图
    """

    # if graph is connected already.
    if nx.is_connected(graph):
        return graph

    # n_components > 1
    components = nx.connected_components(graph)
    node = None
    edges = []
    for comp in components:
        if node is None:
            node = comp.pop()
        else:
            edges.append((node, comp.pop()))  # 随机加边

    graph.add_edges_from(edges)
    return graph


def get_metadata_of_networks():
    networks = ["bell", "mkarate", "europe", "usa"]

    for name in networks:
        data, _, _ = dataloader(name, directed=False)
        print(name)
        print("radius", nx.radius(data))
        print("diameter", nx.diameter(data))
        print("Average Degree:", np.mean([j for _, j in nx.degree(data)]))



def classify_nodes_by_degree(graph):
    graph, _, _ = load_data(data, directed=False, label="SIR")
    node_degree = {}
    for node, degree in nx.degree(graph):
        node_degree[node] = degree

    res = sorted(node_degree.items(), key=lambda x:x[1])

    fout = open("../../data/usa_degree.label", mode="w+", encoding="utf-8")
    for node, degree in res:
        fout.write("{} {}\n".format(node, degree))
    fout.close()


def plot_vectors(path):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    label_dict, n_class = read_label("E:\workspace\py\graph-embedding\data\\europe_SIR.label")
    fin = open(path, mode="r", encoding="utf-8")
    nodes, _2d_data = [], []
    labels = []
    while True:
        line = fin.readline()
        if not line:
            break
        node, xx, yy = line.strip().split(",")
        nodes.append(int(node))
        labels.append(label_dict[node])
        _2d_data.append([float(xx), float(yy)])

    _2d_data = np.asarray(_2d_data)
    markers = ['o', '*', 'x', '<', '1', 'x', 'D', '>', '^', "v", '1', '2', '3', '4', 'X', '.']

    cm = plt.get_cmap("nipy_spectral")
    cNorm = colors.Normalize(vmin=0, vmax=n_class - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    class_dict = defaultdict(list)
    for idx, node in enumerate(nodes):
        class_dict[int(labels[idx])].append(idx)

    info = sorted(class_dict.items(), key=lambda item: item[0])
    for _class, _indices in info:
        # general case， n_class < 10
        # plt.scatter(_2d_data[_indices, 0], _2d_data[_indices, 1], s=40, marker=markers[_class], cmap=plt.get_cmap("nipy_spectral"))
        # mirror karate network, n_class = 34
        plt.scatter(_2d_data[_indices, 0], _2d_data[_indices, 1], s=100, marker=markers[_class % len(markers)],
                    c=[scalarMap.to_rgba(_class)], label=_class)
    plt.show()
