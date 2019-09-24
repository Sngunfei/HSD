# -*- coding:utf-8 -*-

import numpy as np
import networkx as nx
from tqdm import tqdm


def dataloader(name="", directed=False, auto_label=False, similarity=False, scale=None, metric=None):
    """
    Loda graph data by dataset name.
    :param name: dataset name, str
    :param directed: bool, if True, return directed graph.
    :param similarity: similarity data or edgelist.
    :param scale: i.e. head coefficient, int
    :param metric: similarity metric, like L1 and L2, etc.
    :return: graph, node labels, number of node classes.
    """
    if auto_label:
        label_path = "../../data/{}_auto.label".format(name)
    else:
        label_path = "../../data/{}.label".format(name)
    if not similarity:
        edge_path = "../../data/{}.edgelist".format(name)
    else:
        metric = str.lower(metric)
        edge_path = "../../similarity/{}_{}_{}.csv".format(name, scale, metric)
        directed = False # similarity can't be directed.

    if directed:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.DiGraph,
                                 edgetype=float, data=[('weight', float)])
    else:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.Graph,
                                 edgetype=float, data=[('weight', float)])

    label_dict, num_class = read_label(label_path)

    return graph, label_dict, num_class


def read_label(path):
    """
    Get graph nodes' label.
    :param path: label file path.
    :return: return dict-type, {node:label}, number of class.
    """
    try:
        label_set = set()
        with open(path, mode="r", encoding="utf-8") as fin:
            label_dict = dict()
            while True:
                line = fin.readline()
                if not line:
                    break
                node, label = line.strip().split(" ")
                label_dict[node] = label
                label_set.add(label)
        return label_dict, len(label_set)

    except FileNotFoundError:
        print("Warning: Label file: {} not found.".format(path))
        return None, 0


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


def build_node_idx_map(graph):
    """
    建立图节点与标号之间的映射关系，方便采样。
    :param graph:
    :return:
    """
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
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


def compute_cheb_coeff_basis(scale, order):
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


if __name__ == '__main__':
    write_label(name="europe")
