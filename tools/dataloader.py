# -*- encoding: utf-8 -*-

"""
load graph data
"""

import networkx as nx


def load_data(graphName, directed=False) -> (nx.Graph, dict):
    """
    Loda graph data by dataset name.
    :param graphName: graph name, e.g. mkarate
    :param directed: bool, if True, return directed graph.
    :return: graph, node labels, number of node classes.
    """

    edge_path = "../data/graph/{}.edgelist".format(graphName)
    label_path = "../data/label/{}.label".format(graphName)
    label_dict = read_label(label_path)
    graph = nx.read_edgelist(path=edge_path, create_using=nx.DiGraph if directed else nx.Graph,
                             edgetype=float, data=[('weight', float)])
    return graph, label_dict


def load_data_from_distance(graph_name, label_name, metric, hop, scale, multi="no", directed=False):
    """
    Loda graph data by dataset name.
    :param graph_name: graph name, e.g. mkarate
    :param label_name: label name, e.g. mkarate_origin
    :param directed: bool, if True, return directed graph.
    :return: graph, node labels, number of node classes.
    """
    if multi == "yes":
        edge_path = "../distance/{}/HSD_multi_{}_hop{}.edgelist".format(
                    graph_name, metric, hop)
    else:
        edge_path = "../distance/{}/HSD_{}_scale{}_hop{}.edgelist".format(
                    graph_name, metric, scale, hop)

    label_path = f"../data/label/{graph_name}.label"
    label_dict, n_class = read_label(label_path)

    if directed:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.DiGraph,
                                 edgetype=float, data=[('weight', float)])
    else:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.Graph,
                                 edgetype=float, data=[('weight', float)])

    return graph, label_dict, n_class


def read_label(path) -> dict:
    """
    read graph node labels.
    :param path: label file path.
    :return: return dict-type, {node:label}, number of class.
    """
    try:
        with open(path, mode="r", encoding="utf-8") as fin:
            label_dict = dict()
            while True:
                line = fin.readline().strip()
                if not line:
                    break
                node, label = line.split(" ")
                # node本身标号用str表示，为了和矩阵中的idx区分开来，防止混淆。
                label_dict[node] = int(label)
        return label_dict
    except FileNotFoundError:
        print("Warning: Label file: {} not found.".format(path))
        return {}
