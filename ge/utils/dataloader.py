# -*- encoding: utf-8 -*-

"""
Load graph data and labels.
"""

import networkx as nx


def load_data(graph_name, label_name, distance=False, directed=False):
    """
    Loda graph data by dataset name.
    :param graph_name: graph name, e.g. mkarate
    :param label_name: label name, e.g. mkarate_origin
    :param directed: bool, if True, return directed graph.
    :return: graph, node labels, number of node classes.
    """

    edge_path = "../data/graph/{}.edgelist".format(graph_name)
    label_path = "../data/label/{}_{}.label".format(graph_name, label_name)

    label_dict, n_class = read_label(label_path)

    if directed:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.DiGraph,
                                 edgetype=float, data=[('weight', float)])
    else:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.Graph,
                                 edgetype=float, data=[('weight', float)])

    return graph, label_dict, n_class


def load_data_from_distance(graph_name, label_name, metric, hop, scale, multi="no", directed=False):
    """
    Loda graph data by dataset name.
    :param graph_name: graph name, e.g. mkarate
    :param label_name: label name, e.g. mkarate_origin
    :param directed: bool, if True, return directed graph.
    :return: graph, node labels, number of node classes.
    """
    if multi == "yes":
        edge_path = "../distance/HSD_multi_{}_{}_hop{}.edgelist".format(
                    graph_name, metric, hop)
    else:
        edge_path = "../distance/HSD_{}_{}_scale{}_hop{}.edgelist".format(
                    graph_name, metric, scale, hop)

    label_path = "../data/label/{}_{}.label".format(graph_name, label_name)

    label_dict, n_class = read_label(label_path)

    if directed:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.DiGraph,
                                 edgetype=float, data=[('weight', float)])
    else:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.Graph,
                                 edgetype=float, data=[('weight', float)])

    return graph, label_dict, n_class


def read_label(path):
    """
    read graph node labels.
    :param path: label file path.
    :return: return dict-type, {node:label}, number of class.
    """
    try:
        n_label = set()
        with open(path, mode="r", encoding="utf-8") as fin:
            label_dict = dict()
            while True:
                line = fin.readline()
                if not line:
                    break
                node, label = line.strip().split(" ")
                label_dict[node] = label
                n_label.add(label)
        return label_dict, len(n_label)

    except FileNotFoundError:
        print("Warning: Label file: {} not found.".format(path))
        return None, 0
