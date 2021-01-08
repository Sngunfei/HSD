# -*- encoding: utf-8 -*-

import networkx as nx
from tools import const

# 如果没有label_type，标注为“default”
def load_data(graph_name: str, label_type: str) -> (nx.Graph, dict):
    if const.System == "Windows":
        edge_path = const.WindowsRootPath + "\data\graph\{}.edgelist".format(graph_name)
        if label_type == "default":
            label_path = const.WindowsRootPath + "\data\label\{}.label".format(graph_name)
        else:
            label_path = const.WindowsRootPath + "\data\label\{}_{}.label".format(graph_name, label_type)
    elif const.System == "Linux":
        edge_path = const.LinuxRootPath + "/data/graph/{}.edgelist".format(graph_name)
        if label_type == "default":
            label_path = const.LinuxRootPath + "/data/label/{}.label".format(graph_name)
        else:
            label_path = const.LinuxRootPath + "/data/label/{}_{}.label".format(graph_name, label_type)
    else:
        raise EnvironmentError("only support Windows and Linux")

    label_dict = read_label(label_path)
    graph = nx.read_edgelist(path=edge_path, create_using=nx.Graph, nodetype=str,
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
    format:
        str(node): int(label)
    """
    with open(path, mode="r", encoding="utf-8") as fin:
        label_dict = dict()
        while True:
            line = fin.readline().strip()
            if not line:
                break
            node, label = line.split(" ")
            label_dict[node] = int(label)
    return label_dict
