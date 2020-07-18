# -*- encoding: utf-8 -*-

"""
PRD：对图的层级划分不应该内嵌到具体的模型中，而是抽象出一套公共方法
能够实现对图的层级处理，比如io操作，省去每次模型run的时候都去重构一套
"""

import networkx as nx
import copy
from tqdm import tqdm
import os

from ge.utils.dataloader import load_data

Path = "E:\workspace\py\graph-embedding\data\hierarchy\{}.hie"

def get_hierarchical_representation(graph: nx.DiGraph, layer_cnt=5):
    """
    无向图是特殊的有向图，如果要对无向图进行构建，只需要传参的时候DiGraph(g)即可
    :param graph:
    :param layer_cnt: 分为几个层级，节点本身也算一层
    :return:
    """
    nodes = nx.nodes(graph)
    hierarchy = {}
    for node in tqdm(nodes):
        rings = [[node]]
        queue = [node] # 利用队列进行层级遍历
        visited = [] # 记录已经查过的，按最短路径划分层次
        for layer in range(layer_cnt):
            capacity = len(queue)
            if capacity == 0:
                break
            for _ in range(capacity):
                neighbor = queue.pop(0)
                if neighbor in visited:
                    continue
                visited.append(neighbor)
                for next_layer_neighbor in nx.neighbors(graph, neighbor):
                    if next_layer_neighbor in visited:
                        continue
                    queue.append(next_layer_neighbor)
            rings.append(copy.deepcopy(queue))
        hierarchy[node] = rings
    return hierarchy


def get_node_hierarchical_structure(graph: nx.DiGraph, node, layer_cnt=5):
    """
    构建node在graph中层级结构表示
    :param graph:
    :param node:
    :param layer_cnt:
    :return:
    """
    rings = [[node]]
    queue = [node]  # 利用队列进行层级遍历
    visited = set()  # 记录已经查过的，按最短路径划分层次
    visited.add(node)
    for _ in range(layer_cnt):
        capacity = len(queue)
        if capacity == 0:
            break
        for _ in range(capacity):
            neighbor = queue.pop(0)
            for next_layer_neighbor in nx.neighbors(graph, neighbor):
                if next_layer_neighbor in visited:
                    continue
                queue.append(next_layer_neighbor)
                visited.add(next_layer_neighbor)
        rings.append(copy.deepcopy(queue))
    return rings


def save_hierarchical_representation(graph_name: str, graph):
    """
    将层级结构存到文件里，以备后续重复使用，因为高层级包含低层级，所以直接存高层级的
    layer_cnt = 10
    :param graph_name:
    :param graph:
    :return:
    """
    nodes = nx.nodes(graph)
    layer_cnt = 5
#    hierarchy = get_hierarchical_representation(graph, layer_cnt=10)
    file_path = Path.format(graph_name)
    with open(file_path, encoding="utf-8", mode="w+") as fout:
        # 每个节点占一行，行首node为自己，同时也为第一层
        # 然后每一层之间的节点，由#分隔，层内节点由逗号分隔
        for node in tqdm(nodes):
            record = ""
            rings = get_node_hierarchical_structure(graph, node, layer_cnt)
            for layer_nodes in rings:
                if len(layer_nodes) == 0:
                    break
                for idx, neighbor in enumerate(layer_nodes):
                    if idx == len(layer_nodes) - 1:
                        record += str(neighbor) + '#'
                    else:
                        record += str(neighbor) + ','
            fout.write(record + '\n')
            fout.flush()
    print("Save hierarchical representation done!\n")


def read_hierarchical_representation(graph_name: str, layer_cnt=5) -> dict:
    path = Path.format(graph_name)
    hierarchy = {}
    if not os.path.exists(path):
        return hierarchy
    with open(path, mode="r", encoding="utf-8") as fin:
        while True:
            line = fin.readline().strip()
            if not line:
                break
            rings = {}
            for idx, ring in enumerate(line.split("#")):
                if idx >= layer_cnt:
                    break
                neighbors = ring.split(",")
                rings[idx] = neighbors
            hierarchy[rings[0][0]] = rings
    return hierarchy

if __name__ == '__main__':
    graphs =["europe", "bio_dmela", "bio_grid_human"]
    for graph_name in graphs:
        ring_avg_cap = [0] * 10
        res = read_hierarchical_representation(graph_name, layer_cnt=5)
        n_nodes = len(res)
        for node, rings in res.items():
            for idx, ring in enumerate(rings):
                ring_avg_cap[idx] += len(ring)
        print([degree / n_nodes for degree in ring_avg_cap])
        # graph = nx.read_edgelist(path=f"E:\workspace\py\graph-embedding\data\graph\\{graph_name}.edgelist", create_using=nx.Graph,
        #                          edgetype=float, data=[('weight', float)])
        # save_hierarchical_representation(graph_name, graph)
