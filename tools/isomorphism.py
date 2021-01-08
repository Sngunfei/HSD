# -*- encoding: utf-8 -*-

"""
在图中判断节点是否同构。
    1. 严格：精确匹配
    2. 近似：todo
"""

import networkx as nx
import networkx.algorithms.isomorphism as iso
from functools import cmp_to_key
from collections import defaultdict
from tqdm import tqdm


def get_subgraph_around_node(graph: nx.Graph, node, hop):
    node_set = {node}
    queue = [node]
    for _ in range(hop):
        size = len(queue)
        for _ in range(size):
            cur_node = queue.pop(0)
            neighbors = nx.neighbors(graph, cur_node)
            for neigh in neighbors:
                if neigh not in node_set:
                    queue.append(neigh)
                node_set.add(neigh)
    #print(node, node_set)
    return graph.subgraph(node_set)


def is_isomorphism_between_pair_nodes(G: nx.Graph, node1, node2, hop: int):
    nodes = nx.nodes(G)
    assert (node1 in nodes) and (node2 in nodes), f"node {node1, node2} doesn't exist in graph"

    subgraph1 = get_subgraph_around_node(G, node1, hop)
    subgraph2 = get_subgraph_around_node(G, node2, hop)

    def node_degree_match(node1, node2):
        return nx.degree(G, nodes[node1]) == nx.degree(G, nodes[node2])

    return nx.is_isomorphic(subgraph1, subgraph2)


class UnionFindSet(object):
    def __init__(self, n_node):
        self.root = list(range(n_node))
        self.rank = [0] * n_node

    def find_root(self, node_idx):
        if self.root[node_idx] == node_idx:
            return node_idx
        self.root[node_idx] = self.find_root(self.root[node_idx])
        return self.root[node_idx]

    def union(self, u_idx, v_idx):
        u_root = self.find_root(u_idx)
        v_root = self.find_root(v_idx)

        if self.rank[v_root] > self.rank[u_root]:
            u_root, v_root = v_root, u_root
        self.root[v_root] = u_root

        if self.rank[v_root] == self.rank[u_root]:
            self.rank[u_root] += 1


# 在一张图上找到所有“局部同构”节点，以集合划分，返回集合数组
# 采用并查集实现
def find_isomorphism_nodes(G: nx.Graph, hop: int):
    nodes = list(nx.nodes(G))
    ufs = UnionFindSet(len(nodes))
    for node_idx, node in tqdm(enumerate(nodes)):
        for node_idx2 in range(node_idx+1, len(nodes)):
            node2 = nodes[node_idx2]
            if is_isomorphism_between_pair_nodes(G, node, node2, hop):
                ufs.union(node_idx, node_idx2)

    dict_set = defaultdict(set)
    for node_idx, node in enumerate(nodes):
        dict_set[nodes[ufs.find_root(node_idx)]].add(node)

    return dict_set


# 在图G中找到同构节点集合，并贴上标签。
def get_isomorphism_label(G: nx.Graph, hop: int) -> dict:
    label_dict = {}
    isomorphism_set_dict = find_isomorphism_nodes(G, hop)
    label = 0
    for node_set in isomorphism_set_dict.values():
        for node in node_set:
            label_dict[node] = label
        label += 1

    return label_dict


if __name__ == '__main__':
    graphName = "mkarate"
    hop = 7

    graph = nx.read_edgelist(f"../data/graph/{graphName}.edgelist", create_using=nx.Graph,
                             nodetype=str, edgetype=float, data=[('weight', float)])
    isomorphism_set_dict = find_isomorphism_nodes(graph, hop)
    label = 1
    fout = open(f"../data/label/{graphName}_isomorphism.label", encoding="utf-8", mode="w+")
    for k, v in isomorphism_set_dict.items():
        print(v)
        # for node in v:
        #     fout.write(f"{node} {label}\n")
        # label += 1
    fout.close()


