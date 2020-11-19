# -*- encoding: utf-8 -*-

"""
Susceptible - Infected - Recover Model.
"""

import math
from collections import defaultdict

import networkx as nx
from tqdm import tqdm

from tools import build_node_idx_map

class SIR:
    def __init__(self, graph: nx.Graph, p=1.0, t=25, random_state=42):
        """
        :param p: infect probability
        :param t: spread time
        """

        self.G = graph
        self.p = p
        self.t = t
        self.random_state = random_state
        self.idx2node, self.node2idx = build_node_idx_map(graph)
        self.influence = defaultdict(int)


    def _calculate_average_degree(self):
        d = 0
        for _, degree in nx.degree(self.G):
            d += degree
        return d / nx.number_of_nodes(self.G)


    def start(self):
        for _ in tqdm(range(1)):
            for idx, node in tqdm(self.idx2node.items()):
                self.influence[node] += self._diffuse_from_node(node)


    def _diffuse_from_node(self, origin):
        """
        从原始节点出发，向外传染，如果仅仅统计感染个数，则无法很好的区分不同阶邻居，所以需要带衰减的感染
        weight = e^(-0.5 * t)
        """
        cur_infected = [origin]
        all_infected = {origin}
        influence = 0.0
        for iteration in range(self.t):
            if len(cur_infected) == 0:
                break
            n = len(cur_infected)
            for _ in range(n):
                cur = cur_infected.pop(0)
                influence += math.e ** (-0.5 * iteration)

                for neighbor in nx.neighbors(self.G, cur):
                    if neighbor not in all_infected:
                        cur_infected.append(neighbor)
                        all_infected.add(neighbor)
        return influence


    def label_nodes(self, n_class):
        """
        Split all nodes into different groups using structural influence.
        """
        scores = sorted(self.influence.items(), key=lambda x: x[1])
        interval = len(scores) / n_class

        labels = {}
        label = 1
        for idx, (node, score) in enumerate(scores):
            labels[node] = label
            if idx >= interval * label:
                label += 1

        return labels

def split_nodes(graph, n_class, infect_prob, t=5, save_path=None) -> dict:
    model = SIR(graph, p=infect_prob, t=t)
    model.start()
    labels = model.label_nodes(n_class)

    if save_path:
        with open(save_path, mode="w+", encoding="utf-8") as fout:
            for node, label in sorted(labels.items(), key=lambda x: int(x[0])):
                fout.write("{} {}\n".format(node, label))
    return labels

def get_SIR_labels(g: nx.Graph, n: int, t: int, infect_p: float) -> dict:
    # 快速得到一张图的SIR标签信息
    sir = SIR(g, infect_p, t, )
    sir.start()
    labels = sir.label_nodes(n_class=n)
    return labels


if __name__ == '__main__':
    graphName = "cora"
    save_path = f"../data/label/{graphName}.label"
    graph = nx.read_edgelist(path=f"../data/graph/{graphName}.edgelist", create_using=nx.Graph,
                            edgetype=float, data=[('weight', float)])
    split_nodes(graph, n_class=5, infect_prob=1.0, t=5, save_path=save_path)
