# -*- encoding: utf-8 -*-

"""
Susceptible - Infected - Recover Model.
"""

import networkx as nx
from utils.util import build_node_idx_map
import copy, random
from collections import defaultdict


class SIR:
    def __init__(self, graph, p=1.0, r=None, t=25, random_state=42):
        """
        :param graph:
        :param p: infect probability
        :param r: recover probability, default：1 / (average degree)
        :param t: spread time
        """

        self.G = graph
        self.p = p
        if r is None:
            self.r = 1.0 / self._calculate_average_degree()
        else:
            self.r = r

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
        """
        Susceptible - Infected - Recover Model
        :return:
        """
        for idx, node in self.idx2node.items():
            self.influence[node] += self._diffuse_from_node(node)


    def _diffuse_from_node(self, origin):
        """

        :param origin: original infected node
        :return:
        """
        infected = set() # infected set
        infected.add((origin, 1))
        recoverd = set() # recovered set

        for _ in range(self.t):
            current_infectd = copy.deepcopy(infected)
            for _node, _weight in current_infectd:
                for neighbor in nx.neighbors(self.G, _node):
                    if neighbor not in recoverd and neighbor not in infected and random.random() <= self.p:
                        infected.add((neighbor, _weight - 0.3)) # decay

                if random.random() <= self.r:
                    recoverd.add((_node, _weight))
                    infected.remove((_node, _weight))

        # structural influence
        influence = sum([weight for _, weight in infected]) + sum([weight for _, weight in recoverd])
        return influence


    def label_nodes(self, n_class):
        """
        divide all nodes into some different groups by structural influence.
        :param n_class: number of class
        :return:
        """
        scores = sorted(self.influence.items(), key=lambda x: x[1])
        labels = {}
        cnt = defaultdict(int)
        interval = len(scores) / n_class
        i = 1
        for idx, (node, score) in enumerate(scores):
            if idx <= interval * i:
                pass
            else:
                i += 1

            labels[node] = i - 1
            cnt[i - 1] += 1
        return labels


def split_nodes(name, graph, n_class, infect_prob=0.9, recover_prob=None, t=5, save_path=None):
    print("Start SIR process, graph:{}, number of class: {}, infect prob: {}, recover prob: {}, time: {}"
          .format(name, n_class, infect_prob, recover_prob, t))

    print("Graph radius: {}".format(nx.radius(graph)))
    print("Graph diameter: {}".format(nx.diameter(graph)))

    model = SIR(graph, p=infect_prob, r=recover_prob, t=t)
    model.start()
    labels = model.label_nodes(n_class)

    if save_path:
        with open(save_path, mode="w+", encoding="utf8") as fout:
            for node, label in labels.items():
                fout.write("{} {] \n".format(node, label))

    return labels


if __name__ == '__main__':
    name = "mkarate"
    graph = nx.read_edgelist(path="../../data/mkarate.edgelist", create_using=nx.Graph,
                            edgetype=float, data=[('weight', float)])
    split_nodes(name, graph, n_class=5, infect_prob=1.0, recover_prob=0.2, t=3)






















