# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

"""
多个连通图分开计算，和一个不连通图错开计算有区别吗？
"""

class SimilarityAcrossGraphs(object):

    def __init__(self, graphs: list, metric="wasserstein", n_layers=5):
        """
        :param graphs:
        :param metric:
        :param n_layers:
        """
        self.graphs = graphs
        self.nodes = [list(nx.nodes(g)) for g in graphs]
        self.n_nodes = [len(nodes) for nodes in self.nodes]
        self.metric = metric
        self.n_layers = n_layers


    def parallel_calc(self):
        """
        并行计算各个节点的相似性，但每个节点要贴一个前缀。
        :return:
        """
        pass




def t():
    from scipy.linalg import block_diag
    a = np.random.randn(5, 5)
    a = np.triu(a)
    a += a.T - np.diag(a.diagonal())

    b = np.random.randn(6, 6)
    b = np.triu(b)
    b += b.T - np.diag(b.diagonal())

    c = np.random.randn(7, 7)
    c = np.triu(c)
    c += c.T - np.diag(c.diagonal())

    e1, v1 = np.linalg.eig(a)
    e2, v2 = np.linalg.eig(b)
    e3, v3 = np.linalg.eig(c)

    print("a", e1, v1)
    print("b", e2, v2)
    print("c", e3, v3)

    d = np.asarray(block_diag(a, b, c))
    print(d.shape)

    e4, v4 = np.linalg.eig(d)
    print("d", e4, v4)


def f():
    f1 = open("G:\pyworkspace\graph-embedding\data\\bell_origin.label", mode='r', encoding="utf-8")
    fout = open("G:\pyworkspace\graph-embedding\data\\bell2_origin.label", mode='w+', encoding="utf-8")
    while True:
        line = f1.readline()
        if not line:
            break
        a, b = line.strip().split(" ")
        fout.write("{} {}\n".format(int(a)+100, b))
    f1.close()
    fout.close()


if __name__ == '__main__':
    f()

