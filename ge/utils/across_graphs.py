# -*- encoding: utf-8 -*-

"""
跨图分析，因为SIR的模型评分会受到图规模的影响

我们决定选取一张图，然后对其进行改造，如删边，删节点，

这样会得到性质比较接近的两张图

但要注意，HSD本身并不受到SIR模型的影响，
"""
from utils.util import dataloader
from utils.robustness import random_remove_edges
import networkx as nx

def across_graphs(name):
    graph, _, _ = dataloader("europe", directed=False)
    graph1 = random_remove_edges(graph, prob=0.7)
    graph, _, _ = dataloader("europe", directed=False)
    graph2 = random_remove_edges(graph, prob=0.7)
    fout = open("../../data/across_{}.edgelist".format(name), mode="w+", encoding="utf8")

    for edge in nx.edges(graph1):
        u, v = edge[0], edge[1]
        fout.write("{} {}\n".format(u, v))
    n_nodes = nx.number_of_nodes(graph1)
    fout.write("---------\n")
    for edge in nx.edges(graph2):
        u, v = int(edge[0]) + n_nodes, int(edge[1]) + n_nodes
        fout.write("{} {}\n".format(u, v))

    fout.close()


if __name__ == '__main__':
    across_graphs("europe")