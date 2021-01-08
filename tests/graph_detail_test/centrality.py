# -*- encoding: utf-8 -*-

import networkx as nx
import numpy as np
from tools.dataloader import read_label
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

GraphName = ""


def degree_centrality(graph: nx.Graph):
    # 分析图中有哪些独立的节点，度数如何
    return None


"""
While the degree based centrality considers a node with many neighbors as important, 
it treats all the neighbors equally. However, the neighbors themselves can have different importance; 
thus they could affect the importance of the central node differently. 
The eigenvector centrality (Bonacich, 1972, 2007) defines
the centrality score of a given node vi by considering the centrality scores of
its neighboring nodes as:

邻接矩阵A的最大特征向量
"""
def eigenvector_centrality(graph: nx.Graph):
    global GraphName

    # A = nx.adjacency_matrix(graph).todense()
    # print(A)
    # eigenvalues, eigenvectors = np.linalg.eigh(A)
    # print(eigenvalues)
    # print(eigenvectors[-2])
    central = nx.eigenvector_centrality(graph, max_iter=1000)
    scores = sorted(central.items(), key=lambda x: x[1])

    interval = len(scores) / 5
    labels = {}
    label = 1
    for idx, (node, score) in enumerate(scores):
        labels[node] = label
        if idx >= interval * label:
            label += 1

    # SIR_labels = read_label(f"../../data/label/{GraphName}.label")
    #
    # labels1, labels2 = [], []
    # for node, label in labels.items():
    #     labels1.append(label)
    #     labels2.append(SIR_labels[node])
    # print(accuracy_score(labels1, labels2))
    #
    PageRank_scores = nx.pagerank(graph, max_iter=1000)

    plt.figure()
    xs = list(range(nx.number_of_nodes(graph)))
    plt.ylim(0, 0.3)
    plt.ylabel("score value")
    plt.xlabel("node index")
    plt.title(GraphName)
    plt.plot(xs, sorted(PageRank_scores.values()), label="PageRank")
    plt.plot(xs, sorted(central.values()), label="Eigenvector Centrality")
    plt.legend()
    plt.show()

    with open(f"../../data/label/{GraphName}_EigenCentrality.label", mode="w+", encoding="utf-8") as fout:
        for node, label in labels.items():
            fout.write(f"{node} {label}\n")


def katz_centrality(graph: nx.Graph):

    return


def PageRank_centrality(graph: nx.Graph):
    central = nx.pagerank(graph, max_iter=1000)
    scores = sorted(central.items(), key=lambda x: x[1])
    print(scores)
    ans = 0
    for node, score in scores:
        ans += score * score
    print(ans)
    return


def run():
    global GraphName
    GraphName = "mkarate"
    graph = nx.read_edgelist(f"../../data/graph/{GraphName}.edgelist", create_using=nx.Graph, edgetype=float,
                             data=[('weight', float)])
    #PageRank_centrality(graph)
    eigenvector_centrality(graph)

if __name__ == '__main__':
    run()
