# -*- encoding: utf-8 -*-

# 测试各个标准贴的label，内部具体分数

import matplotlib.pyplot as plt
import networkx as nx

from tools import SIR

def SIR_test(graphName):
    graph = nx.read_edgelist(path=f"../../data/graph/{graphName}.edgelist", create_using=nx.Graph,
                             edgetype=float, data=[('weight', float)])
    sir = SIR.SIR(graph, t=5)
    sir.start()

    scores = sorted(sir.influence.items(), key=lambda x: x[1])
    with open(f"{graphName}_SIR.label", mode="w+", encoding="utf-8") as fout:
        for node, score in scores:
            fout.write(f"{node}, {score}\n")

def PageRank_test(graphName):
    graph = nx.read_edgelist(path=f"../../data/graph/{graphName}.edgelist", create_using=nx.Graph,
                             edgetype=float, data=[('weight', float)])
    ranks = nx.pagerank(graph, max_iter=1000)
    scores = sorted(ranks.items(), key=lambda x: x[1])
    with open(f"{graphName}_PageRank.label", mode="w+", encoding="utf-8") as fout:
        for node, score in scores:
            fout.write(f"{node}, {score}\n")

def plot_scores(graphName):
    graph = nx.read_edgelist(path=f"../../data/graph/{graphName}.edgelist", create_using=nx.Graph,
                             edgetype=float, data=[('weight', float)])
    sir = SIR.SIR(graph, t=5)
    sir.start()
    scores1 = sorted(sir.influence.values())
    sum_scores1 = sum(scores1)
    #scores1 = [score / sum_scores1 for score in scores1]
    ranks = nx.pagerank(graph, max_iter=1000)
    scores2 = sorted(ranks.values())
    x = list(range(len(scores1)))

    checkpoints = [i * len(scores2) // 5 for i in range(1, 5)]

    plt.figure()
    plt.subplot(211)

    plt.plot(x, scores1, label="SIR")
    for point in checkpoints:
        plt.axvline(x=point, ls="--", c="green")
        plt.scatter([point], [scores1[point]])
        plt.text(point, scores1[point] + 1, '%.0f' % scores1[point], ha='center', va='bottom', fontsize=11)

    plt.yticks([])
    plt.legend()

    plt.subplot(212)
    plt.plot(x, scores2, label="PageRank")
    for point in checkpoints:
        plt.axvline(x=point, ls="--", c="green")
        plt.scatter([point], [scores2[point]], )
        plt.text(point, scores2[point], '%.6f' % scores2[point], ha='center', va='bottom', fontsize=11)
    plt.yticks([])
    plt.xlabel(graphName)
    plt.legend()
    plt.savefig(f"{graphName}.png")

    plt.show()

def plot_scores_simulately(graphName):
    graph = nx.read_edgelist(path=f"../../data/graph/{graphName}.edgelist", create_using=nx.Graph,
                             edgetype=float, data=[('weight', float)])
    sir = SIR.SIR(graph, t=5)
    sir.start()
    scores1 = sorted(sir.influence.values())
    sum_scores1 = sum(scores1)
    scores1 = [score / sum_scores1 for score in scores1]

    ranks = nx.pagerank(graph, max_iter=1000)
    scores2 = sorted(ranks.values())
    x = list(range(len(scores1)))

    plt.figure()
    plt.plot(x, scores1, label="SIR")
    plt.plot(x, scores2, label="PageRank")

    checkpoints = [i * len(scores2) // 5 for i in range(1, 5)]
    for point in checkpoints:
        plt.axvline(x=point, ls="--", c="green")
        plt.scatter([point], [scores1[point]], )
        plt.scatter([point], [scores2[point]], )

        plt.text(point, scores1[point], '%.6f' % scores1[point], ha='center', va='bottom', fontsize=11)
        plt.text(point, scores2[point], '%.6f' % scores2[point], ha='center', va='bottom', fontsize=11)

    plt.xlabel(graphName)
    #plt.yticks([])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # graphs = ['europe', 'usa', 'cora', 'bio_dmela', 'bio_grid_human']
    # for name in graphs:
    #     plot_scores(name)
    plot_scores_simulately("bio_grid_human")
