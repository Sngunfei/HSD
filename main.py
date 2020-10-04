# -*- encoding: utf-8 -*-

from model import MultiHSD
from tools import evaluate, dataloader, rw, metrics
import networkx as nx
import numpy as np


def base_HSD_Test():
    pass


def multi_HSD_Test(graphName, hop=3, n_scales=200):
    graph = nx.read_edgelist(f"data/graph/{graphName}.edgelist", create_using=nx.Graph,edgetype=float, data=[('weight', float)])
    label_dict = dataloader.read_label(f"data/label/{graphName}.label")

    model = MultiHSD(graph, graphName, hop, n_scales)
    model.init()
    embedding_dict = model.parallel_embed(n_workers=10)

    #rw.save_vectors_dict(embedding_dict, f"output/multi_HSD_{graphName}_{n_scales}.csv")

    embeddings, labels = [], []
    for node, vector in embedding_dict.items():
        embeddings.append(vector)
        labels.append(label_dict[node])
    knn_score = evaluate.KNN_evaluate(embeddings, labels)
    lr_score = evaluate.LR_evaluate(embeddings, labels)

    # hellinger distance
    # dists = np.zeros((model.n_node, model.n_node), dtype=np.float)
    # step = hop + 1
    # for idx1 in range(model.n_node):
    #     for idx2 in range(idx1+1, model.n_node):
    #         cur_idx = 0
    #         while cur_idx + step <= len(embeddings[0]):
    #             dists[idx1][idx2] += metrics.hellinger_distance(p=embeddings[idx1][cur_idx:cur_idx+step], q=embeddings[idx2][cur_idx:cur_idx+step])
    #             cur_idx += step
    #         dists[idx2][idx1] = dists[idx1][idx2]
    # print("hellinger")
    # knn_score = evaluate.KNN_evaluate(dists, labels, metric="precomputed")

    return knn_score, lr_score


def dynamic_HSD_Test():
    pass


def evaluate_embeddings():
    method = "rolx"
    graphName = "usa"
    candidates = list(range(1, 17))
    candidates.extend([32, 64, 128, 256, 512, 1024])
    for dimension in candidates:
        embedding_dict = rw.read_vectors(f"output/{method}_{graphName}_{dimension}.csv")
        label_dict = dataloader.read_label(f"data/label/{graphName}.label")

        embeddings, labels = [], []
        for node, vector in embedding_dict.items():
            embeddings.append(vector)
            labels.append(label_dict[node])
        print(f"{method}, {graphName}, dimension: {dimension}")
        evaluate.KNN_evaluate(embeddings, labels)
        evaluate.LR_evaluate(embeddings, labels)


if __name__ == '__main__':
    taus = [i * 10 for i in range(1, 31)]
    graphs = ["bio_dmela", "bio_grid_human", "usa"]
    for name in graphs:
        print(name)
        for t in taus:
            with open(f"{name}_score.txt", mode="a+", encoding="utf-8") as fout:
                for _ in range(10):
                    print(f"graph:{name}, n_scales:{t}\n")
                    c1, c2 = multi_HSD_Test(name, n_scales=t)
                    fout.write(f"n_scales: {t}, knn score: {c1}, lr_score: {c2}\n")
                    fout.flush()
    #evaluate_embeddings()
