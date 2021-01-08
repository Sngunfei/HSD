# -*- encoding: utf-8 -*-

import networkx as nx
import numpy as np

from model import MultiHSD, HSD
from tools import evaluate, dataloader, rw, util

def base_HSD_Test(graphName, hop=3, metric="wasserstein"):
    graph = nx.read_edgelist(f"data/graph/{graphName}.edgelist", create_using=nx.Graph, edgetype=float,
                             data=[('weight', float)])
    label_dict = dataloader.read_label(f"data/label/{graphName}.label")

    model = HSD(graph, graphName, 0, hop, metric)
    model.construct_hierarchy()

    labels = []
    for idx, node in enumerate(model.nodes):
        labels.append(label_dict[node])

    model.eigenvalues, model.eigenvectors = np.linalg.eigh(model.laplacian)
    scale_min, scale_max = util.recommend_scale_range(list(model.eigenvalues))

    score_max, scale_opt = 0, 0
    for scale in np.linspace(scale_min, scale_max, num=5, dtype=np.float):
        model.scale = scale
        model.calculate_wavelets(model.scale, approx=True)
        dists = model.parallel_calculate_HSD(n_workers=10)
        knn_score = evaluate.KNN_evaluate(dists, labels, metric="precomputed", cv=10, n_neighbor=20)
        if knn_score > score_max:
            score_max = knn_score
            scale_opt = scale
    print(f"max score: {score_max}, optimal scale: {scale_opt}\n")
    return score_max, scale_opt

def multi_HSD_Test(graphName, hop=3, n_scales=200, cv=5, n_neighbor=10):
    graph = nx.read_edgelist(f"data/graph/{graphName}.edgelist", create_using=nx.Graph, edgetype=float,
                             data=[('weight', float)])
    PageRank_label_dict = dataloader.read_label(f"data/label/{graphName}_PageRank.label")
    SIR_label_dict = dataloader.read_label(f"data/label/{graphName}.label")

    model = MultiHSD(graph, graphName, hop, n_scales)
    model.init()
    embedding_dict = model.parallel_embed(n_workers=10)

    embeddings, SIR_labels, PageRank_labels = [], [], []
    for node, vector in embedding_dict.items():
        embeddings.append(vector)
        SIR_labels.append(SIR_label_dict[node])
        PageRank_labels.append(PageRank_label_dict[node])
    SIR_val = evaluate.KNN_evaluate(embeddings, SIR_labels, cv=cv, n_neighbor=n_neighbor)
    PageRank_val = evaluate.KNN_evaluate(embeddings, PageRank_labels, cv=cv, n_neighbor=n_neighbor)
    return SIR_val, PageRank_val
    #lr_score = evaluate.LR_evaluate(embeddings, labels)

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

    #return knn_score, lr_score


def dynamic_HSD_Test():
    pass

def evaluate_embeddings():
    method = "multi-HSD"
    graphName = "europe"
    candidates = list(range(1, 17))
    candidates.extend([32, 64, 128])  #, 256, 512, 1024])
    SIR_val = 0.0
    PageRank_val = 0.0
    for dimension in candidates:
        embedding_dict = rw.read_vectors(f"output/{method}_{graphName}_{dimension}.csv")
        SIR_label_dict = dataloader.read_label(f"data/label/{graphName}.label")
        PageRank_label_dict = dataloader.read_label(f"data/label/{graphName}_PageRank.label")
        embeddings, SIR_labels = [], []
        PageRank_labels = []
        for node, vector in embedding_dict.items():
            embeddings.append(vector)
            SIR_labels.append(SIR_label_dict[node])
            PageRank_labels.append(PageRank_label_dict[node])
        print(f"{method}, {graphName}, dimension: {dimension}")
        SIR_val = max(SIR_val, evaluate.KNN_evaluate(embeddings, SIR_labels, cv=10, n_neighbor=20))
        PageRank_val = max(PageRank_val, evaluate.KNN_evaluate(embeddings, PageRank_labels, cv=10, n_neighbor=20))

        #evaluate.LR_evaluate(embeddings, labels)
    print(f"max score, SIR:{SIR_val}, PageRank:{PageRank_val}\n")


if __name__ == '__main__':
    taus = [50]
    graphs = ["europe"]
    for name in graphs:
        print(name)
        for t in taus:
            with open(f"{name}_score.txt", mode="a+", encoding="utf-8") as fout:
                for _ in range(1):
                    print(f"graph:{name}, n_scales:{t}\n")
                    SIR_v, PageRank_v = multi_HSD_Test(name, n_scales=t, hop=4, cv=5, n_neighbor=20)
                    fout.write(f"n_scales: {t}, SIR score: {SIR_v}, PageRank score: {PageRank_v}\n")
                    fout.flush()
    evaluate_embeddings()
