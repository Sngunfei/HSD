# -*- encoding: utf-8 -*-

from model.HSD import HSD
from model.multiscale_HSD import MultiHSD
from model.dynamic_HSD import DynamicHSD
from tools import dataloader, evaluate, rw
import networkx as nx


def base_HSD_Test():
    pass


def multi_HSD_Test(graphName, hop=3, n_scales=200):
    graph = nx.read_edgelist(f"data/graph/{graphName}.edgelist", create_using=nx.Graph,edgetype=float, data=[('weight', float)])
    label_dict = dataloader.read_label(f"data/label/{graphName}.label")

    model = MultiHSD(graph, graphName, hop, n_scales)
    model.init()
    embedding_dict = model.parallel_embed(n_workers=10)

    rw.save_vectors_dict(embedding_dict, f"output/multi_HSD_{graphName}_{(hop+1) * n_scales}.csv")

    embeddings, labels = [], []
    for node, vector in embedding_dict.items():
        embeddings.append(vector)
        labels.append(label_dict[node])

    knn_score = evaluate.KNN_evaluate(embeddings, labels)
    lr_score = evaluate.LR_evaluate(embeddings, labels)

    return knn_score, lr_score


def dynamic_HSD_Test():
    pass


if __name__ == '__main__':
    taus = [i * 10 for i in range(2, 20)]
    for t in taus:
        with open("score.txt", mode="a+", encoding="utf-8") as fout:
            c1, c2 = multi_HSD_Test("bio_dmela_new", n_scales=t)
            fout.write(f"n_scales: {t}, knn score: {c1}, lr_score: {c2}\n")
