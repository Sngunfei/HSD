# -*- coding:utf-8 -*-

import os
import time
import networkx as nx
from tqdm import tqdm
from utils.visualize import plot_embeddings, heat_map
import numpy as np
from utils.util import dataloader
from utils.evaluate import evaluate_LR_accuracy, evaluate_SVC_accuracy, evaluate_KNN_accuracy
from model.GraphWave import GraphWave
from model.struc2vec import Struc2Vec
from model.LocallyLinearEmbedding import LocallyLinearEmbedding
from model.LaplacianEigenmaps import LaplacianEigenmaps
from model.LINE import LINE
from model.node2vec import Node2Vec
from model.HOPE import HOPE

from ge.utils.robustness import random_remove_edges
from ge.utils.db import Database


def graphWave(graph, scale):
    wave_machine = GraphWave(graph, heat_coefficient=scale)
    embeddings_dict = wave_machine.single_scale_embedding(scale)
    return embeddings_dict


def node2vec(graph):
    graph = nx.DiGraph(graph)
    model = Node2Vec(graph, walk_length=10, num_walks=10, p=1, q=2.0, workers=1)
    model.train(window_size=10, iter=500)
    embeddings_dict = model.get_embeddings()
    return embeddings_dict


def LE(graph):
    model = LaplacianEigenmaps(graph)
    embeddings_dict = model.create_embedding()
    return embeddings_dict


def struc2vec(graph=None, walk_length=10, window_size=10, num_walks=15, stay_prob=0.3, dim=16):
    model = Struc2Vec(graph, walk_length=walk_length, num_walks=num_walks, stay_prob=stay_prob)
    model.train(embed_size=dim, window_size=window_size)
    embeddings_dict = model.get_embeddings()
    return embeddings_dict


def hseLLE(name="", graph=None, scale=10, method='l1', dim=16, reuse=True):
    save_path = "../../similarity/{}_{}_{}.csv".format(name, scale, method)
    if not (reuse and os.path.exists(save_path)):
        wave_machine = GraphWave(graph, heat_coefficient=scale)
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        wave_machine.calc_wavelet_similarity(coeff_mat=coeffs, method=method, layers=10, save_path=save_path)

    new_graph, _, _ = dataloader(name, directed=False, similarity=True, scale=scale, metric=method)
    model = LocallyLinearEmbedding(new_graph)
    embeddings_dict = model.create_embedding(dim)
    return embeddings_dict


def hseLE(name="", graph=None, scale=10, method='l1', dim=16, threshold=None, percentile=None, reuse=True):
    save_path = "../../similarity/{}_{}_{}.csv".format(name, scale, method)
    if not (reuse and os.path.exists(save_path)):
        wave_machine = GraphWave(graph, heat_coefficient=scale)
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        wave_machine.calc_wavelet_similarity(coeff_mat=coeffs, method=method, layers=10, save_path=save_path)

    new_graph, _, _ = dataloader(name, directed=False, similarity=True, scale=scale, metric=method)
    model = LaplacianEigenmaps(new_graph, dim=dim)
    embeddings_dict = model.create_embedding(threshold=threshold, percentile=percentile)
    #embeddings_dict = model.spectralEmbedding()
    return embeddings_dict


def embedd(data):
    graph, label_dict, n_class = dataloader(data, directed=False, auto_label=True)
    embedding_dict = hseLE(name=data, graph=graph, scale=10, method='l1', dim=32, percentile=0.5, reuse=True)
    #embedding_dict = hseLLE(name=data, graph=graph, scale=10, method='l1', dim=32, reuse=True)

    #embedding_dict = struc2vec(graph, walk_length=10, window_size=10, num_walks=15, stay_prob=0.3, dim=32)
    #embedding_dict = node2vec(graph)
    #embedding_dict = LE(graph)
    nodes = []
    labels = []
    embeddings = []
    for node, embedding in embedding_dict.items():
        nodes.append(node)
        embeddings.append(embedding)
        labels.append(label_dict.get(node, str(n_class)))
    #evaluate_LR_accuracy(embeddings, labels, random_state=42)
    evaluate_KNN_accuracy(embeddings, labels, random_state=42)
    #evaluate_SVC_accuracy(embeddings, labels, random_state=42)
    #plot_embeddings(nodes, embeddings, labels, method="tsne", perplexity=10)
    #heat_map(embeddings, labels)


def robustness(data, db=None, probs=None, cnt=10, scale=10, method="LR", metric='l1', dim=32, percentile=0.75):
    if not db:
        db = Database()
    classifiers = {"LR": evaluate_LR_accuracy, "SVC": evaluate_SVC_accuracy, "KNN": evaluate_KNN_accuracy}
    clf = classifiers[method]
    graph, label_dict, n_class = dataloader(data, directed=False, auto_label=True)
    for i, prob in enumerate(probs):
        start = time.time()
        scores = []
        for _ in tqdm(range(cnt)):
            _graph = random_remove_edges(nx.Graph(graph), prob=prob)
            embedding_dict = hseLE(name=data, graph=_graph, scale=scale, method=metric,
                                   dim=dim, percentile=percentile, reuse=False)
            nodes = []
            labels = []
            embeddings = []
            for node, embedding in embedding_dict.items():
                nodes.append(node)
                embeddings.append(embedding)
                labels.append(label_dict.get(node, str(n_class)))
            score = clf(embeddings, labels, random_state=42)
            scores.append(score)
        t = time.time() - start
        res = {"ge_name": "HSELE",
               "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
               "data": data,
               "prob": prob,
               "percentile": percentile,
               "scale": scale,
               "metric": metric,
               "dim": dim,
               "cnt": cnt,
               "evaluate": method,
               "scores": scores,
               "mean": np.mean(scores),
               "std": np.std(scores),
               "mean_time": 1.0 * t / cnt
               }
        db.insert_score(res)
        """
        print(data, "prob={}, cnt={}".format(prob, cnt))
        print(scores)
        print("mean: ", np.mean(scores), "std: ", np.std(scores))
        print("run {} times, time={}, mean={}".format(cnt, t, 1.0 * t / cnt))
        """


def _time_test(dataset=None, cnt=10):
    """
    在数据集上记录运行时长
    :param dataset: dataset name, str.
    :param cnt: execute cnt times, int.
    :return: the average time on the dataset, float.
    """
    graph, _, _ = dataloader(dataset, directed=False, auto_label=True)
    n_nodes = nx.number_of_nodes(graph)
    n_edges = nx.number_of_edges(graph)
    start = time.time()
    for _ in range(cnt):
        hseLE(name=dataset, graph=graph, scale=50, method='l1', dim=16, percentile=0.4, reuse=False)
    end = time.time()
    _time = end - start
    _mean_time = 1.0 * _time / cnt
    print("Number of nodes: {}, number of edges: {}, run {} times, time = {}s, mean time = {}s\n"
          .format(n_nodes, n_edges, cnt, _time, _mean_time))
    return n_nodes, n_edges, _mean_time


def scalability_test(datasets=None, cnt=10):
    """
    在不同规模的数据集上记录运行时长
    :param datasets: a list of dataset. [names]
    :param cnt: excute cnt times, int
    :return:
    """
    with open("../../output/time_report.txt", mode="w+", encoding="utf-8") as fout:
        for _, data in enumerate(datasets):
            n_nodes, n_edges, mean_time = _time_test(data, cnt)
            fout.write("dataset: {}, run {} times.\n".format(data, cnt))
            fout.write("Number of nodes: {}, number of edges: {}, mean time = {}s\n\n".format(n_nodes, n_edges, mean_time))


if __name__ == '__main__':
    #embedd("europe")
    robustness("europe", db=Database(), probs=[i * 0.05 for i in range(10, 12)], method="KNN", cnt=25, percentile=0.5)
    #scalability_test(datasets=['bell', 'mkarate', 'subway', 'railway', 'brazil', 'europe'])
