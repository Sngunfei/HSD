# -*- coding:utf-8 -*-

import os
import time
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
from utils.visualize import plot_embeddings, heat_map
import numpy as np
from utils.util import dataloader, sparse_process
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

import warnings
warnings.filterwarnings("ignore")


def graphWave(graph, scale=10, d=32):
    wave_machine = GraphWave(graph, heat_coefficient=scale, sample_number=d)
    embeddings_dict = wave_machine.single_scale_embedding(scale)
    return embeddings_dict


def node2vec(graph):
    graph = nx.DiGraph(graph)
    model = Node2Vec(graph, walk_length=50, num_walks=15, p=1.0, q=2.0, workers=3)
    model.train(window_size=15, iter=300)
    embeddings_dict = model.get_embeddings()
    return embeddings_dict


def LE(graph, dim=32):
    model = LaplacianEigenmaps(graph, dim=dim)
    embeddings_dict = model.create_embedding()
    return embeddings_dict


def struc2vec(graph=None, walk_length=10, window_size=10, num_walks=15, stay_prob=0.3, dim=16):
    model = Struc2Vec(graph, walk_length=walk_length, num_walks=num_walks, stay_prob=stay_prob)
    model.train(embed_size=dim, window_size=window_size)
    embeddings_dict = model.get_embeddings()
    return embeddings_dict


def hseLLE(name="", graph=None, scale=10, method='l1', dim=16, percentile=0.0, reuse=True):
    save_path = "../../similarity/{}_{}_{}.csv".format(name, scale, method)
    if not (reuse and os.path.exists(save_path)):
        wave_machine = GraphWave(graph, heat_coefficient=scale)
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        #wave_machine.calc_wavelet_similarity(coeff_mat=coeffs, method=method, layers=5, normalized=False, save_path=save_path)
        wave_machine.parallel_calc_similarity(coeff_mat=coeffs, metric=method, layers=3, save_path=save_path)

    new_graph, _, _ = dataloader(name, directed=False, similarity=True, scale=scale, metric=method)
    new_graph = sparse_process(new_graph, percentile=percentile)
    model = LocallyLinearEmbedding(new_graph, dim)
    #embeddings_dict = model.create_embedding(dim)
    embeddings_dict = model.sklearn_lle(n_neighbors=3, dim=dim, random_state=42)
    return embeddings_dict


def hseNode2vec(idx, name="", graph=None, scale=10, metric='l1', dim=16, percentile=0.0, reuse=True):
    save_path = "../../similarity/{}_{}_{}_idx.csv".format(name, scale, metric)
    if not (reuse and os.path.exists(save_path)):
        wave_machine = GraphWave(graph, heat_coefficient=scale)
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        wave_machine.calc_wavelet_similarity(coeff_mat=coeffs, method=metric, layers=10, save_path=save_path)

    new_graph, _, _ = dataloader(name, directed=True, label='auto', similarity=True, scale=scale, metric=metric)
    new_graph = sparse_process(new_graph, percentile=percentile)
    model = Node2Vec(new_graph, walk_length=15, num_walks=10, p=1, q=1.0, workers=1)
    model.train(window_size=15, iter=300, embed_size=dim)
    embeddings_dict = model.get_embeddings()
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
    graph, label_dict, n_class = dataloader(data, directed=False, label="SIR")
    #embedding_dict = hseLE(name=data, graph=graph, scale=50, method='wasserstein', dim=64, percentile=0.7, reuse=True)
    embedding_dict = hseLLE(name=data, graph=graph, scale=50, percentile=0.5, method='wasserstein', dim=64, reuse=True)
    #embedding_dict = hseNode2vec(name=data, graph=graph, scale=10, metric='l1', dim=32, percentile=0.5, reuse=False)
    #embedding_dict = struc2vec(graph, walk_length=50, window_size=15, num_walks=15, stay_prob=0.3, dim=64)
    #embedding_dict = node2vec(graph)
    #embedding_dict = LE(graph, dim=64)
    #embedding_dict = graphWave(graph, scale=5, d=32)
    #embedding_dict = LocallyLinearEmbedding(graph=graph, dim=64).create_embedding()

    nodes = []
    labels = []
    embeddings = []
    for node, embedding in embedding_dict.items():
        nodes.append(node)
        embeddings.append(embedding)
        labels.append(label_dict[node])
    evaluate_LR_accuracy(embeddings, labels, random_state=42)
    evaluate_KNN_accuracy(embeddings, labels, random_state=42)
    plot_embeddings(nodes, embeddings, labels, method="tsne", perplexity=10)
    #heat_map(embeddings, labels)


def robustness(data, probs=None, cnt=10, scale=10, method="LR", metric='l1', dim=32, percentile=0.75):

    graph, label_dict, n_class = dataloader(data, directed=False, label="auto")
    pool = mp.Pool(5)
    results = []
    for i, p in enumerate(probs):
        print(i, p)
        _res = pool.apply_async(worker, args=(i, p, data, graph, label_dict, n_class, cnt, scale, metric, dim, percentile))
        results.append(_res)

    pool.close()
    pool.join()

    for _res in results:
        _res.get()


def worker(idx, prob, data, graph, label_dict, n_class, cnt=10, scale=10, metric='l1', dim=32, percentile=0.75):
    db = Database()
    print("idx = {}, prob = {}".format(idx, prob))
    knn_scores = []
    lr_scores = []
    times = []
    for _ in tqdm(range(cnt)):
        start = time.time()
        _graph = random_remove_edges(nx.Graph(graph), prob=prob)
        #embedding_dict = hseLE(name=data, graph=_graph, scale=scale, method=metric,
        #                       dim=dim, percentile=percentile, reuse=False)

        embedding_dict = hseNode2vec(idx, name=data, graph=_graph, scale=scale, metric=metric,
                                     dim=dim, percentile=percentile, reuse=False)

        #embedding_dict = hseLLE(name=data, graph=_graph, scale=scale, method=metric,
        #                       dim=dim, percentile=percentile, reuse=False)
        times.append(time.time() - start)

        nodes = []
        labels = []
        embeddings = []
        for node, embedding in embedding_dict.items():
            nodes.append(node)
            embeddings.append(embedding)
            labels.append(label_dict.get(node, str(n_class)))
        knn_scores.append(evaluate_KNN_accuracy(embeddings, labels, random_state=42))
        lr_scores.append(evaluate_LR_accuracy(embeddings, labels, random_state=42))

    res = {"ge_name": "hseNode2vec",
           "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
           "data": data,
           "prob": prob,
           "percentile": percentile,
           "scale": scale,
           "metric": metric,
           "dim": dim,
           "cnt": cnt,
           "evaluate": "LR",
           "scores": lr_scores,
           "mean": np.mean(lr_scores),
           "std": np.std(lr_scores),
           "mean_time": np.mean(times)
           }
    db.insert_score(res)

    res = {"ge_name": "hseNode2vec",
           "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
           "data": data,
           "prob": prob,
           "percentile": percentile,
           "scale": scale,
           "metric": metric,
           "dim": dim,
           "cnt": cnt,
           "evaluate": "KNN",
           "scores": knn_scores,
           "mean": np.mean(knn_scores),
           "std": np.std(knn_scores),
           "mean_time": np.mean(times)
           }
    db.insert_score(res)

    return True


def _time_test(dataset=None, cnt=10):
    """
    在数据集上记录运行时长
    :param dataset: dataset name, str.
    :param cnt: execute cnt times, int.
    :return: the average time on the dataset, float.
    """
    graph, _, _ = dataloader(dataset, directed=False, label="auto")
    n_nodes = nx.number_of_nodes(graph)
    n_edges = nx.number_of_edges(graph)
    times = []
    for _ in range(cnt):
        start = time.time()
        hseLLE(name=dataset, graph=graph, scale=50, method='l1', dim=16, percentile=0.4, reuse=False)
        times.append(time.time() - start)
    mean_time = sum(times) / cnt
    print("Number of nodes: {}, number of edges: {}, run {} times, time = {}s, mean time = {}s\n"
          .format(n_nodes, n_edges, cnt, times, mean_time))
    return n_nodes, n_edges, mean_time


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
    #start = time.time()
    embedd("usa")
    #print("all", time.time() - start)
    #_time_test("europe")
    #robustness("europe", probs=[i * 0.05 for i in range(10, 21)], method="KNN", cnt=25, percentile=0.5)
    #scalability_test(datasets=['bell', 'mkarate', 'subway', 'railway', 'brazil', 'europe'])
