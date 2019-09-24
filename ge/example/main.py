# -*- coding:utf-8 -*-

import os
import time
import networkx as nx
from utils.visualize import plot_embeddings, heat_map
import numpy as np
from utils.util import dataloader
from utils.evaluate import evaluate_LR_accuracy, evaluate_SVC_accuracy, cluster_evaluate
from model.GraphWave import GraphWave
from model.struc2vec import Struc2Vec
from model.LocallyLinearEmbedding import LocallyLinearEmbedding
from model.LaplacianEigenmaps import LaplacianEigenmaps
from model.LINE import LINE
from model.node2vec import Node2Vec
from model.HOPE import HOPE


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
    embedding_dict = hseLE(name=data, graph=graph, scale=10, method='l1', dim=32, percentile=0.9, reuse=False)
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

    evaluate_LR_accuracy(embeddings, labels, random_state=42)
    #evaluate_SVC_accuracy(embeddings, labels, random_state=42)
    plot_embeddings(nodes, embeddings, labels, method="tsne", perplexity=10)
    #heat_map(embeddings, labels)


def robustness(data, prob=0.3, cnt=10):
    from ge.utils.robustness import random_remove_edges
    graph, label_dict, n_class = dataloader(data, directed=False, auto_label=True)
    scores = []
    start = time.time()
    for _ in range(cnt):
        _graph = random_remove_edges(nx.Graph(graph), prob=prob)
        embedding_dict = hseLE(name=data, graph=_graph, scale=10, method='l1', dim=32, percentile=0.85, reuse=False)
        nodes = []
        labels = []
        embeddings = []
        for node, embedding in embedding_dict.items():
            nodes.append(node)
            embeddings.append(embedding)
            labels.append(label_dict.get(node, str(n_class)))

        score = evaluate_LR_accuracy(embeddings, labels, random_state=42)
        scores.append(score)
    t = time.time() - start
    print(scores)
    print("mean: ", np.mean(scores), "std: ", np.std(scores))
    print("run {} times, time={}, mean={}".format(cnt, t, 1.0 * t / 10))


if __name__ == '__main__':
    #embedd("europe")
    robustness("europe", prob=0.7, cnt=10)