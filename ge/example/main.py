# -*- coding:utf-8 -*-

import os
from example import parser
import networkx as nx
from utils.visualize import plot_embeddings, plot_subway_embedding
import numpy as np
from utils.util import dataloader, evaluate_SVC_accuracy, evaluate_LR_accuracy, cluster_evaluate
from model.GraphWave import GraphWave
from model.struc2vec import Struc2Vec
from model.LocallyLinearEmbedding import LocallyLinearEmbedding
from model.LaplacianEigenmaps import LaplacianEigenmaps
from model.LINE import LINE
from model.node2vec import Node2Vec
from model.HOPE import HOPE


def graphWave(graph, scale):
    settings = parser.parameter_parser()
    wave_machine = GraphWave(graph, settings)
    embeddings_dict = wave_machine.single_scale_embedding(scale)
    return embeddings_dict


def node2vec(graph):
    graph = nx.DiGraph(graph)
    model = Node2Vec(graph, walk_length=5, num_walks=10, p=1, q=2.0, workers=1)
    model.train(window_size=5, iter=500)
    embeddings_dict = model.get_embeddings()
    return embeddings_dict


def LE(graph):
    model = LaplacianEigenmaps(graph)
    embeddings_dict = model.create_embedding(16)
    return embeddings_dict


def struc2vec(graph=None, walk_length=10, window_size=10, num_walks=15, stay_prob=0.3, dim=16):
    model = Struc2Vec(graph, walk_length=walk_length, num_walks=num_walks, stay_prob=stay_prob)
    model.train(embed_size=dim, window_size=window_size)
    embeddings_dict = model.get_embeddings()
    return embeddings_dict


def hse(name="", graph=None, scale=10, method='l1', dim=16):
    save_path = "../../similarity/{}_{}_{}.csv".format(name, scale, method)
    if not os.path.exists(save_path):
        settings = parser.parameter_parser()
        wave_machine = GraphWave(graph, settings)
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        wave_machine.calc_wavelet_similarity(coeff_mat=coeffs, method=method, save_path=save_path)

    new_graph, _, _ = dataloader(name, directed=False, similarity=True, scale=scale, metric=method)
    model = LocallyLinearEmbedding(new_graph)
    embeddings_dict = model.create_embedding(dim)
    return embeddings_dict


def embedd(data):
    graph, label_dict, n_class = dataloader(data, directed=False)
    #embedding_dict = hse(name=data, graph=graph, scale=20, method='l1', dim=32)
    embedding_dict = struc2vec(graph, walk_length=10, window_size=10, num_walks=15, stay_prob=0.3, dim=32)
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
    evaluate_SVC_accuracy(embeddings, labels, random_state=42)
    plot_embeddings(nodes, embeddings, labels, method="tsne", perplexity=5)


if __name__ == '__main__':
    embedd("europe")

