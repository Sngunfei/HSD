# -*- encoding:utf-8 -*-

import os
import time
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
from utils.visualize import plot_embeddings, heat_map
import numpy as np
from utils.util import dataloader, sparse_process
from utils.evaluate import evaluate_LR_accuracy, evaluate_SVC_accuracy, evaluate_KNN_accuracy, cluster_evaluate
from model.GraphWave import GraphWave
from model.struc2vec import Struc2Vec
from model.LocallyLinearEmbedding import LocallyLinearEmbedding
from model.LaplacianEigenmaps import LaplacianEigenmaps
from model.node2vec import Node2Vec

from ge.utils.robustness import random_remove_edges

import warnings
warnings.filterwarnings("ignore")

def hseLLE(graph=None, name="mkarate.edgelist", method='wasserstein', percentile=None, dim=16):
    path = "G:\pyworkspace\graph-embedding\similarity\\norm_{}_{}.csv".format(name, method)
    if not os.path.exists(path):
        wave_machine = GraphWave(graph)
        wave_machine.coefficient_elapse_by_scale(dataname=name)
    graph = nx.read_edgelist(path=path, create_using=nx.Graph, edgetype=float, data=[('weight', float)])
    graph = sparse_process(graph, percentile=percentile)
    model = LocallyLinearEmbedding(graph, dim)
    #embeddings_dict = model.create_embedding(dim)
    embeddings_dict = model.sklearn_lle(n_neighbors=10, dim=dim, random_state=42)
    return embeddings_dict

def hseLE(name="", graph=None, scale=10, method='l1', dim=16, threshold=None, percentile=None, reuse=True):
    path = "G:\pyworkspace\graph-embedding\similarity\{}_{}.csv".format(name, method)
    if not os.path.exists(path):
        wave_machine = GraphWave(graph)
        wave_machine.coefficient_elapse_by_scale(dataname=name)

    graph = nx.read_edgelist(path=path, create_using=nx.Graph, edgetype=float, data=[('weight', float)])
    graph = sparse_process(graph, percentile=percentile)
    model = LaplacianEigenmaps(graph, dim=dim)
    embeddings_dict = model.create_embedding(threshold=threshold, percentile=percentile)
    #embeddings_dict = model.spectralEmbedding()
    return embeddings_dict


def embedd(data):
    graph, label_dict, n_class = dataloader(data, directed=False, similarity=False, label="SIR")

    embedding_dict = hseLLE(graph=graph, name=data, method='gauss', percentile=0.7, dim=64)
    #embedding_dict = hseLE(name=data, graph=graph, scale=50, method='wasserstein', dim=32, percentile=0.8, reuse=True)

    nodes = []
    labels = []
    embeddings = []
    for node, embedding in embedding_dict.items():
        nodes.append(node)
        embeddings.append(embedding)
        labels.append(label_dict[node])

    #cluster_evaluate(embeddings, labels, class_num=n_class)
    evaluate_LR_accuracy(embeddings, labels, random_state=42)
    print("--------------------------------------------------------------------")
    evaluate_KNN_accuracy(embeddings, labels, random_state=42)
    plot_embeddings(nodes, embeddings, labels, n_class, method="tsne", perplexity=5)
    #heat_map(embeddings, labels)


if __name__ == '__main__':
    #embedd("europe")
    pass