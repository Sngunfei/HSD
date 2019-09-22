# -*- coding:utf-8 -*-

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


def graphWave(data, scale):
    settings = parser.parameter_parser()

    graph, label_dict, n_class = dataloader(data, directed=False)
    wave_machine = GraphWave(graph, settings)
    embeddings_dict = wave_machine.single_scale_embedding(scale)
    embeddings = []
    nodes = []
    labels = []
    for node, embedd in embeddings_dict.items():
        embeddings.append(embedd)
        nodes.append(node)
        labels.append(label_dict[node])

    evaluate_LR_accuracy(embeddings, labels, random_state=42)
    evaluate_SVC_accuracy(embeddings, labels, random_state=42)
    cluster_evaluate(embeddings, labels, class_num=n_class)


def node2vec(data):
    graph, label_dict, n_class = dataloader(data, directed=True)

    model = Node2Vec(graph, walk_length=19, num_walks=15, p=1, q=2.0, workers=1)
    model.train(window_size=10, iter=1000)
    embeddings_dict = model.get_embeddings()
    nodes = []
    embeddings = []
    labels = []
    for node, embedding in embeddings_dict.items():
        nodes.append(node)
        embeddings.append(embedding)
        labels.append(label_dict[node])

    evaluate_LR_accuracy(embeddings, labels, random_state=42)
    evaluate_SVC_accuracy(embeddings, labels, random_state=42)



if __name__ == '__main__':
    import sys
    print(sys.getwindowsversion())

