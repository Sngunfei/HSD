# -*- encoding: utf-8 -*-

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from example.parser import GraphWaveParameterParser, tab_printer
from ge.model.GraphWave import GraphWave, scale_boundary
from ge.utils.dataloader import load_data
from ge.utils.rw import save_results
import pandas as pd
from sklearn.manifold import TSNE
from ge.utils.visualize import plot_embeddings
from ge.evaluate.evaluate import LR_evaluate, KNN_evaluate, cluster_evaluate
from ge.utils.robustness import add_noise

import networkx as nx
import numpy as np


def run(model, label_dict, n_class, params=None):
    e1, en = 0, model.e[-1]
    for e in model.e:
        if e > 0.001:
            e1 = e
            break
    scale_min, scale_max = scale_boundary(e1, en)
    scale = (scale_min + scale_max) / 2
    print(scale)
    embedding_dict = model.single_scale_embedding(scale)

    nodes, labels, embeddings = [], [], []
    for node, embedding_vector in embedding_dict.items():
        nodes.append(node)
        labels.append(label_dict[node])
        embeddings.append(embedding_vector)

    if model.graph_name in ["varied_graph"]:
        h, c, v, s = cluster_evaluate(embeddings, labels, n_class)
        res = KNN_evaluate(embeddings, labels, cv=5, n_neighbor=4)
        return h, c, v, s, res['accuracy'], res['macro f1'], res['micro f1']

    _res = {'sacle': model.scale,
            'samples': model.sample_number,
            'step_size':model.step_size}

    if model.graph_name in ['europe', 'usa']:
        lr_res = LR_evaluate(embeddings, labels)
        lr_res.update(_res)
        save_results(lr_res, "../results/lr/graphwave_{}.txt".format(graph_name))

        knn_res = KNN_evaluate(embeddings, labels)
        knn_res.update(_res)
        save_results(knn_res, "../results/knn/graphwave_{}.txt".format(graph_name))

    df = pd.DataFrame(data=embeddings, index=nodes, columns=None, dtype=float)
    # file_name: graphwave_mkarate_scale.csv
    df.to_csv("../embeddings/graphwave_{}_scale{}.csv".format(
        model.graph_name, model.scale), header=False, float_format="%.8f")

    tsne_res = TSNE(n_components=2, metric="euclidean", learning_rate=50.0, n_iter=2000,
               perplexity=params.tsne, random_state=params.random).fit_transform(embeddings)

    df = pd.DataFrame(data=tsne_res, index=nodes, columns=None, dtype=float)
    df.to_csv("../tsne_results/graphwave_{}_scale{}_perp{}.csv".format(
        graphwave.graph_name, graphwave.scale, params.tsne))

    figure_path = "../figures/graphwave_{}_scale{}_tsne{}.png".format(
        graphwave.graph_name, graphwave.scale, params.tsne)
    plot_embeddings(nodes, tsne_res, labels=labels, n_class=n_class, node_text=False, save_path=figure_path)


def exec(graph, labels, n_class, perp=10):
    model = GraphWave(graph, "varied_graph", scale=0.2, sample_number=32, step_size=5)

    e1, en = 0, model.e[-1]
    for e in model.e:
        if e > 0.001:
            e1 = e
            break
    scale_min, scale_max = scale_boundary(e1, en)
    scale = (scale_min + scale_max) / 2
    print(scale)
    embedding_dict = model.single_scale_embedding(scale)

    nodes, embeddings = [], []
    for node, embedding_vector in embedding_dict.items():
        nodes.append(node)
        embeddings.append(embedding_vector)

    tsne_res = TSNE(n_components=2, metric="euclidean", learning_rate=50.0, n_iter=2000,
                    perplexity=perp, random_state=42).fit_transform(embeddings)

    res = dict()
    for name, label_dict in labels.items():
        _labels = [label_dict[node] for node in nodes]
        h, c, v, s = cluster_evaluate(embeddings, _labels, n_class)
        _res = KNN_evaluate(embeddings, _labels, cv=5, n_neighbor=4)
        res[name] = [h, c, v, s, _res['accuracy'], _res['macro f1'], _res['micro f1']]
        plot_embeddings(nodes, tsne_res, labels=_labels, n_class=n_class, save_path=
        f"E:\workspace\py\graph-embedding\\figures\graphwave-{name}-{perp}.png")
    return res


if __name__ == '__main__':
    params = GraphWaveParameterParser()
    tab_printer(params)

    graph_name = params.graph
    if graph_name in ["europe", "usa"]:
        graph, label_dict, n_class = load_data(graph_name, label_name="SIR")
    elif graph_name in ["varied_graph"]:
        from collections import defaultdict
        from tqdm import tqdm

        res = defaultdict(list)
        for _ in tqdm(range(25)):
            graph, label_dict, n_class = load_data(graph_name)
            print(nx.number_of_edges(graph))
            graph = add_noise(graph, ratio=0.1)
            print(nx.number_of_edges(graph))
            model = GraphWave(graph, graph_name, params.scale, params.sample_number, params.step_size)
            h, c, v, s, acc, macro, micro = run(model, label_dict, n_class, params)
            res["h"].append(h)
            res["c"].append(c)
            res["v"].append(v)
            res["s"].append(s)
            res["acc"].append(acc)
            res["macro"].append(macro)
            res["micro"].append(micro)
        for k, v in res.items():
            print(k, v, np.mean(v), "\n")
        assert False
    else:
        graph, label_dict, n_class = load_data(graph_name, label_name=None)

    #def __init__(self, graph, heat_coefficient=5.0, sample_number=16, step_size=20.0):
    model = GraphWave(graph, graph_name, params.scale, params.sample_number, params.step_size)
    run(model, label_dict, n_class, params)

