# -*- coding:utf-8 -*-

"""
HSD
"""

import sys
import os
import logging
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from ge.utils.dataloader import load_data, load_data_from_distance
from ge.utils.robustness import add_noise
from ge.utils.visualize import plot_embeddings
from ge.utils.rw import save_vectors, save_results, read_distance
from ge.utils.util import sparse_graph
from ge.model.GraphWave import scale_boundary
from example.parser import HSDParameterParser, tab_printer
from ge.model.HSD import HSD
from ge.model.LaplacianEigenmaps import LaplacianEigenmaps
from ge.model.LocallyLinearEmbedding import LocallyLinearEmbedding
from ge.evaluate.evaluate import KNN_evaluate, LR_evaluate, cluster_evaluate
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import networkx as nx


def run(hsd, label_dict, n_class, params):

    # 多尺度
    if str.lower(params.multi_scales) == "yes":
        hsd.initialize(multi=True)
        # 结构距离存储路径
        dist_file_path = "../distance/{}/HSD_multi_{}_hop{}.edgelist".format(
            hsd.graph_name, params.metric, params.hop)
        # tsne得到的2维向量存储路径
        tsne_vect_path = "../tsne_results/{}/HSD_multi_{}_hop{}_tsne{}.csv".format(
            hsd.graph_name, hsd.metric, hsd.hop, params.tsne)
        # tsne得到的图片存储路径
        tsne_figure_path = "../figures/{}/HSD_multi_{}_hop{}_tsne{}.png".format(
            hsd.graph_name, hsd.metric, hsd.hop, params.tsne)
    else: # 单尺度
        hsd.initialize(multi=False)
        dist_file_path = "../distance/{}/HSD_{}_scale{}_hop{}.edgelist".format(
            hsd.graph_name, params.metric, params.scale, params.hop)
        tsne_vect_path = "../tsne_results/{}/HSD_{}_scale{}_hop{}_tsne{}.csv".format(
            hsd.graph_name, hsd.metric, hsd.scale, hsd.hop, params.tsne)
        tsne_figure_path = "../figures/{}/HSD_{}_scale{}_hop{}_tsne{}.png".format(
            hsd.graph_name, hsd.metric, hsd.scale, hsd.hop, params.tsne)

    # reuse，直接读取之前已经计算好的距离
    if params.reuse == "yes" and os.path.exists(path=dist_file_path):
        dist_info = read_distance(dist_file_path, hsd.n_nodes)
        distMat = np.zeros((hsd.n_nodes, hsd.n_nodes), dtype=np.float)
        for idx, node in enumerate(hsd.nodes):
            node = int(node)
            for idx2 in range(idx + 1, hsd.n_nodes):
                node2 = int(hsd.nodes[idx2])
                distMat[idx, idx2] = distMat[idx2, idx] = dist_info[node, node2]
        logging.info("Reuse distance information.")
    else:
        # Need to compute
        if str.lower(params.multi_scales) == "no":
            e1, en = 0, hsd.wavelet.e[-1]
            for e in hsd.wavelet.e:
                if e > 0.001:
                    e1 = e
                    break
            scale_min, scale_max = scale_boundary(e1, en)
            scale = (scale_min + scale_max) / 2
            print("scale: ", scale)
            hsd.scale = scale
            distMat = hsd.parallel_calculate_distance()
        elif str.lower(params.multi_scales) == "yes":
            distMat = hsd.parallel_multi_scales_wavelet(n_scales=100, reuse=False)
        else:
            raise ValueError("Multi-scales mode should be yes/no.")

    labels = [label_dict[node] for node in hsd.nodes]

    result_args = {"multi-scales": params.multi_scales,
                   "scale": hsd.scale,
                   "hop": hsd.hop,
                   "metric": hsd.metric,
                   "dim": params.dim,
                   "sparse": params.sparse}

    if hsd.graph_name in ["varied_graph"]:
        h, c, v, s = cluster_evaluate(distMat, labels, n_class, metric="precomputed")
        res = KNN_evaluate(distMat, labels, metric="precomputed", cv=10, n_neighbor=4)
        tsne_res = TSNE(n_components=2, metric="precomputed", learning_rate=50.0, n_iter=2000,
                        perplexity=params.tsne, random_state=params.random).fit_transform(distMat)

        save_vectors(hsd.nodes, vectors=tsne_res, path=tsne_vect_path)
        plot_embeddings(hsd.nodes, tsne_res, labels=labels, n_class=n_class, save_path=tsne_figure_path)
        return h, c, v, s, res['accuracy'], res['macro f1'], res['micro f1']

    if hsd.graph_name in ["europe", "usa"]:
        knn_res = KNN_evaluate(distMat, labels, metric="precomputed", cv=params.cv,
                               n_neighbor=params.neighbors)
        knn_res.update(result_args)
        save_results(knn_res, "../results/knn/HSD_{}.txt".format(hsd.graph_name))

    tsne_res = TSNE(n_components=2, metric="precomputed", learning_rate=50.0, n_iter=2000,
                    perplexity=params.tsne, random_state=params.random).fit_transform(distMat)

    save_vectors(hsd.nodes, vectors=tsne_res, path=tsne_vect_path)
    plot_embeddings(hsd.nodes, tsne_res, labels=labels, n_class=n_class, save_path=tsne_figure_path)

    method = str.lower(params.embedding_method)
    if method in ['le', 'lle']:
        print("Start graph embedding.")
        new_graph, _, _, = load_data_from_distance(hsd.graph_name, label_name="SIR",
                                        metric=params.metric, hop=hsd.hop, scale=hsd.scale,
                                        multi=params.multi_scales, directed=False)
        _sparsed_graph = sparse_graph(new_graph, percentile=params.sparse)
        test_graph(_sparsed_graph)
        if method == "le":
            LE = LaplacianEigenmaps(_sparsed_graph, n_neighbors=params.neighbors, dim=params.dim)
            embeddings_dict = LE.create_embedding()
        else:
            LLE = LocallyLinearEmbedding(_sparsed_graph, n_neighbors=params.neighbors, dim=params.dim)
            embeddings_dict = LLE.sklearn_lle(random_state=params.random)

        embeddings = []
        labels = []
        for idx, node in enumerate(hsd.nodes):
            embeddings.append(embeddings_dict[node])
            labels.append(label_dict[node])

        save_vectors(nodes=hsd.nodes, vectors=embeddings,
                     path="../embeddings/{}_{}_{}.csv".format(method, hsd.graph_name, hsd.metric))

        if hsd.graph_name in ['europe', 'usa']:
            lr_res = LR_evaluate(embeddings, labels, cv=params.cv, test_size=params.test_size,
                                  random_state=params.random)
            lr_res.update(result_args)
            save_results(lr_res, "../results/lr/HSD{}_{}.txt".format(str.upper(method), graph_name))

            knn_res = KNN_evaluate(embeddings, labels, cv=params.cv, n_neighbor=params.neighbors,
                                  random_state=params.random)
            knn_res.update(result_args)
            save_results(knn_res, "../results/knn/HSD{}_{}.txt".format(str.upper(method), graph_name))

        tsne_res = TSNE(n_components=2, metric="euclidean", learning_rate=50.0, n_iter=2000,
                        perplexity=params.tsne, random_state=params.random).fit_transform(embeddings)

        save_vectors(hsd.nodes, tsne_res, path="../tsne_results/{}/HSD_{}_{}_tsne{}.csv".format(
                                            graph_name, str.upper(method), hsd.metric, params.tsne))
        plot_embeddings(hsd.nodes, tsne_res, labels, n_class,
                        save_path="../figures/{}/HSD_{}_{}_tsne{}.png".format(
                            hsd.graph_name, method, hsd.metric, params.tsne))


def test_graph(sparsed_graph: nx.Graph):
    """
    发现sparse处理过的图，效果会很差，专门开个函数测一下相关属性
    :return:
    """
    print("Number of edges: {}".format(nx.number_of_edges(sparsed_graph)))
    print("Number of nodes: {}".format(nx.number_of_nodes(sparsed_graph)))
    print("Number of nodes without neighbor: {}".format(nx.number_of_isolates(sparsed_graph)))
    print("Number of component: {}".format(nx.number_connected_components(sparsed_graph)))

    average_degree = sum([d for _, d in nx.degree(sparsed_graph)]) / nx.number_of_nodes(sparsed_graph)
    print("Average degree: {}".format(average_degree))


def exec(graph, labels, n_class, mode, perp=10):
    """

    :param graph:
    :param labels: dict{label_name: label_dict}
    :param mode:
    :param n_class:
    :return:
    """
    model = HSD(graph, "varied_graph", scale=1, hop=2, metric=None, n_workers=3)
    if mode == 0:
        # 单尺度
        model.initialize(multi=False)
        distMat = model.parallel_calculate_distance(metric="wasserstein")
    else:
        # 多尺度
        model.initialize(multi=True)
        distMat = model.parallel_multi_scales_wavelet(n_scales=100, metric="hellinger", reuse=False)

    tsne_res = TSNE(n_components=2, metric="precomputed", learning_rate=50.0, n_iter=2000,
                    perplexity=perp, random_state=42).fit_transform(distMat)

    res = dict()
    for name, label_dict in labels.items():
        _labels = [label_dict[node] for node in model.nodes]
        h, c, v, s = cluster_evaluate(distMat, _labels, n_class, metric="precomputed")
        _res = KNN_evaluate(distMat, _labels, metric="precomputed", cv=5, n_neighbor=4)
        res[name] = [h, c, v, s, _res['accuracy'], _res['macro f1'], _res['micro f1']]
        plot_embeddings(model.nodes, tsne_res, labels=_labels, n_class=n_class, save_path=
        f"E:\workspace\py\graph-embedding\\figures\HSD-{mode}-{name}-{perp}.png")

    return res


if __name__ == '__main__':
    params = HSDParameterParser()
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
            model = HSD(graph, graph_name, scale=params.scale, hop=params.hop,
                        metric=params.metric, n_workers=params.workers)
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
        graph, label_dict, n_class = load_data(graph_name)

    model = HSD(graph, graph_name, scale=params.scale, hop=params.hop,
                metric=params.metric, n_workers=params.workers)
    #print(model.nodes)
    """
    res = model.constructNodeLayers_BFS()
    node1, node2 = '3', '20'
    print(len(res[node1][1]), len(res[node1][2]), len(res[node1][3]), len(res[node1].get(4, [])))
    print(len(res[node2][1]), len(res[node2][2]), len(res[node2][3]), len(res[node2].get(4, [])))
    """
    run(model, label_dict, n_class, params)


