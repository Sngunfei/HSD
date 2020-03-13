# -*- coding:utf-8 -*-

"""
HSD
"""

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from ge.utils.dataloader import load_data, load_data_from_distance
from ge.utils.visualize import plot_embeddings
from ge.utils.rw import save_vectors, save_results
from ge.utils.util import sparse_graph
from example.parser import HSDParameterParser, tab_printer
from ge.model.HSD import HSD
from ge.model.LaplacianEigenmaps import LaplacianEigenmaps
from ge.model.LocallyLinearEmbedding import LocallyLinearEmbedding
from ge.evaluate.evaluate import KNN_evaluate, LR_evaluate
from sklearn.manifold import TSNE
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def run(hsd, label_dict, n_class, params):

    if str.lower(params.multi_scales) == "no":
        # single scale
        hsd.initialize()
        distMat = hsd.calculateDistanceParallel(metric=model.metric, n_workers=params.workers, save=True)
        save_path = "../tsne_results/HSD_{}_scale{}_hop{}_{}_tsne{}.csv".format(
                    hsd.graph_name, hsd.scale, hsd.hop, hsd.metric, params.tsne)
        figure_path = "../figures/HSD_{}_scale{}_hop{}_{}_tsne{}.png".format(
                    hsd.graph_name, hsd.scale, hsd.hop, hsd.metric, params.tsne)
    elif str.lower(params.multi_scales) == "yes":
        # multi scales
        distMat = hsd.multi_scales_wavelet()
        save_path = "../tsne_results/HSD_multi_{}_hop{}_{}_tsne{}.csv".format(
            hsd.graph_name, hsd.scale, hsd.hop, hsd.metric, params.tsne)
        figure_path = "../tsne_results/HSD_multi_{}_hop{}_{}_tsne{}.png".format(
            hsd.graph_name, hsd.scale, hsd.hop, hsd.metric, params.tsne)
    else:
        raise ValueError("multi scales mode should be yes/no.")

    labels = [label_dict[node] for node in hsd.nodes]

    if hsd.graph_name in ["europe", "usa"]:
        knn_res = KNN_evaluate(distMat, metric="precomputed", labels=labels, cv=params.cv, n_neighbor=params.neighbors)
        knn_res['scale'] = hsd.scale
        knn_res['hop'] = hsd.hop
        knn_res['metric'] = hsd.metric

        save_results(knn_res, "../results/knn/HSD_{}.txt".format(graph_name))

    tsne_res = TSNE(n_components=2, metric="precomputed", learning_rate=50.0, n_iter=2000,
                    perplexity=params.tsne, random_state=params.random).fit_transform(distMat)

    df = pd.DataFrame(data=tsne_res, index=hsd.nodes, columns=None, dtype=float)
    df.to_csv(save_path, header=False, float_format="%.8f")

    plot_embeddings(hsd.nodes, tsne_res, labels=labels, n_class=n_class, save_path=figure_path)

    method = str.lower(params.embedding_method)
    if method in ['le', 'lle']:
        print("Start graph embedding.")
        new_graph, _, _, = load_data_from_distance(hsd.graph_name, label_name="SIR", metric=params.metric,
                                hop=hsd.hop, scale=hsd.scale, multi=params.multi_scales, directed=False)
        _sparsed_graph = sparse_graph(new_graph, percentile=params.sparse)
        if method == "le":
            LE = LaplacianEigenmaps(_sparsed_graph, n_neighbors=params.neighbors, dim=params.dim)
            embeddings_dict = LE.create_embedding()
        else:
            LLE = LocallyLinearEmbedding(_sparsed_graph, n_neighbors=params.neighbors, dim=params.dim)
            embeddings_dict = LLE.sklearn_lle(random_state=params.random)

        embeddings = []
        for idx, node in enumerate(hsd.nodes):
            embeddings.append(embeddings_dict[node])

        save_vectors(nodes=hsd.nodes, vectors=embeddings, path="../embeddings/{}_{}.csv".format(method, graph_name))

        if hsd.graph_name in ['europe', 'usa']:
            knn_res = LR_evaluate(embeddings, labels, cv=params.cv, test_size=params.test_size, random_state=params.random)
            knn_res['scale'] = hsd.scale
            knn_res['hop'] = hsd.hop
            knn_res['metric'] = hsd.metric
            save_results(knn_res, "../results/knn/HSD{}_{}.txt".format(str.upper(method), graph_name))

            lr_res = KNN_evaluate(embeddings, labels, cv=params.cv, n_neighbor=params.neighbors, random_state=params.random)
            lr_res['scale'] = hsd.scale
            lr_res['hop'] = hsd.hop
            lr_res['metric'] = hsd.metric
            save_results(lr_res, "../results/lr/HSD{}_{}.txt".format(str.upper(method), graph_name))

        tsne_res = TSNE(n_components=2, metric="euclidean", learning_rate=50.0, n_iter=2000,
                        perplexity=params.tsne, random_state=params.random).fit_transform(embeddings)

        df = pd.DataFrame(data=tsne_res, index=hsd.nodes, columns=None, dtype=float)
        df.to_csv("../tsne_results/HSD_{}_{}_tsne{}.csv".format(str.upper(method), graph_name, params.tsne),
                  header=False, float_format="%.8f")


if __name__ == '__main__':
    params = HSDParameterParser()
    tab_printer(params)

    graph_name = params.graph
    if graph_name in ["barbell", "mkarate"]:
        graph, label_dict, n_class = load_data(graph_name, label_name="origin")
    else: # europe, usa
        graph, label_dict, n_class = load_data(graph_name, label_name="SIR")

    assert params.metric in ["wasserstein", "hellinger"], "Distance metric not supported."
    model = HSD(graph, graph_name, scale=params.scale, hop=params.hop, metric=params.metric)

    run(model, label_dict, n_class, params)



