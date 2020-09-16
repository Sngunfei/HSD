# -*- coding:utf-8 -*-

"""
HSD
"""

import logging
import datetime
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from tools import load_data, load_data_from_distance, util
from tools.visualize import plot_embeddings
from tools import rw
from example.params_parser import  HSDParameterParser, tab_printer
from model import HSD
from tools.evaluate import KNN_evaluate, LR_evaluate, cluster_evaluate
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import time
import warnings
warnings.filterwarnings("ignore")


def run(model, labels: dict, n_class: int):
    if params.graph == "bio_dmela":
        scale = util.recommend_scale(hsd.wavelet.e)
    else:
        scale = 1
    params.scale = scale
    model.scale = scale
    #hsd.initialize(multi=True)
    # 结构距离存储路径
    distance_path = "../distance/{}/multi_{}_hop{}.edgelist".format(hsd.graph_name, params.metric, params.hop)
    # tsne得到的2维向量存储路径
    tsne_vector_path = "../tsne_vectors/{}/multi_{}_hop{}_tsne{}.csv".format(hsd.graph_name, hsd.metric, hsd.hop, params.tsne)
    # tsne得到的图片存储路径
    tsne_figure_path = "../tsne_figures/{}/multi_{}_hop{}_tsne{}.png".format(hsd.graph_name, hsd.metric, hsd.hop, params.tsne)

    # 直接读取之前已经计算好的距离
    if params.reuse == "yes" and os.path.exists(path=distance_path):
        dist_info = rw.read_distance(distance_path, model.n_nodes)
        dist_mat = np.zeros((model.n_nodes, model.n_nodes), dtype=np.float)
        for idx, node in enumerate(hsd.nodes):
            node = int(node)
            for idx2 in range(idx + 1, hsd.n_nodes):
                node2 = int(hsd.nodes[idx2])
                dist_mat[idx, idx2] = dist_mat[idx2, idx] = dist_info[node, node2]
        logging.info("Reuse distance information.")
    else:
        if str.lower(params.multi_scales) == "no":
            scale = util.recommend_scale(hsd.wavelet.e)
            print("scale: ", scale)
            hsd.scale = scale
            # dist_mat = hsd.parallel_calculate_distance()
            dist_mat = hsd.calculate_structural_distance()
        elif str.lower(params.multi_scales) == "yes":
            #hsd.calculate_multi_scales_coeff_sum(n_scales=200)
            dist_mat = hsd.single_multi_scales_wavelet(n_scales=200, reuse=True)
            #dist_mat = hsd.parallel_multi_scales_wavelet(n_scales=200, reuse=False)
        else:
            raise ValueError("Multi-scales mode should be yes/no.")

    # 过滤，只保留重要的边，缩小重建图的规模
    util.filter_distance_matrix(dist_mat, nodes=hsd.nodes, save_path=dist_file_path, ratio=0.2)

    result_args = {
        "date": datetime.datetime.now() - datetime.timedelta(hours=8),
        "multi-scales": params.multi_scales,
        "scale": hsd.scale,
        "hop": hsd.hop,
        "metric": hsd.metric,
        "dim": params.dim,
        "sparse": params.sparse
    }

    labels = [label_dict[node] for node in hsd.nodes]

    if hsd.graph_name not in ["mkarate", "barbell"]:
        knn_res = KNN_evaluate(dist_mat, labels, metric="precomputed", cv=params.cv, n_neighbor=params.neighbors)
        knn_res.update(result_args)
        rw.save_results(knn_res, "../output/knn/HSD_{}.txt".format(hsd.graph_name))

    tsne_res = TSNE(n_components=2, metric="precomputed", learning_rate=5.0, n_iter=2000,
                    perplexity=params.tsne, random_state=params.random).fit_transform(dist_mat)

    rw.save_vectors(hsd.nodes, vectors=tsne_res, path=tsne_vect_path)
    plot_embeddings(hsd.nodes, tsne_res, labels=labels, n_class=n_class, save_path=tsne_figure_path)


if __name__ == '__main__':
    start = time.time()
    params = HSDParameterParser()
    params.graph = "bio_dmela"
    tab_printer(params)
    graph_name = params.graph
    graph, label_dict, n_class = load_data(graph_name, label_name=None)
    model = HSD(graph, graph_name, scale=params.scale, hop=params.hop, metric=params.metric, n_workers=params.workers)
    run(model, label_dict, n_class)
    print("time: ", time.time() - start)
