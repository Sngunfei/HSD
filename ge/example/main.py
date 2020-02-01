# -*- coding:utf-8 -*-

import os
import time
import networkx as nx
from tqdm import tqdm
import multiprocessing as mp
from utils.visualize import plot_embeddings
import numpy as np
from utils.util import dataloader, sparse_process, save_vectors, read_vectors, read_distance
from utils.evaluate import evaluate_LR_accuracy, evaluate_KNN_accuracy, cluster_evaluate
from model.GraphWave import GraphWave, scale_boundary
from model.struc2vec import Struc2Vec
from model.LocallyLinearEmbedding import LocallyLinearEmbedding
from model.LaplacianEigenmaps import LaplacianEigenmaps
from model.node2vec import Node2Vec

from ge.utils.robustness import random_remove_edges
from ge.utils.db import Database
from ge.utils.tsne import tsne, cal_pairwise_dist

import warnings
warnings.filterwarnings("ignore")


def graphWave(name, graph, reused=False, scale=10.0, dim=32):
    path = "../../output/graphwave_{}_{}.csv".format(name, scale)
    if reused and os.path.exists(path):
        embeddings_dict = read_vectors(path)
    else:
        wave_machine = GraphWave(graph, heat_coefficient=scale, sample_number=dim)
        embeddings_dict = wave_machine.single_scale_embedding(scale)
        save_vectors(embeddings_dict, path)
    return embeddings_dict


def node2vec(name, graph, reused=False, walk_length=50, window_size=20, num_walks=30, p=1.0, q=2.0, dim=64):
    path = "../../output/node2vec_{}.csv".format(name)
    if reused and os.path.exists(path):
        embeddings_dict = read_vectors(path)
    else:
        graph = nx.DiGraph(graph)
        model = Node2Vec(graph, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=3)
        model.train(embed_size=dim, window_size=window_size, iter=500)
        embeddings_dict = model.get_embeddings()
        save_vectors(embeddings_dict, path)
    return embeddings_dict


def LE(graph, dim=32):
    model = LaplacianEigenmaps(graph, dim=dim)
    embeddings_dict = model.create_embedding()
    return embeddings_dict


def struc2vec(name, graph, walk_length=10, window_size=10,
              num_walks=15, stay_prob=0.3, dim=16, reused=False):

    path = "../../output/struc2vec_{}_length15.csv".format(name)
    if reused and os.path.exists(path):
        embeddings_dict = read_vectors(path)
    else:
        model = Struc2Vec(graph, walk_length=walk_length, num_walks=num_walks, stay_prob=stay_prob)
        model.train(embed_size=dim, window_size=window_size)
        embeddings_dict = model.get_embeddings()
        save_vectors(embeddings_dict, path=path)

    return embeddings_dict


def hseLLE(name, graph, scale=10.0, method='l1', dim=16, percentile=0.0, reuse=True):
    save_path = "../../similarity/{}_{}_{}.csv".format(name, scale, method)
    if not (reuse and os.path.exists(save_path)):
        wave_machine = GraphWave(graph, heat_coefficient=scale)
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        #wave_machine.calc_wavelet_similarity(coeff_mat=coeffs, method=method, layers=5, normalized=False, save_path=save_path)
        #wave_machine.parallel_calc_similarity(coeff_mat=coeffs, metric=method, layers=10, mode="similarity", save_path=save_path)
        wave_machine.calc_wavelet_similarity(coeff_mat=coeffs, hierachical=True, method=method, normalized=False, layers=5, save_path=save_path)


    new_graph, _, _ = dataloader(name, directed=False, similarity=True, scale=scale, metric=method)
    new_graph = sparse_process(new_graph, percentile=percentile)
    model = LocallyLinearEmbedding(new_graph, dim)
    #embeddings_dict = model.create_embedding(dim)
    embeddings_dict = model.sklearn_lle(n_neighbors=10, dim=dim, random_state=42)
    return embeddings_dict


def hseNode2vec(idx, name, graph, scale=10, metric='l1', dim=16, percentile=0.0, reuse=True):
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


def hseLE(name, graph, scale=10.0, method='l1', dim=16, threshold=None, percentile=None, reuse=True):
    save_path = "../../similarity/{}_{}_{}.csv".format(name, scale, method)
    if not (reuse and os.path.exists(save_path)):
        wave_machine = GraphWave(graph, heat_coefficient=scale)
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        wave_machine.calc_wavelet_similarity(coeff_mat=coeffs, hierachical=True, normalized=False, method=method, layers=5, save_path=save_path)

    new_graph, _, _ = dataloader(name, directed=False, similarity=True, scale=scale, metric=method)
    model = LaplacianEigenmaps(new_graph, dim=dim)
    embeddings_dict = model.create_embedding(threshold=threshold, percentile=percentile)
    #embeddings_dict = model.spectralEmbedding()
    return embeddings_dict


def rolx(data_name):
    embeddings_dict = read_vectors("../../output/rolx_{}.csv".format(data_name))
    return embeddings_dict


def embedd(data_name, label_class="SIR"):
    graph, label_dict, n_class = dataloader(data_name, directed=False, label=label_class)
    wave_machine = GraphWave(graph)
    eigenvalues = wave_machine._e
    sMin, sMax = scale_boundary(eigenvalues[1], eigenvalues[-1])
    scale = (sMin + sMax) / 2  # 根据GraphWave论文中推荐的尺度进行设置。
    #scale = 2
    #print(scale)

    #embedding_dict = hseLE(name=data_name, graph=graph, scale=scale, method='wasserstein', dim=64, percentile=0.7, reuse=True)
    #embedding_dict = hseLLE(name=data_name, graph=graph, scale=0.1, percentile=0.7, method='wasserstein', dim=64, reuse=True)
    #embedding_dict = hseNode2vec(name=data, graph=graph, scale=10, metric='l1', dim=32, percentile=0.5, reuse=False)
    #embedding_dict, method = struc2vec(data_name, graph=graph, walk_length=15, window_size=10, num_walks=10, stay_prob=0.5, dim=64, reused=False), "struc2vec"
    #embedding_dict, method = node2vec(data_name, graph, walk_length=50, num_walks=10, window_size=15, p=1, q=2, dim=64, reused=False), "node2vec"
    #embedding_dict = LE(graph, dim=64)
    #embedding_dict, method = graphWave(data_name, graph, scale=scale, dim=64, reused=True), "graphwave"
    #embedding_dict = LocallyLinearEmbedding(graph=graph, dim=64).create_embedding()
    #embedding_dict = rolx(data_name)
    embedding_dict = read_vectors("C:\\Users\86234\Desktop\论文相关\\tsne\\node2vec_usa.csv")
    nodes = []
    labels = []
    embeddings = []
    for node, embedding in embedding_dict.items():
        nodes.append(node)
        embeddings.append(embedding)
        labels.append(label_dict[node])

    #dist = cal_pairwise_dist(np.array(embeddings))
    #embeddings = tsne(distance_mat=dist, dim=2, perplexity=30)

    #cluster_evaluate(embeddings, labels, class_num=n_class)
    #evaluate_LR_accuracy(embeddings, labels, random_state=42)
    #evaluate_KNN_accuracy(embeddings, labels, "euclidean", random_state=42,  n_neighbor=20)

    _2d_data = plot_embeddings(nodes, embeddings, labels, n_class, method="tsne", init="random",
                               perplexity=30, node_text=False, random_state=42)
    tmp = {}
    for idx, node in enumerate(nodes):
        tmp[node] = _2d_data[idx]
    #save_vectors(tmp, "../../output/{}_{}_tsne3.csv".format(method, data_name))
    #heat_map(embeddings, labels)


def robustness(data, probs=None, cnt=10, scale=10.0, label="SIR", metric='l1', dim=32, percentile=0.75):
    """
    测试鲁棒性
    :param data:
    :param probs:
    :param cnt:
    :param scale:
    :param label:
    :param metric:
    :param dim:
    :param percentile:
    :return:
    """
    graph, label_dict, n_class = dataloader(data, directed=False, label=label)
    pool = mp.Pool(5)
    results = []
    for i, p in enumerate(probs):
        _res = pool.apply_async(worker, args=(i, p, data, graph, label_dict, n_class, cnt, scale, metric, dim, percentile))
        results.append(_res)

    pool.close()
    pool.join()
    for _res in results:
        _res.get()


def worker(idx, prob, data, graph, label_dict, n_class, cnt=10, scale=2, metric='l1', dim=64, percentile=0.75):
    db = Database()
    print("idx = {}, prob = {}".format(idx, prob))
    for _ in tqdm(range(cnt)):
        start = time.time()

        if prob == 1.0:
            _graph = graph
        else:
            _graph = random_remove_edges(nx.Graph(graph), prob=prob)

        #embedding_dict = hseLE(name=data, graph=_graph, scale=scale, method=metric, dim=dim, percentile=percentile, reuse=False)
        #method = "HSELE"

        #embedding_dict = hseNode2vec(idx, name=data, graph=_graph, scale=scale, metric=metric,
        #                             dim=dim, percentile=percentile, reuse=False)

        embedding_dict = hseLLE(name=data, graph=_graph, scale=scale, method=metric, dim=dim, percentile=percentile, reuse=False)
        method = "HSELLE"

        #embedding_dict = graphWave(name=data, graph=_graph, scale=scale, dim=64, reused=False)
        #method = "graphwave"

        #embedding_dict = struc2vec(name=data, graph=_graph, walk_length=60, window_size=25, num_walks=10, stay_prob=0.3, dim=64, reused=False)
        #method = "struc2vec"

        #embedding_dict = node2vec(name=data, graph=_graph, walk_length=60, num_walks=10, window_size=25, p=1, q=2, dim=64, reused=False)
        #method = "node2vec"

        _time = time.time() - start

        nodes = []
        labels = []
        embeddings = []
        for node, embedding in embedding_dict.items():
            nodes.append(node)
            embeddings.append(embedding)
            labels.append(label_dict.get(node, str(n_class)))

        accuracy, balanced_accuracy, precision, recall, macro_f1, micro_f1 = evaluate_LR_accuracy(embeddings,
                                                                                                  labels,
                                                                                                  random_state=42)
        res_lr = {"method": method,
               "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
               "graph": data,
               "prob": prob,
               "percentile": percentile,
               "scale": scale,
               "metric": metric,
               "dim": dim,
               "evaluate model": "LR",
               "accuracy": accuracy,
               "balanced_accuracy": balanced_accuracy,
               "precision": precision,
               "recall": recall,
               "macro f1": macro_f1,
               "micro f1": micro_f1,
               "time": _time,
               "label": "SIR_2"
               }

        accuracy, balanced_accuracy, precision, recall, macro_f1, micro_f1 = evaluate_KNN_accuracy(embeddings, labels,
                                                                                                   n_neighbor=20,
                                                                                                   random_state=42,
                                                                                                   metric="euclidean")
        res_knn = {"method": method,
               "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
               "graph": data,
               "prob": prob,
               "percentile": percentile,
               "scale": scale,
               "metric": metric,
               "dim": dim,
               "cnt": cnt,
               "evaluate model": "KNN",
               "accuracy": accuracy,
               "balanced_accuracy": balanced_accuracy,
               "precision": precision,
               "recall": recall,
               "macro f1": macro_f1,
               "micro f1": micro_f1,
               "time": _time,
               "label": "SIR_2"
               }

        db.insert(res_lr, "graph embedding")
        db.insert(res_knn, "graph embedding")
        #db.insert(res_lr, "robust")
        #db.insert(res_knn, "robust")

    db.close()
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


def mkarate_wavelet():
    from utils.guass_charac_analyze import mkarate_wavelet_analyse, mkarate_wavelet_analyse_2
    data, _, _ = dataloader("mkarate", directed=False)
    wave_machine = GraphWave(data)
    eigenvalues = wave_machine._e
    sMin, sMax = scale_boundary(eigenvalues[1], eigenvalues[-1])
    scale = (sMin + sMax) / 2  # 根据GraphWave论文中推荐的尺度进行设置。
    print(sMin, sMax, scale)
    scale=30
    node1, node2, node3 = '3', '38', '20'
    node2idx, idx2node = wave_machine.node2idx, wave_machine.nodes
    wavelets = wave_machine.cal_all_wavelet_coeffs(scale=scale)
    index1, index2, index3 = node2idx[node1], node2idx[node2], node2idx[node3]
    wavelet1, wavelet2, wavelet3 = wavelets[index1], wavelets[index2], wavelets[index3]
    similarity = wave_machine.calc_wavelet_similarity(wavelets, method="wasserstein", hierachical=True, layers=5)
    mkarate_wavelet_analyse(scale, node1, wavelet1, node2, wavelet2, node3, wavelet3,
                            similarity[index1, index2], similarity[index1, index3])


def visulize_via_smilarity_tsne(name, db, label_class="SIR", perplexity=30, scale = 2, reused=False):
    from sklearn.manifold import TSNE
    graph, label_dict, n_class = dataloader(name, label=label_class, directed=False, similarity=False)
    wave_machine = GraphWave(graph)
    eigenvalues = wave_machine._e
    idx2node, node2idx = wave_machine.nodes, wave_machine.node2idx

    sMin, sMax = scale_boundary(eigenvalues[2], eigenvalues[-1])
    s = (sMin + sMax) / 2   # 根据GraphWave论文中推荐的尺度进行设置。
    print(sMin, sMax)
    scale = 1
    print("scale: ", scale)
    path = "../../output/{}_distance_{}_{}.csv".format(name, scale, perplexity)
    if not reused:
        coeff_mat = wave_machine.cal_all_wavelet_coeffs(scale=scale)
        print(coeff_mat[:5, :15])
        mat = wave_machine.parallel_calc_similarity(coeff_mat, layers=5, metric="wasserstein",
                                                    mode="distance", save_path=path)
    else:
        mat = read_distance(path, wave_machine.n_nodes)

    # mkarate, layer=1, scale=0.5, p=5, random=32
    # mkarate, layer=2, scale=0.5, p=5, random=35
    # mkarate, layer=3, scale=0.5, p=5, random=25
    # mkarate, layer=4, scale=0.5, p=5, random=25
    # mkarate, layer=5, scale=0.5, p=5, random=32
    res = TSNE(n_components=2, metric="precomputed", perplexity=perplexity, random_state=32).fit_transform(mat)
    tmp = {}
    for idx, node in enumerate(idx2node):
        tmp[node] = res[idx]
    save_vectors(tmp, "../../output/HSD_{}_{}_{}_tsne.csv".format(name, scale, perplexity))

    labels = []
    for idx, node in enumerate(idx2node):
        node_label = label_dict[node]
        labels.append(node_label)

    print(type(mat), len(mat), len(mat[0]))
    # 展示2维数据，参数tsne和perplexity没用
    """
    accuracy, balanced_accuracy, precision, recall, macro_f1, micro_f1 = evaluate_KNN_accuracy(X=mat, labels=labels, metric="precomputed", n_neighbor=20)

    res_knn = {"method": "HSD",
               "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
               "graph": name,
               "scale": scale,
               "metric": "wasserstein",
               "evaluate model": "KNN",
               "accuracy": accuracy,
               "balanced_accuracy": balanced_accuracy,
               "precision": precision,
               "recall": recall,
               "macro f1": macro_f1,
               "micro f1": micro_f1,
               "label": "SIR_2"
               }

    if db:
        db.insert(res_knn, "nodes classification")
    """
    plot_embeddings(idx2node, res, labels=labels, n_class=n_class, method="tsne", perplexity=30, node_text=False)


def bell_scales():
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import matplotlib.colors as colors
    import matplotlib.cm as cmx

    graph, label_dict, n_class = dataloader("bell", label="origin", directed=False, similarity=False)
    model = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=3, init='random')
    machine = GraphWave(graph)

    cm = plt.get_cmap("nipy_spectral")
    cNorm = colors.Normalize(vmin=0, vmax=n_class - 1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    markers = ['<', '*', 'x', 'D', 'H', 'x', 'D', '>', '^', "v", '1', '2', '3', '4', 'X', '.']

    e1, eN = machine._e[1], machine._e[-1]
    sMin, sMax = scale_boundary(e1, eN)
    step = (sMax - sMin) / 4
    scales = [50.0 + step * i * 500 for i in range(16)]
    for i, scale in enumerate(scales):
        plt.subplot(4, 4, i+1)
        plt.xlabel("scale = {} ".format(scale))
        plt.xticks([])
        plt.yticks([])

        embedding_dict = machine.single_scale_embedding(scale)
        embeddings = []
        labels = []
        nodes = []
        for node, vector in embedding_dict.items():
            embeddings.append(vector)
            labels.append(label_dict[node])
            nodes.append(node)

        data = model.fit_transform(embeddings)
        class_dict = defaultdict(list)
        for idx, node in enumerate(nodes):
            class_dict[labels[idx]].append(idx)

        for _class, _indices in class_dict.items():
            _class = int(_class)
            plt.scatter(data[_indices, 0], data[_indices, 1], s=60, marker=markers[_class], c=[scalarMap.to_rgba(_class)], label=_class)

        for idx, (x, y) in enumerate(data):
            plt.text(x, y, nodes[idx])

    plt.show()


if __name__ == '__main__':
    #start = time.time()
    #db = Database()

    #visulize_via_smilarity_tsne("usa", None, label_class="SIR_2", perplexity=100, reused=False)
    mkarate_wavelet()
    #bell_scales()
    #embedd("usa", label_class="SIR_2")
    #mkarate_wavelet()
    #print("all", time.time() - start)
    #_time_test("europe")
    #for scale in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.4, 0.5]
    #for scale in [0.01, 0.03, 0.05, 0.07, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2, 2.5, 3, 4, 5]:
    #    robustness("usa", probs=[1.0], cnt=1, metric="wasserstein", label="SIR_2", dim=64, scale=scale, percentile=0.9)
    #scalability_test(datasets=['bell', 'mkarate.edgelist', 'subway', 'railway', 'brazil', 'europe'])
