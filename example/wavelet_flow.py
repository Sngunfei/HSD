# -*- coding:utf-8 -*-
import numpy as np
from utils.util import dataloader, save_vectors, sparse_process
from evaluate.evaluate import evaluate_LR_accuracy
from utils.distance import Hellinger_distance
from utils.visualize import plot_embeddings
from model.GraphWave import GraphWave
import pandas as pd
import math
from sklearn.manifold import TSNE
from model.LaplacianEigenmaps import LaplacianEigenmaps
from model.LocallyLinearEmbedding import LocallyLinearEmbedding


def HSD_LE(name, dim=16, threshold=None, percentile=None):
    save_path = "../../distance/{}.edgelist".format(name)

    new_graph, _, _ = dataloader(name=name, path=save_path, directed=False)
    new_graph = sparse_process(new_graph, percentile=percentile)

    model = LaplacianEigenmaps(new_graph, dim=dim)
    embeddings_dict = model.create_embedding()
    return embeddings_dict


def HSD_LLE(name, dim=16, threshold=None, percentile=None):
    save_path = "../../distance/{}.edgelist".format(name)
    new_graph, _, _ = dataloader(name=name, path=save_path, directed=False)
    new_graph = sparse_process(new_graph, percentile=percentile)

    model = LocallyLinearEmbedding(new_graph, dim=dim)
    embeddings_dict = model.sklearn_lle(n_neighbors=10, dim=dim, random_state=42)
    return embeddings_dict


def embed(data_name, dim=64, label_class="SIR_2", perplexity=30):
    _, label_dict, n_class = dataloader(data_name, directed=False, label=label_class)
    embedding_dict, method = HSD_LE(data_name, dim=dim, threshold=None, percentile=0.7), "HSDLE"
    #embedding_dict, method = HSD_LLE(data_name, dim=dim, threshold=None, percentile=0.7), "HSDLLE"

    nodes = []
    labels = []
    embeddings = []
    for node, embedding in embedding_dict.items():
        nodes.append(node)
        embeddings.append(embedding)
        labels.append(label_dict[node])

    evaluate_LR_accuracy(embeddings, labels, random_state=42)

    _2d_data = plot_embeddings(nodes, embeddings, labels, n_class, method="tsne", init="random",
                               perplexity=perplexity, node_text=False, random_state=35)
    tmp = {}
    for idx, node in enumerate(nodes):
        tmp[node] = _2d_data[idx]
    save_vectors(tmp, "../../output_Hellinger/{}_{}_tsne_{}.csv".format(method, data_name, perplexity))


def multiScalesWavelet(data_name, label_class="origin"):
    graph, label_dict, n_class = dataloader(data_name, directed=False, label=label_class)

    wave_machine = GraphWave(graph)
    eigenvalues = wave_machine._e
    print(min(eigenvalues), max(eigenvalues))
    # 尺度参数
    scales = np.exp(np.linspace(np.log(0.01), np.log(max(eigenvalues)), 200))
    nodes = wave_machine.nodes
    infos = dict() # 各尺度下的数据汇总
    max_hop = 5
    # 层级结构，dict(dict())
    hierarchy = wave_machine.get_nodes_layers_bfs(max_hop)
    for scale in scales:
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        state = dict() # 单尺度下的数据
        for idx, node in enumerate(nodes):
            p = [coeffs[idx, idx]]
            for k_hop in range(1, max_hop):
                k_hop_neighbors = hierarchy[idx].get(k_hop, [])
                if not k_hop_neighbors:
                    k_hop_sum = 0.0
                else:
                    k_hop_sum = np.sum([coeffs[idx][neighbor] for neighbor in k_hop_neighbors])
                p.append(k_hop_sum)
            if math.isclose(1.0 - sum(p), 0.0, abs_tol=1e-6):
                p.append(0.0)
            else:
                p.append(1.0 - sum(p))
            state[idx] = p
        infos[scale] = state

    # 各尺度下分别计算距离
    dist_mat = np.zeros((len(nodes), len(nodes)), dtype=np.float)
    for scale, state in infos.items():
        for idx1 in range(len(nodes)):
            for idx2 in range(idx1+1, len(nodes)):
                p, q = state[idx1], state[idx2]
                d = Hellinger_distance(p, q)
                dist_mat[idx1, idx2] += d
                dist_mat[idx2, idx1] += d
    dist_mat = dist_mat / len(scales) # 取均值

    # ---- reindex -----
    node_int = [int(node) for node in nodes]
    df = pd.DataFrame(data=dist_mat, dtype=np.float, index=node_int, columns=node_int)
    df.sort_index(axis=1, inplace=True)
    df.sort_index(axis=0, inplace=True)
    df.to_csv("../../distance/{}_distance.csv".format(data_name), mode="w+", encoding="utf-8", index=True, header=True)


def experiment(data_name, perplexity, label):
    graph, label_dict, n_class = dataloader(data_name, label=label, directed=False)
    wavelet_machine = GraphWave(graph)
    idx2node = wavelet_machine.idx2node

    df = pd.read_csv("../../distance/{}_distance.csv".format(data_name), index_col=0, header=0)
    mat = df.to_numpy()

    """
    fout = open("../../distance/europe.edgelist", mode="w+", encoding="utf-8")
    for idx in range(len(mat)):
        for idx2 in range(idx+1, len(mat)):
            fout.write("{} {} {}\n".format(idx, idx2, mat[idx, idx2]))
    fout.close()
    """

    res = TSNE(n_components=2, metric="precomputed", perplexity=perplexity, random_state=24).fit_transform(mat)
    tmp = {}
    labels = []
    for idx in range(len(mat)):
        tmp[str(idx+1)] = res[idx]
        labels.append(label_dict[str(idx+1)])
    print(len(mat))
    save_vectors(tmp, "../../output_Hellinger/HSD_{}_{}_tsne.csv".format(data_name, perplexity))

    #evaluate_KNN_accuracy(X=mat, labels=labels, metric="precomputed", n_neighbor=20)

    plot_embeddings(idx2node, res, labels=labels, n_class=n_class, method="tsne", perplexity=30, node_text=False)

if __name__ == '__main__':
    #multiScalesWavelet("usa")
    #experiment("mkarate", 10, "origin")
    embed("europe", dim=64, label_class="SIR_2", perplexity=15)
