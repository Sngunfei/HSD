# -*- encoding: utf-8 -*-

from collections import defaultdict

import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import euclidean_distances

from model import MultiHSD
from tools import dataloader, visualize, rw
import tools
from tools import evaluate
from tools.rw import read_vectors


def multi_HSD(graphName, hop, n_scales):
    graph, label_dict = dataloader.load_data(graphName, "default")
    model = MultiHSD(graph, graphName, hop, n_scales)
    model.init()
    embedding_dict = model.embed()

    nodes, labels = [], []
    vectors = np.empty(shape=(len(embedding_dict), len(embedding_dict['1'])))
    idx = 0
    for node, vector in embedding_dict.items():
        nodes.append(node)
        labels.append(label_dict[node])
        vectors[idx] = vector
        idx += 1
    #df = pd.DataFrame(data=vectors, index=nodes, columns=None, dtype=float)
    #df.to_csv(f"{graphName}.csv", header=False, float_format="%.8f")
    return nodes, vectors, labels


def relation_plot(graphName:str, vectors: np.ndarray, labels):
    n = len(vectors)
    n_class = 5
    vectors = np.asarray(vectors)
    corr = np.zeros(shape=(n_class, n_class))
    cnt_mat = np.zeros(shape=(n_class, n_class))
    distance_matrix = euclidean_distances(vectors, vectors)

    for idx1 in range(n):
        vector1, label1 = vectors[idx1], labels[idx1]
        label1 -= 1
        for idx2 in range(idx1 + 1, n):
            vector2, label2 = vectors[idx2], labels[idx2]
            label2 -= 1
            # 多种metric
            #distance = math.sqrt(np.linalg.norm(vector1 - vector2) / len(vector1))
            #distance = cosine_distances(vector1, vector2)
            distance = distance_matrix[idx1][idx2]
            corr[label1][label2] = (corr[label1][label2] * cnt_mat[label1][label2] + distance) / (
                        cnt_mat[label1][label2] + 1)
            corr[label2][label1] = corr[label1][label2]
            cnt_mat[label1][label2] += 1
            cnt_mat[label2][label1] = cnt_mat[label1][label2]

    mask = np.zeros_like(corr)
    #mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(n_class, n_class))
        ax = sns.heatmap(corr, mask=None, vmax=np.max(corr), square=True,
                         cmap="YlGnBu")  #, linewidths=.3 , annot=True, fmt=".3f")
        plt.savefig(f"{graphName}_relation.png")
        #plt.show()


def scatterplot(graphName, vectors, labels, nodes=None):
    vectors = np.asarray(vectors)
   # pca = PCA(n_components=2, whiten=False, random_state=42)
   # results = np.asarray(pca.fit_transform(vectors), dtype=np.float)
    tsne = TSNE(init="pca", n_components=2, perplexity=10, n_iter=5000, learning_rate=0.05, random_state=42)
    results = np.asarray(tsne.fit_transform(vectors), dtype=np.float)

    # df = pd.DataFrame(data={"node": nodes,
    #                         "x": results[:, 0],
    #                         "y": results[:, 1],
    #                         })
    # df.to_csv(f"{graphName}_HSD_tsne.csv", columns=["node", "x", "y"], index=None)

    #visualize.plot_2D_points(nodes, results, labels)
    visualize.plot_node_str(nodes, results)

# 热量流动
def heat_diffusion_plot():
    pass

if __name__ == '__main__':

    graphs = ['europe', 'usa', 'cora', 'bio_dmela', 'bio_grid_human']
    graphName = "barbell"
    nodes, vectors, labels = multi_HSD(graphName, hop=7, n_scales=100)
    scatterplot(graphName, vectors, labels, nodes)
