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
from sklearn.metrics.pairwise import euclidean_distances

from model import MultiHSD
from tools import dataloader
from tools import evaluate
from tools.hierarchy import read_hierarchy
from tools.rw import read_vectors

def multi_HSD(graphName, hop, n_scales):
    graph = nx.read_edgelist(f"../../data/graph/{graphName}.edgelist", create_using=nx.Graph, edgetype=float,
                             data=[('weight', float)])
    label_dict = dataloader.read_label(f"../../data/label/{graphName}_PageRank.label")

    model = MultiHSD(graph, graphName, hop, n_scales)
    model.init2()
    model.hierarchy = read_hierarchy(f"../../data/hierarchy/{graphName}.layers", hop)
    embedding_dict = model.parallel_embed(n_workers=4)

    nodes, vectors, labels = [], np.empty(shape=(len(embedding_dict), len(embedding_dict['1']))), []
    idx = 0
    for node, vector in embedding_dict.items():
        nodes.append(node)
        labels.append(label_dict[node])
        vectors[idx] = vector
        idx += 1

    df = pd.DataFrame(data=vectors, index=nodes, columns=None, dtype=float)
    df.to_csv(f"{graphName}.csv", header=False, float_format="%.8f")

    return nodes, vectors, labels

graphName = ""

def main():
    import os
    graphs = ['europe', 'usa', 'cora', 'bio_dmela', 'bio_grid_human']
    global graphName
    graphName = "cora"
    if os.path.exists(f"{graphName}.csv"):
        embedding_dict = read_vectors(f"{graphName}.csv")
        label_dict = dataloader.read_label(f"../../data/label/{graphName}_PageRank.label")
        nodes, vectors, labels = [], np.empty(shape=(len(embedding_dict), len(embedding_dict['1']))), []
        idx = 0
        for node, vector in embedding_dict.items():
            nodes.append(node)
            labels.append(label_dict[node])
            vectors[idx] = vector
            idx += 1
    else:
        print("embed...")
        nodes, vectors, labels = multi_HSD(graphName, hop=2, n_scales=50)

    PageRank_val = evaluate.KNN_evaluate(vectors, labels, cv=5, n_neighbor=20)
    scatterplot(vectors, labels, nodes)
    relation_plot(vectors, labels)
    # pca = PCA(n_components=2, random_state=42)
    # tsne = TSNE(n_components=2, perplexity=3, n_iter=2000, random_state=42)
    # pca_results = pca.fit_transform(vectors)
    # tsne_results = tsne.fit_transform(vectors)
    # plot_embeddings(pca_results, labels, save_path=None)
    # plot_embeddings(tsne_results, labels, save_path=None)

def relation_plot(vectors: np.ndarray, labels):
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

            #distance = math.sqrt(np.linalg.norm(vector1 - vector2) / len(vector1))
            #distance = cosine_distances(vector1, vector2)
            distance = distance_matrix[idx1][idx2]

            corr[label1][label2] = (corr[label1][label2] * cnt_mat[label1][label2] + distance) / (
                        cnt_mat[label1][label2] + 1)
            corr[label2][label1] = corr[label1][label2]
            cnt_mat[label1][label2] += 1
            cnt_mat[label2][label1] = cnt_mat[label1][label2]

    global graphName

    mask = np.zeros_like(corr)
    #mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(n_class, n_class))
        ax = sns.heatmap(corr, mask=None, vmax=np.max(corr), square=True,
                         cmap="YlGnBu")  #, linewidths=.3 , annot=True, fmt=".3f")
        plt.savefig(f"{graphName}_relation.png")
        #plt.show()

def scatterplot(vectors: np.ndarray, labels, nodes=None):
    vectors, labels = np.asarray(vectors), np.asarray(labels)
    pca = PCA(n_components=2)
    #pca = kernel_pca.KernelPCA(n_components=2, kernel="linear", gamma=0.01)
    results = np.asarray(pca.fit_transform(vectors), dtype=np.float)

    #tsne = TSNE(init="pca", n_components=2, perplexity=7, n_iter=5000, learning_rate=0.05, random_state=10)
    #results = np.asarray(tsne.fit_transform(vectors), dtype=np.float)

    df = pd.DataFrame(data={"node": nodes,
                            "x": results[:, 0],
                            "y": results[:, 1],
                            "label": labels})
    global graphName
    df.to_csv(f"{graphName}_pca.csv", columns=["node", "x", "y"], index=None)
    #print(df.head(10))
    #sns.set_theme(style="ticks")

    plot_embeddings(results, labels, nodes=nodes, save_path=f"{graphName}_scatter.png")
    #ax = sns.scatterplot(data=df, x="x", y="y", hue="label", markers="label", legend=False)
    #plt.show()

def plot_embeddings(embeddings, labels, save_path=None, nodes=None):
    matplotlib.use("TkAgg")
    fig = plt.figure()
    markers = ['o', '*', 'x', '<', '1', 'D', '>', '^', "v", 'p', '2', '3', '4', 'X', '.']
    cm = plt.get_cmap("nipy_spectral")
    cNorm = colors.Normalize(vmin=0, vmax=5)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    class_dict = defaultdict(list)
    for idx in range(len(labels)):
        class_dict[int(labels[idx])].append(idx)

    info = sorted(class_dict.items(), key=lambda item: item[0])
    for _class, _indices in info:
        plt.scatter(embeddings[_indices, 0],
                    embeddings[_indices, 1],
                    s=100,
                    marker=markers[_class % len(markers)],
                    c=[scalarMap.to_rgba(_class)], label=_class)

        for idx in _indices:
            x, y = embeddings[idx, 0], embeddings[idx, 1]
            node_str = str(int(nodes[idx]) + 1)
            if node_str in ["1", "37", "34", "42", "6", "7", "15", "16", "18", "19", "21", "22", "23"]:
                plt.text(x + 0.1, y + 0.1, s=node_str)

    plt.legend()
    plt.xticks([])
    plt.yticks([])
    if save_path:
        plt.savefig(save_path)
    #plt.show()

def visualize(graphName: str, path: str):
    embedding_dict = read_vectors(path)
    label_dict = dataloader.read_label(f"../../data/label/{graphName}.label")
    nodes, vectors, labels = [], [], []
    for node, vector in embedding_dict.items():
        nodes.append(node)
        vectors.append(vector)
        labels.append(label_dict[node])
    #relation_plot(vectors, labels)
    scatterplot(vectors, labels, nodes)

if __name__ == '__main__':
    main()
    #visualize("mkarate", "struc2vec_mkarate_64.csv")
