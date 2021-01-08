# -*- encoding: utf-8 -*-

from collections import defaultdict

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


def scatterplot(graphName, vectors, labels, nodes=None):
    vectors = np.asarray(vectors)
   # pca = PCA(n_components=2, whiten=False, random_state=42)
   # results = np.asarray(pca.fit_transform(vectors), dtype=np.float)
    tsne = TSNE(init="pca", n_components=2, perplexity=2, n_iter=5000, learning_rate=0.05, random_state=42)
    results = np.asarray(tsne.fit_transform(vectors), dtype=np.float)

    # df = pd.DataFrame(data={"node": nodes,
    #                         "x": results[:, 0],
    #                         "y": results[:, 1],
    #                         })
    # df.to_csv(f"output/node2vec/{graphName}_tsne.csv", columns=["node", "x", "y"], index=None)

    #visualize.plot_2D_points(nodes, results, labels)
    visualize.plot_node_str(nodes, results)


def visualize_from_csv():
    graph = "barbell"
    vector_dict = rw.read_vectors(f"output/node2vec/{graph}.csv")
    nodes = list(vector_dict.keys())
    vectors = list(vector_dict.values())
    scatterplot(graph, vectors, [], nodes=nodes)


if __name__ == '__main__':
    visualize_from_csv()
