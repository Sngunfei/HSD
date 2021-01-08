# -*- encoding: utf-8 -*-

import networkx as nx
import numpy as np

from model.multiscale_HSD import MultiHSD
from tools.isomorphism import get_isomorphism_label
from tools.util import merge_dicts_to_lists
from tools.visualize import plot_2D_points


def load_data():
    graph_name = "mkarate"
    G = nx.read_edgelist(f"../../data/graph/{graph_name}.edgelist", create_using=nx.Graph,
                         nodetype=str, edgetype=float, data=[("weight", float)])
    return G


def multi_HSD(G: nx.Graph, hop: int, n_scales: int):
    model = MultiHSD(G, "mkarate", hop, n_scales)
    model.init()
    embedding_dict = model.parallel_embed(n_workers=1)
    return embedding_dict


# PCA降维
def dimension_reduction(features):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    model = PCA(n_components=2, whiten=False, random_state=42)
    #model = TSNE(n_components=2, perplexity=10, learning_rate=1.0, n_iter=5000)
    results = model.fit_transform(np.array(features))
    return results


def run():
    G = load_data()
    embedding_dict = multi_HSD(G, hop=3, n_scales=100)
    # for hop in range(1, 8):
    label_dict = get_isomorphism_label(G, 1)

    nodes, embeddings, labels = [], [], []
    for node in nx.nodes(G):
        nodes.append(node)
        embeddings.append(embedding_dict[node])
        labels.append(label_dict[node])

    vectors = dimension_reduction(embeddings)
    plot_2D_points(nodes, vectors, labels)


if __name__ == '__main__':
    run()