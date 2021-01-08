# -*- encoding: utf-8 -*-

import sys
sys.path.append("/home/data/users/master/2019/songyunfei/workspace/py/HSD")

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import wasserstein_distance

from model import GraphWave
from tools import util, dataloader, evaluate, const, visualize


def node_sets(label_dict: dict) -> list:
    classes = [[]]
    for node, label in label_dict.items():
        label = int(label)
        if label >= len(classes):
            classes.append([node])
        else:
            classes[label].append(node)
    return classes


def visualize_run(graph_name):
    graph, label_dict = dataloader.load_data(graph_name, "default")
    classes = node_sets(label_dict)

    graphwave = GraphWave.GraphWave(graph)
    scale_min, scale_max = GraphWave.recommend_scale_range(graphwave.eigenvalues)
    candidate_scales = np.linspace(scale_min, scale_max * 2, 10)
    sample_points = np.linspace(0, 50, 100)

    nodes_dict = {}
    for node in graphwave.nodes:
        nodes_dict[node] = node

    candidate_scales = [2.5]
    for scale in candidate_scales:
        print(scale)
        wavelets = graphwave.calculate_wavelets(scale, approx=False)
        embedding_dict = graphwave.embed(sample_points)

        for node_list in classes:
            for i in range(len(node_list)):
                for j in range(i+1, len(node_list)):
                    assert np.sum(np.abs(embedding_dict[node_list[i]] - embedding_dict[node_list[j]])) < 0.001

        for i in range(len(classes)):
            node1 = classes[i][0]
            for j in range(i+1, len(classes)):
                node2 = classes[j][0]
                print(i, j, np.linalg.norm(embedding_dict[node1] - embedding_dict[node2]))

        nodes, vectors, labels = util.merge_dicts_to_lists(nodes_dict, embedding_dict, label_dict)
        vectors = reduce_2D_dimension(vectors, method="TSNE", random_seed=42)

        #visualize.plot_2D_points(nodes, vectors, labels)
        visualize.plot_node_str(nodes, vectors)


def run(graph_name, label_type):
    graph, label_dict = dataloader.load_data(graph_name, label_type)
    graphwave = GraphWave.GraphWave(graph)
    scale_min, scale_max = GraphWave.recommend_scale_range(graphwave.eigenvalues)
    candidate_scales = np.linspace(scale_min, scale_max*2, 10)
    candidate_sample_points = [np.linspace(0, upper, 100) for upper in range(10, 100, 10)]

    nodes_dict = {}
    for node in graphwave.nodes:
        nodes_dict[node] = node

    accuracy_file = open(f"{graph_name}_accuacy.txt", mode="w+", encoding="utf-8")
    max_accuracy = 0.0
    for scale in candidate_scales:
        graphwave.calculate_wavelets(scale, approx=True)
        for points in candidate_sample_points:
            embedding_dict = graphwave.embed(points)

            nodes, vectors, labels = util.merge_dicts_to_lists(nodes_dict, embedding_dict, label_dict)
            acc = evaluate.KNN_evaluate(vectors, labels, cv=5, n_neighbor=20)
            max_accuracy = max(acc, max_accuracy)

            accuracy_file.write(f"scale: {scale}, sample_upper: {points[-1]}, accuracy: {acc} \n")
            accuracy_file.flush()
    accuracy_file.close()
    print(f"max accuracy: {max_accuracy}")


def save(graph_name, nodes, vectors):
    df = pd.DataFrame(data={"node": nodes, "x": vectors[:, 0], "y": vectors[:, 1]})
    df.to_csv(f"{graph_name}_pca.csv", columns=["node", "x", "y"], index=None)


def reduce_2D_dimension(vecotrs: np.ndarray, method: str, random_seed: int) -> np.ndarray:
    if method == "PCA":
        model = PCA(n_components=2, whiten=False, random_state=random_seed)
    elif method == "TSNE":
        model = TSNE(init="pca", n_components=2, perplexity=10,
                     n_iter=5000, learning_rate=0.05, random_state=random_seed)
    else:
        raise NotImplementedError(f"{method} not implemented")
    results = np.asarray(model.fit_transform(vecotrs), dtype=np.float)
    return results


def calculate_wasserstein_distance(wavelet_coeffs: np.ndarray) -> np.ndarray:
    number_nodes = len(wavelet_coeffs)
    distance_matrix = np.zeros(shape=(number_nodes, number_nodes))

    for idx1 in range(0, number_nodes):
        for idx2 in range(idx1 + 1, number_nodes):
            distance_matrix[idx1, idx2] = distance_matrix[idx2, idx1] = wasserstein_distance(wavelet_coeffs[idx1], wavelet_coeffs[idx2])

    return distance_matrix


if __name__ == '__main__':
    graphs = ['europe', 'usa', 'cora', 'bio_dmela', 'bio_grid_human']
    #run("mkarate", "default")
    visualize_run("barbell")
