# -*- coding:utf-8 -*-

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np
from sklearn.manifold import TSNE


def dataloader(name="", directed=False):
    """
    Loda graph data by dataset name.
    :param name: dataset name, str
    :param directed: bool, if True, return directed graph.
    :return: graph, node labels, number of node classes.
    """
    import networkx as nx
    edge_path = "../../data/{}.edgelist".format(name)
    label_path = "../../data/{}.label".format(name)

    if directed:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.Graph,
                                 edgetype=float, data=[('weight', float)])
    else:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.DiGraph,
                                 edgetype=float, data=[('weight', float)])

    label_dict, num_class = read_label(label_path)

    return graph, label_dict, num_class



def read_label(path):
    """
    Get graph nodes' label.
    :param path: label file path.
    :return: return dict-type, {node:label}, number of class.
    """
    label_set = set()
    with open(path, mode="r", encoding="utf-8") as fin:
        label_dict = dict()
        while True:
            line = fin.readline()
            if not line:
                break
            node, label = line.strip().split(" ")
            label_dict[node] = label
            label_set.add(label)

    return label_dict, len(label_set)


def write_label(data_path):
    import networkx as nx
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, nodetype=str, edgetype=float, data=[('weight', float)])

    fout = open("G:\pyworkspace\graph-embedding\out\subway_label_2.txt", mode="w+", encoding="utf-8")

    nodes = list(nx.nodes(graph))
    rings = dict()
    for node1 in nodes:
        hop1, hop2 = 0, 0
        for node2 in nodes:
            length = nx.dijkstra_path_length(graph, node1, node2)
            if length > 2:
                continue
            elif length == 1:
                hop1 += 1
            elif length == 2:
                hop2 += 1
        rings[node1] = [hop1, hop2]

    for node, hop in rings.items():
        hop1 = min(hop[0], 4)
        hop2 = min(hop[1], 6) // 3 + 1
        label = (hop1 - 1) * 3 + hop2
        fout.write("{} {}\n".format(node, label))

    fout.close()


def preprocess_nxgraph(graph):
    """
    建立图节点与标号之间的映射关系，方便采样。
    :param graph:
    :return:
    """
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def compute_cheb_coeff_basis(scale, order):
    xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * math.pi)
                   for i in range(1, order + 1)])
    basis = [np.ones((1, order)), np.array(xx)]
    for k in range(order + 1 - 2):
        basis.append(2 * np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    products = np.einsum("j,ij->ij", f, basis)
    coeffs = 2.0 / order * products.sum(1)
    coeffs[0] = coeffs[0] / 2
    return list(coeffs)


def cluster_evaluate(embeddings, labels, class_num=2, perplexity=5):
    """
        Unsupervised setting: We assess the ability of each method to embed close together nodes
        with the same ground-truth structural role. We use agglomerative clustering (with single linkage)
        to cluster embeddings learned by each method and evaluate the clustering quality via:
            (1) homogeneity, conditional entropy of ground-truth structural roles given the predicted clustering;
            (2) completeness, a measure of how many nodes with the same ground-truth structural role are assigned to the same cluster;
            (3) silhouette score, a measure of intra-cluster distance vs. inter-cluster distance.

        Supervised setting: We assess the performance of learned embeddings for node classifcation.
        Using 10-fold cross validation, we predict the structural role (label) of each node in the test set
        based on its 4-nearest neighbors in the training set as determined by the embedding space.
        The reported score is then the average accuracy and F1-score over 25 trials.
    """
    #model = TSNE(n_components=2, random_state=42, n_iter=5000, perplexity=perplexity, init="pca")
    #embeddings = model.fit_transform(embeddings)
    clusters = AgglomerativeClustering(n_clusters=class_num, linkage='single').fit_predict(embeddings)
    h, c, v = metrics.homogeneity_completeness_v_measure(labels, clusters)
    s = metrics.silhouette_score(embeddings, clusters)
    print("cluster:", clusters, "labels:", labels)
    print("homogeneity: ", h)
    print("completeness: ", c)
    print("v-score: ", v)
    print("silhouette: ", s)

    return h, c, v, s


if __name__ == '__main__':
    #write_label("G:\pyworkspace\graph-embedding\data\subway.edgelist")
    a = [1, 2, 3]
    c, d, e = a
    print(c, d, e)



