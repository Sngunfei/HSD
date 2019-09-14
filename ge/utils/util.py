from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np
from sklearn.manifold import TSNE


def read_label(path):
    """
    读取节点的标签信息, 返回字典。
    """
    with open(path, mode="r", encoding="utf-8") as fin:
        labels = dict()
        while True:
            line = fin.readline()
            if not line:
                break
            node, label = line.strip().split(" ")
            labels[node] = label

    return labels


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
    model = TSNE(n_components=2, random_state=42, n_iter=5000, perplexity=perplexity, init="pca")
    embeddings = model.fit_transform(embeddings)
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



