# -*- coding:utf-8 -*-

from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx
import matplotlib.pyplot as plt


def dataloader(name="", directed=False, similarity=False, scale=None, metric=None):
    """
    Loda graph data by dataset name.
    :param name: dataset name, str
    :param directed: bool, if True, return directed graph.
    :param similarity: similarity data or edgelist.
    :param scale: i.e. head coefficient, int
    :param metric: similarity metric, like L1 and L2, etc.
    :return: graph, node labels, number of node classes.
    """

    label_path = "../../data/{}.label".format(name)
    if not similarity:
        edge_path = "../../data/{}.edgelist".format(name)
    else:
        metric = str.lower(metric)
        edge_path = "../../similarity/{}_{}_{}.csv".format(name, scale, metric)
        directed = False # similarity can't be directed.

    if directed:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.DiGraph,
                                 edgetype=float, data=[('weight', float)])
    else:
        graph = nx.read_edgelist(path=edge_path, create_using=nx.Graph,
                                 edgetype=float, data=[('weight', float)])

    label_dict, num_class = read_label(label_path)

    return graph, label_dict, num_class



def evaluate_LR_accuracy(embeddings=None, labels=None, random_state=42):
    """
    Evaluate embedding effect using Logistic Regression. Mode = One vs Rest (OVR)

    :param embeddings: learned representation vectors. shape=(n_samples, n_dim)
    :param labels: nodes' label for classification.
    :param random_state: random seed.
    :return: Accuracy score, float.
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
    #from sklearn.multiclass import OneVsRestClassifier

    xtrain, xtest, ytrain, ytest = train_test_split(embeddings, labels, test_size=0.2,
                                                    random_state=random_state, shuffle=True)

    lrc = LogisticRegressionCV(cv=10, solver="lbfgs", penalty='l2', max_iter=1000, verbose=0, multi_class='ovr')
    lrc.fit(xtrain, ytrain)
    preds = lrc.predict(xtest)
    score = accuracy_score(preds, ytest)
    balanced_score = balanced_accuracy_score(ytest, preds)
    reprot = classification_report(ytest, preds)
    print("logistic regression(ovr) accuracy score:{}.".format(score))
    print("logistic regression(ovr) balanced accuracy score:{}.".format(balanced_score))

    print("classification report: ")
    print(reprot)


# todo
def evaluate_SVC_accuracy(embeddings=None, labels=None, random_state=42):
    """
    Evaluate embedding effect using support vector classifier. Mode = One vs Rest (OVR)

    :param embeddings: learned representation vectors. shape=(n_samples, n_dim)
    :param labels: nodes' label for classification.
    :param random_state: random seed.
    :return: Accuracy score, float.
    """
    raise NotImplementedError("The svc for node embedding classification is not implented yet.")


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



# todo
"""
对度数的研究。考虑1，2,3阶。最好能够很好的展示出来。
"""
def flight_data_analyze():

    flights = ['brazil', 'europe', 'usa']
    graphs, label_dicts, n_classes = [], [], []
    for name in flights:
        graph, label_dict, n_class = dataloader(name=name)
        graphs.append(graph)
        label_dicts.append(label_dict)
        n_classes.append(n_class)


    def get_degree_info(graph, label_dict):
        """
        Get nodes degree information.
        :return: dict{label: [degrees]}
        """
        class_degree = dict()
        nodes = list(nx.nodes(graph))
        for node in nodes:
            label = label_dict[node]
            degree = nx.degree(graph, node)
            class_degree[label] = class_degree.get(label, []) + [degree]

        return class_degree

    color = ['r', 'g', 'b', 'y']
    markers = ['+', 'o', '<', '*', 'D', 'x', 'H', '>', '^', "v", '1', '2', '3', '4', 'X', '.']

    plt.figure()
    for idx, data in enumerate(flights):
        plt.subplot(221 + idx)
        class_degree = get_degree_info(graphs[idx], label_dicts[idx])
        i = 0
        for label, degrees in class_degree.items():
            x = list(range(len(degrees)))
            avg_degree = np.mean(degrees)
            plt.scatter(x, degrees, c=color[i], marker=markers[i])
            plt.plot(x, [avg_degree] * len(x), c=color[i])
            i += 1
        plt.xlabel(data)
        plt.ylabel("degree")
    #plt.show()
    plt.savefig("../../image/test.png")


def subway_data_analyze():
    graph, label_dict, _ = dataloader("subway")
    class_degree = dict()
    for node, label in label_dict.items():
        degree = nx.degree(graph, node)
        class_degree[label] = class_degree.get(label, []) + [degree]
    color = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'b', 'b', 'b']
    markers = ['+', 'o', '<', '*', 'D', 'x', 'H', '>', '^', "v", '1', '2', '3', '4', 'X', '.']

    plt.figure()
    i = 1
    for label, degrees in class_degree.items():
        x = list(range(len(degrees)))
        avg_degree = np.mean(degrees)
        plt.scatter(x, degrees, c=color[i], marker=markers[i])
        plt.plot(x, [avg_degree] * len(x), c=color[i])
        i += 1
    plt.xlabel("subway")
    plt.ylabel("degree")
    plt.show()
    plt.savefig("../../image/subway_degree.png")


if __name__ == '__main__':
    #write_label("G:\pyworkspace\graph-embedding\data\subway.edgelist")
    subway_data_analyze()
