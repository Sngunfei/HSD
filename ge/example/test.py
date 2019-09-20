from example import parser
from model.GraphWave import GraphWave
from model import AutoEncoder
from example import datasets
import networkx as nx
from utils.visualize import plot_embeddings, plot_subway_embedding
from utils.util import read_label
import numpy as np

def hse(scale, dataset, method, save):
    settings = parser.parameter_parser()
    data_path = datasets.edgelist[dataset]
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, nodetype=str, edgetype=float, data=[('weight', float)])
    wavelet_model = GraphWave(graph, settings)
    #wavelet_model.fb1_dist(scale)

    dists = wavelet_model.dist_measure(scale, method, save_path="G:\pyworkspace\graph-embedding\out\{}_{}_{}.txt".format(dataset, scale, method) if save else None)
    """
    label_dict = read_label("G:\pyworkspace\graph-embedding\out\{}_label.txt".format(dataset))
    labels = []
    for node in wavelet_model.nodes:
        labels.append(int(label_dict[node]))
    embeddings = wavelet_model.single_scale_embedding(scale)
    plot_embeddings(wavelet_model.nodes, embeddings, label=True, labels=labels, method="tsne")
    """
    """
    wavlet_coeffs = wavelet_model.dev_cal_all_wavelet_coeffs(scale)

    fout1 = open("G:\pyworkspace\graph-embedding\\coeff.txt", mode="w+", encoding="utf-8")
    fout2 = open("G:\pyworkspace\graph-embedding\\embed.txt", mode="w+", encoding="utf-8")

    for i in range(wavelet_model.n_nodes):
        coe = list(wavlet_coeffs[i])
        coe = ",".join(list(map(str, coe)))
        fout1.write("{} {}\n".format(wavelet_model.nodes[i], coe))
        fout2.write("{} {}\n".format(wavelet_model.nodes[i], embeddings[i]))
    fout1.close()
    fout2.close()
    """

    # wavelet_model.save_coeff_fig(dataset, scale, 20)
    #embeddings = AutoEncoder.create_embedding(wavlet_coeffs, d_in=wavelet_model.n_nodes)


    #wavelet_model.dev_wavlet_KL(dataset, scale, save=False, fig=True, top=False)
    #wavelet_model.dev_coeff_autoencoder(scale)


def graphwave(scale, dataset):
    settings = parser.parameter_parser()
    data_path = datasets.edgelist[dataset]
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, nodetype=str, edgetype=float,
                             data=[('weight', float)])
    wavelet_model = GraphWave(graph, settings)
    label_dict = read_label("G:\pyworkspace\graph-embedding\out\{}_label_2.txt".format(dataset))
    labels = []
    for node in wavelet_model.nodes:
        labels.append(int(label_dict[node]))

    embeddings = wavelet_model.single_scale_embedding(scale)
    plot_subway_embedding(wavelet_model.nodes, embeddings, labels=labels)



# todo
def nothing():
    """
    整合各种model
    :return:
    """
    raise NotImplementedError("1")


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=5)
    graphwave(20, dataset="subway")
    #hse(10, "subway", method="L3", save=True)

