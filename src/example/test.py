from example import parser
from model.GraphWave import GraphWave
from model import AutoEncoder
from example import datasets
import networkx as nx
from utils.plt import plot_embeddings
import numpy as np

def run(scale, dataset, method):
    settings = parser.parameter_parser()
    #parser.tab_printer(settings)

    #data_path = datasets.edgelist[dataset]
    data_path = "G:\pyworkspace\graph-embedding\data\\fb1.edgelist"
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, nodetype=str, edgetype=float, data=[('weight', float)])

    wavelet_model = GraphWave(graph, settings)
    #wavelet_model.embedding_similarity(dataset, scale)
    embeddings = wavelet_model.single_scale_embedding(scale)

    #wavelet_model.dist_measure(scale, method, save_path="G:\pyworkspace\graph-embedding\out\{}_{}_{}.txt".format(dataset, scale, method))

    #embeddings = wavelet_model.single_scale_embedding(scale)
    #wavlet_coeffs = wavelet_model.dev_cal_all_wavelet_coeffs(scale)
    # wavelet_model.save_coeff_fig(dataset, scale, 20)
    #embeddings = AutoEncoder.create_embedding(wavlet_coeffs, d_in=wavelet_model.n_nodes)

    plot_embeddings(wavelet_model.nodes, embeddings, method="tsne")

    #wavelet_model.dev_wavlet_KL(dataset, scale, save=False, fig=True, top=False)
    #wavelet_model.dev_coeff_autoencoder(scale)


if __name__ == '__main__':
    run(5, dataset="fb1", method="L1")

