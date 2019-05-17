from example import parser
from model.GraphWave import GraphWave
from model import AutoEncoder
from example import datasets
import networkx as nx
from utils.plt import plot_embeddings

def run(scale, dataset="bell"):
    settings = parser.parameter_parser()
    parser.tab_printer(settings)

    data_path = datasets.edgelist[dataset]
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, nodetype=str, edgetype=float, data=[('weight', float)])

    wavelet_model = GraphWave(graph, settings)
    wavelet_model.dist_measure(scale)
    #wavlet_coeffs = wavelet_model.dev_cal_all_wavelet_coeffs(scale)
    # wavelet_model.save_coeff_fig(dataset, scale, 20)
    #embeddings = AutoEncoder.create_embedding(wavlet_coeffs, d_in=wavelet_model.n_nodes)
    #plot_embeddings(wavelet_model.nodes, embeddings, method="tsne")

    #wavelet_model.dev_wavlet_KL(dataset, scale, save=False, fig=True, top=False)
    #wavelet_model.dev_coeff_autoencoder(scale)


if __name__ == '__main__':
    run(1.5, dataset="bell")

