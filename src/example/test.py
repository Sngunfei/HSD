from example import parser
from model.GraphWave import GraphWave
from example import datasets
import networkx as nx

def run(scale, dataset="bell"):
    settings = parser.parameter_parser()
    parser.tab_printer(settings)

    data_path = datasets.edgelist[dataset]
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, nodetype=str, edgetype=float, data=[('weight', float)])

    wavelet_model = GraphWave(graph, settings)
    wavelet_model.dev_wavlet_KL(dataset, scale, save=False, fig=True, top=False)


if __name__ == '__main__':
    run(10, dataset="subway")

