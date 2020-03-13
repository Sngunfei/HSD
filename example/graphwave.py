# -*- encoding: utf-8 -*-

import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from example.parser import GraphWaveParameterParser, tab_printer
from ge.model.GraphWave import GraphWave
from ge.utils.dataloader import load_data
from ge.utils.rw import save_results
import pandas as pd
from sklearn.manifold import TSNE
from ge.utils.visualize import plot_embeddings
from ge.evaluate.evaluate import LR_evaluate, KNN_evaluate


def run(graphwave, label_dict, n_class):
    embedding_dict = graphwave.single_scale_embedding()

    nodes, labels, embeddings = [], [], []
    for node, embedding_vector in embedding_dict.items():
        nodes.append(node)
        labels.append(label_dict[node])
        embeddings.append(embedding_vector)

    if graphwave.graph_name in ['europe', 'usa']:
        lr_res = LR_evaluate(embeddings, labels)
        lr_res['scale'] = graphwave.scale
        lr_res['samples'] = params.sample_number
        lr_res['step_size'] = graphwave.step_size
        save_results(lr_res, "../results/lr/graphwave_{}.txt".format(graph_name))
        knn_res = KNN_evaluate(embeddings, labels)
        knn_res['scale'] = graphwave.scale
        knn_res['samples'] = params.sample_number
        knn_res['step_size'] = graphwave.step_size
        save_results(knn_res, "../results/knn/graphwave_{}.txt".format(graph_name))

    df = pd.DataFrame(data=embeddings, index=nodes, columns=None, dtype=float)
    # file_name: graphwave_mkarate_scale.csv
    df.to_csv("../embeddings/graphwave_{}_scale{}.csv".format(
        graphwave.graph_name, graphwave.scale), header=False, float_format="%.8f")

    tsne_res = TSNE(n_components=2, metric="euclidean", learning_rate=50.0, n_iter=2000,
               perplexity=params.tsne, random_state=params.random).fit_transform(embeddings)

    df = pd.DataFrame(data=tsne_res, index=nodes, columns=None, dtype=float)
    df.to_csv("../tsne_results/graphwave_{}_scale{}_perp{}.csv".format(
        graphwave.graph_name, graphwave.scale, params.tsne))

    figure_path = "../figures/graphwave_{}_scale{}_tsne{}.png".format(
        graphwave.graph_name, graphwave.scale, params.tsne)
    plot_embeddings(nodes, tsne_res, labels=labels, n_class=n_class, node_text=False, save_path=figure_path)


if __name__ == '__main__':
    params = GraphWaveParameterParser()
    tab_printer(params)

    graph_name = params.graph
    if graph_name in ["barbell", "mkarate"]:
        graph, label_dict, n_class = load_data(graph_name, label_name="origin")
    else:  # europe, usa
        graph, label_dict, n_class = load_data(graph_name, label_name="SIR")

    #def __init__(self, graph, heat_coefficient=5.0, sample_number=16, step_size=20.0):
    model = GraphWave(graph, graph_name, params.scale, params.sample_number, params.step_size)
    run(model, label_dict, n_class)

