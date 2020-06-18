# -*- encoding: utf-8 -*-

"""
Struc2vec experiments setup.
"""
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from example.parser import Struc2vecParameterParser, tab_printer
from ge.utils.dataloader import load_data
from ge.model.struc2vec import Struc2Vec
from ge.utils.rw import save_results
from ge.evaluate.evaluate import LR_evaluate, KNN_evaluate, cluster_evaluate
import pandas as pd
from sklearn.manifold import TSNE
from ge.utils.visualize import plot_embeddings


def run(model, label_dict, n_class, params):

    model.train(iter=5)
    embedding_dict = model.get_embeddings()

    nodes, labels, embeddings = [], [], []
    for node, embedding_vector in embedding_dict.items():
        nodes.append(node)
        labels.append(label_dict[node])
        embeddings.append(embedding_vector)

    _res = {'walk_length': model.walk_length,
            'walk_num': model.walk_num,
            'stay_prob': model.stay_prob}

    if model.graph_name in ['europe', 'usa', 'varied_graph']:
        cluster_evaluate(embeddings, labels, n_class)

        lr_res = LR_evaluate(embeddings, labels)
        lr_res.update(_res)
        save_results(lr_res, "../results/lr/struc2vec_{}.txt".format(graph_name))

        knn_res = KNN_evaluate(embeddings, labels)
        knn_res.update(_res)
        save_results(knn_res, "../results/knn/struc2vec_{}.txt".format(graph_name))

    df = pd.DataFrame(data=embeddings, index=nodes, columns=None, dtype=float)
    # file_name: struc2vec_mkarate_walklength_numwalks.csv
    df.to_csv("../embeddings/struc2vec_{}_length{}_num{}.csv".format(
        model.graph_name, model.walk_length, model.walk_num), header=False, float_format="%.8f")

    tsne_res = TSNE(n_components=2, metric="euclidean", learning_rate=50.0, n_iter=2000,
                    perplexity=params.tsne, random_state=params.random).fit_transform(embeddings)

    df = pd.DataFrame(data=tsne_res, index=nodes, columns=None, dtype=float)
    df.to_csv("../tsne_results/struc2vec_{}_length{}_num{}.csv".format(
        model.graph_name, model.walk_length, model.walk_num))

    figure_path = "../figures/struc2vec_{}_length{}_num{}.png".format(
        model.graph_name, model.walk_length, model.walk_num)
    plot_embeddings(nodes, tsne_res, labels=labels, n_class=n_class, node_text=False, save_path=figure_path)


def exec(graph, labels, n_class, perp=10):
    import networkx as nx
    g = nx.Graph()
    for edge in nx.edges(graph):
        g.add_edge(str(edge[0]), str(edge[1]))

    model = Struc2Vec(g, "varied_graph", walk_length=10, num_walks=10, dim=64, window_size=8,
                      stay_prob=0.3)
    model.train(iter=10)
    embedding_dict = model.get_embeddings()

    nodes, embeddings = [], []
    for node, embedding_vector in embedding_dict.items():
        nodes.append(node)
        embeddings.append(embedding_vector)

    tsne_res = TSNE(n_components=2, metric="euclidean", learning_rate=50.0, n_iter=2000,
                    perplexity=perp, random_state=42).fit_transform(embeddings)

    res = dict()
    for name, label_dict in labels.items():
        _labels = [label_dict[int(node)] for node in nodes]
        h, c, v, s = cluster_evaluate(embeddings, _labels, n_class)
        _res = KNN_evaluate(embeddings, _labels, cv=5, n_neighbor=4)
        res[name] = [h, c, v, s, _res['accuracy'], _res['macro f1'], _res['micro f1']]
        plot_embeddings(nodes, tsne_res, labels=_labels, n_class=n_class, save_path=
        f"E:\workspace\py\graph-embedding\\figures\struc2vec-{name}-{perp}.png")
    return res


if __name__ == '__main__':
    params = Struc2vecParameterParser()
    tab_printer(params)

    graph_name = params.graph
    if graph_name in ["europe", "usa"]:
        graph, label_dict, n_class = load_data(graph_name, label_name="SIR")
    else:
        graph, label_dict, n_class = load_data(graph_name, label_name=None)

    model = Struc2Vec(graph, graph_name, walk_length=params.walk_length, num_walks=params.walk_num,
                      stay_prob=params.stay_prob)

    run(model, label_dict, n_class, params)



