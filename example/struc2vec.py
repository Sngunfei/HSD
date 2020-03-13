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
from ge.evaluate.evaluate import LR_evaluate, KNN_evaluate
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

    if model.graph_name in ['europe', 'usa']:
        lr_res = LR_evaluate(embeddings, labels)
        lr_res['walk_length'] = model.walk_length
        lr_res['walk_num'] = model.walk_num
        lr_res['stay_prob'] = model.stay_prob
        save_results(lr_res, "../results/lr/struc2vec_{}.txt".format(graph_name))
        knn_res = KNN_evaluate(embeddings, labels)
        knn_res['walk_length'] = model.walk_length
        knn_res['walk_num'] = model.walk_num
        knn_res['stay_prob'] = model.stay_prob
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


if __name__ == '__main__':
    params = Struc2vecParameterParser()
    tab_printer(params)

    graph_name = params.graph
    if graph_name in ["barbell", "mkarate"]:
        graph, label_dict, n_class = load_data(graph_name, label_name="origin")
    else:  # europe, usa
        graph, label_dict, n_class = load_data(graph_name, label_name="SIR")

    model = Struc2Vec(graph, graph_name, walk_length=params.walk_length, num_walks=params.walk_num,
                      stay_prob=params.stay_prob)

    run(model, label_dict, n_class, params)



