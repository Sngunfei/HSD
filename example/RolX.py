
from example.parser import RolxParameterParser, tab_printer
from ge.model.RolX.rolx import RolX
from ge.utils.dataloader import load_data
from ge.evaluate.evaluate import LR_evaluate, KNN_evaluate
from ge.utils.rw import save_results
import pandas as pd
from sklearn.manifold import TSNE
from ge.utils.visualize import plot_embeddings


def run(model, label_dict, n_class, params):
    embedding_dict = model.train()
    nodes, labels, embeddings = [], [], []
    for node, embedding_vector in embedding_dict.items():
        nodes.append(node)
        labels.append(label_dict[node])
        embeddings.append(embedding_vector)

    if model.graph_name in ['europe', 'usa']:
        lr_res = LR_evaluate(embeddings, labels)
        save_results(lr_res, "../results/lr/rolx_{}.txt".format(graph_name))
        knn_res = KNN_evaluate(embeddings, labels)
        save_results(knn_res, "../results/knn/rolx_{}.txt".format(graph_name))

    df = pd.DataFrame(data=embeddings, index=nodes, columns=None, dtype=float)
    df.to_csv("../embeddings/rolx_{}.csv".format(model.graph_name), header=False, float_format="%.8f")

    tsne_res = TSNE(n_components=2, metric="euclidean", learning_rate=50.0, n_iter=2000,
                    perplexity=params.tsne, random_state=params.random).fit_transform(embeddings)

    df = pd.DataFrame(data=tsne_res, index=nodes, columns=None, dtype=float)
    df.to_csv("../tsne_results/rolx_{}.csv".format(model.graph_name))

    figure_path = "../figures/rolx_{}.png".format(model.graph_name)
    plot_embeddings(nodes, tsne_res, labels=labels, n_class=n_class, node_text=False, save_path=figure_path)


if __name__ == '__main__':
    params = RolxParameterParser()
    tab_printer(params)

    graph_name = params.graph
    if graph_name in ["barbell", "mkarate"]:
        graph, label_dict, n_class = load_data(graph_name, label_name="origin")
    else:  # europe, usa
        graph, label_dict, n_class = load_data(graph_name, label_name="SIR")

    model = RolX(graph, graph_name, params.dim, params)
    run(model, label_dict, n_class, params)