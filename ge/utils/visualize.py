# -*- coding:utf-8 -*-

"""
Visualization for embedding vectors.
"""

from sklearn.manifold import TSNE
import networkx as nx
from collections import defaultdict
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def plot_embeddings(nodes, embeddings, labels, n_class=10, node_text=False, save_path=None):
    """
    :param nodes:
    :param embeddings: 2-dimensional vectors
    :param labels:
    :param n_class:
    :param node_text:
    :return:
    """
    matplotlib.use("TkAgg")
    markers = ['o', '*', 'x', '<', '1', 'D', '>', '^', "v", 'p', '2', '3', '4', 'X', '.']
    cm = plt.get_cmap("nipy_spectral")
    cNorm  = colors.Normalize(vmin=0, vmax=n_class-1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

    class_dict = defaultdict(list)
    for idx, node in enumerate(nodes):
        class_dict[int(labels[idx])].append(idx)

    info = sorted(class_dict.items(), key=lambda item:item[0])
    for _class, _indices in info:
        plt.scatter(embeddings[_indices, 0], embeddings[_indices, 1], s=100,
                    marker=markers[_class % len(markers)],
                    c=[scalarMap.to_rgba(_class)], label=_class)

    if node_text:
        for idx, (x, y) in enumerate(embeddings):
            plt.text(x, y, nodes[idx])

    #plt.legend()
    plt.xticks([])
    plt.yticks([])
    if save_path:
        plt.savefig(save_path)
        print("Save TSNE result figure.")
    #plt.show()


def plot_embedding2D(node_pos, node_colors=None, di_graph=None, labels=None):
    node_num, embedding_dimension = node_pos.shape
    if embedding_dimension > 2:
        print("Embedding dimension greater than 2, use tSNE to reduce it to 2")
        model = TSNE(n_components=2)
        node_pos = model.fit_transform(node_pos)

    if di_graph is None:
        # plot using plt scatter
        plt.scatter(node_pos[:, 0], node_pos[:, 1], c=node_colors)
    else:
        # plot using networkx with edge structure
        pos = {}
        for i in range(node_num):
            pos[i] = node_pos[i, :]
        if node_colors is not None:
            nx.draw_networkx_nodes(di_graph, pos,
                                   node_color=node_colors,
                                   width=0.1, node_size=100,
                                   arrows=False, alpha=0.8,
                                   font_size=5, labels=labels)
        else:
            nx.draw_networkx(di_graph, pos, node_color=node_colors,
                             width=0.1, node_size=300, arrows=False,
                             alpha=0.8, font_size=12, labels=labels)


"""
def robustness_vis():
    db = Database()
    filters = {"evaluate": "LR", "metric": "l1", "ge_name": "HSELE", "data": "europe"}
    cursor = db.find("scores", filters=filters)
    LE_records = []
    for record in cursor:
        LE_records.append(record)
    filters['ge_name'] = 'HSELLE'
    cursor = db.find("scores", filters=filters)
    LLE_records = []
    for record in cursor:
        LLE_records.append(record)
    print(LE_records)
    ratio1, ratio2 = [], []
    LE_scores, LLE_scores = [], []
    for doc1, doc2 in zip(LE_records, LLE_records):
        print(doc1)
        _scores = doc1['scores']
        LE_scores.extend(_scores)
        ratio1 += [1.0 - doc1['prob']] * len(_scores)
        print(doc2)
        _scores = doc2['scores']
        LLE_scores.extend(_scores)
        ratio2 += [1.0 - doc2['prob']] * len(_scores)
    #scores = scores[::-1]
    evaluate = ["HSELE"] * len(LE_scores) + ["HSELLE"] * len(LLE_scores)
    LE_scores.extend(LLE_scores)
    ratio1.extend(ratio2)
    print(LE_scores)

    data = pd.DataFrame(data={"Accuracy": LE_scores, "Deletion Ratio": ratio1, "method": evaluate})
    sns.set(style="ticks")
    sns.relplot(x="Deletion Ratio", y="Accuracy", hue="method", data=data, kind="line")
    plt.ylim((0.6, 1))
    plt.show()


def robustness_from_excel():
    import seaborn as sns
    HSDLE=[0.738888863, 0.751388817, 0.746428551, 0.757142813, 0.787037011, 0.803703607,
           0.820370354, 0.834259237, 0.851851839, 0.870833308, 0.870238073]
    HSDLLE=[0.70208315, 0.724999867, 0.743749975, 0.774999971, 0.790476166, 0.813541638,
            0.824999978, 0.868055543, 0.881249961, 0.89999996, 0.925]
    graphwave=[0.74833333, 0.73666664, 0.748333326, 0.768333312, 0.7883333, 0.754999972,
               0.76833318, 0.79166662, 0.7933333, 0.80666664, 0.825]
    struc2vec=[0.744999852, 0.733333324, 0.746666652, 0.748333306, 0.7533333, 0.754999997,
               0.776666626, 0.789999966, 0.80999998, 0.80833332, 0.814999966]
    node2vec=[0.443333312, 0.403333318, 0.4283333, 0.451666658, 0.473333324, 0.511666652,
              0.486666646, 0.513333318, 0.489999972, 0.544999986, 0.544999972]

    delete_ratio=[0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0]

    data = pd.DataFrame(data={"Accuracy": HSDLE + HSDLLE + graphwave + struc2vec + node2vec,
                              "Deletion Ratio": delete_ratio * 5,
                              "method": ['HSDLE']*len(HSDLE) + ['HSDLLE']*len(HSDLLE) +
                                        ['GraphWave']*len(graphwave) + ['Struc2vec']*len(struc2vec) +
                                        ['Node2vec']*len(node2vec)
                              })
    sns.set(style="ticks")
    sns.relplot(x="Deletion Ratio", y="Accuracy", hue="method", data=data, kind="line")
    plt.ylim((0.0, 1))
    plt.show()


if __name__ == '__main__':
    robustness_from_excel()
    #time_vs()
"""
