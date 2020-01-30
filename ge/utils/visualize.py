# -*- coding:utf-8 -*-


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd

#from pyecharts.charts import Bar
#from pyecharts import options as opts
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

from ge.utils.util import dataloader, save_vectors
from ge.utils.db import Database


def plot_embeddings(nodes, embeddings, labels=None, n_class=10, node_text=False, method="pca", init='random', random_state=42, perplexity=5):
    """
    降维可视化计算得到的嵌入向量
    :param nodes: 节点名称
    :param embeddings: 对应的嵌入向量
    :param labels: 节点的标签信息
    :param method: 降维方法，PCA or TSNE
    :param random_state: 随机种子
    :param perplexity:　困惑度，用于TSNE方法中。
    :return:
    """
    matplotlib.use('TkAgg')
    if method not in ['pca', 'tsne']:
        raise NotImplementedError("The visualize method {} is not implemented.".format(method))

    embeddings = np.array(embeddings)
    n, d = embeddings.shape

    if d > 2:
        if method == 'pca':
            model = PCA(n_components=2, whiten=True, random_state=random_state)
        else:
            """
            perplexity是用来刻画近邻点数量的，如果近邻点多，那么就设置大一点，否则就设置小一点。
            在bell数据集中，有稀疏对称点，数量只有一对，还有稠密对称点，数量有四五对，
            有稀疏又有稠密，所以在bell中无法找到一个很好的值来可视化，一般是1-5。
            """
            model = TSNE(n_components=2,  random_state=random_state, n_iter=1000, perplexity=perplexity, init=init)

        _2d_data = np.array(model.fit_transform(embeddings))
    else:
        _2d_data = embeddings

    if not labels:
        plt.scatter(x=_2d_data[:, 0], y=_2d_data[:, 1], s=40, marker='o')
    else:
        from collections import defaultdict
        import matplotlib.colors as colors
        import matplotlib.cm as cmx

        markers = ['o', '*', 'x', '<', '1', 'x', 'D', '>', '^', "v", '1', '2', '3', '4', 'X', '.']

        cm = plt.get_cmap("nipy_spectral")
        cNorm  = colors.Normalize(vmin=0, vmax=n_class-1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

        class_dict = defaultdict(list)
        for idx, node in enumerate(nodes):
            class_dict[int(labels[idx])].append(idx)

        info = sorted(class_dict.items(), key=lambda item:item[0])
        for _class, _indices in info:
            # general case， n_class < 10
            #plt.scatter(_2d_data[_indices, 0], _2d_data[_indices, 1], s=40, marker=markers[_class], cmap=plt.get_cmap("nipy_spectral"))
            # mirror karate network, n_class = 34
            plt.scatter(_2d_data[_indices, 0], _2d_data[_indices, 1], s=100, marker=markers[_class % len(markers)], c=[scalarMap.to_rgba(_class)], label=_class)

        if node_text:
            for idx, (x, y) in enumerate(_2d_data):
                plt.text(x, y, nodes[idx])

    #plt.legend()

    plt.xticks([])
    plt.yticks([])
    plt.show()
    return _2d_data


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
对度数的研究。考虑1，2,3阶。最好能够很好的展示出来。
"""
def flight_data_analyze(flights=None):

    if not flights:
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

    color = ['r', 'g', 'b', 'y', 'm', 'k']
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
    plt.show()
    plt.savefig("../../image/flight.png")


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

"""
def effectscatter_splitline() -> Bar:
    c = (
        Bar()
            .add_xaxis(["Bell", "Karate", "Subway", "Brazil Flight", "Europe Flight", "USA Flight"])
            .add_yaxis("LLE+HSE", [1, 0.93, 0.89, 0.81, 0.94, 0])
            .add_yaxis("Node2vec", [0.86, 0.43, 0.54,  0.56, 0.54, 0.52])
            .add_yaxis("Struc2vec", [1, 0.93, 0.61, 0.85, 0.94, 1])
            .add_yaxis("GraphWave", [1, 0.86, 0.80, 0.19, 0.14, 0])
            .set_global_opts(title_opts=opts.TitleOpts(title="LR Classification."))
    )
    return c
"""


def heat_map(embeddings, labels):
    """
    画各类中心的heat_map
    :param embeddings:
    :param labels:
    :return:
    """
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    centers = dict()
    labelset = np.unique(labels)
    n_class = len(labelset)
    for label in labelset:
        centers[label] = np.mean(embeddings[labels==label], axis=0)

    sns.set()
    matplotlib.use('TkAgg')
    dis = np.zeros(shape=(n_class, n_class))
    for i, label in enumerate(labelset):
        center1 = centers[label]
        for j in range(i+1, n_class):
            center2 = centers[labelset[j]]
            dis[i, j] = dis[j, i] = np.sum(np.square(center1 - center2))
    sns.heatmap(dis, annot=True, cmap="YlGnBu")
    plt.show()


def robustness_knn():
    db = Database()
    filters = {"evaluate": "LR", "metric": "l1", "ge_name": "HSELE", "data": "europe"}
    cursor = db.find("scores", filters=filters)
    LR_records = []
    for record in cursor:
        LR_records.append(record)
    filters['evaluate'] = 'KNN'
    cursor = db.find("scores", filters=filters)
    KNN_records = []
    for record in cursor:
        KNN_records.append(record)
    ratio1, ratio2 = [], []
    LR_scores, KNN_scores = [], []
    for doc1, doc2 in zip(LR_records, KNN_records):
        print(doc1)
        _scores = doc1['scores']
        LR_scores.extend(_scores)
        ratio1 += [1.0 - doc1['prob']] * len(_scores)
        print(doc2)
        _scores = doc2['scores']
        KNN_scores.extend(_scores)
        ratio2 += [1.0 - doc2['prob']] * len(_scores)
    #scores = scores[::-1]
    evaluate = ["LR"] * len(LR_scores) + ["KNN"] * len(KNN_scores)
    LR_scores.extend(KNN_scores)
    ratio1.extend(ratio2)

    data = pd.DataFrame(data={"Accuracy score": LR_scores, "Deletion Ratio": ratio1, "method": evaluate})
    sns.set(style="ticks")
    sns.relplot(x="Deletion Ratio", y="Accuracy score", hue="method", data=data, kind="line")
    plt.ylim((0.6, 1))
    plt.show()


def time_vs():
    n_nodes = [31, 68, 277, 173, 131, 399]
    times = [0.17806, 0.8108, 11.4251, 5.8922, 2.66540, 51.99065]

    data = pd.DataFrame(data={"Time": times, "Number of Nodes": n_nodes})
    sns.set(style="ticks")
    sns.relplot(x="Number of Nodes", y="Time", data=data, kind="line")
    plt.ylabel("Time  /  s")
    #plt.ylim((0.6, 1))
    plt.show()


def robustness_vis():
    ratio = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    hsele_accuracy = [0.86666665, 0.860833309, 0.848333321, 0.826666643, 0.817499986, 0.794999906,
                      0.781666643, 0.744999967, 0.729999976, 0.71833328, 0.687499968]

    hselle_accuracy = [0.925, 0.888333313, 0.875833299, 0.846666646, 0.807499979, 0.80583331,
                       0.777499976, 0.75999997, 0.716666, 0.69666658, 0.63583322]

    plt.scatter(ratio, hsele_accuracy, marker="*", s=60, c='r', label="HSELE")
    plt.scatter(ratio, hselle_accuracy, marker="o", s=60, c='b', label="HSELLE")
    plt.show()


def robustness():
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
    HSDLE=[0.687499968, 0.71833328, 0.729999976, 0.744999967, 0.781666643, 0.794999906,
           0.817499986, 0.826666643, 0.848333321, 0.860833309, 0.86666665]
    HSDLLE=[0.65583322, 0.69666658, 0.7166666, 0.75999997, 0.777499976, 0.80583331,
            0.807499979, 0.846666646, 0.875833299, 0.888333313, 0.925]
    delete_ratio=[0.5, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.0]

    data = pd.DataFrame(data={"Accuracy": HSDLE + HSDLLE, "Deletion Ratio": delete_ratio + delete_ratio,
                              "method": ['HSDLE']*len(HSDLE) + ['HSDLLE']*len(HSDLLE)})
    sns.set(style="ticks")
    sns.relplot(x="Deletion Ratio", y="Accuracy", hue="method", data=data, kind="line")
    plt.ylim((0.0, 1))
    plt.show()


if __name__ == '__main__':
    robustness_from_excel()
    #time_vs()