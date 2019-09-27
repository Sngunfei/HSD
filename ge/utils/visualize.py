# -*- coding:utf-8 -*-


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd

from pyecharts.charts import  Bar
from pyecharts import options as opts
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

from ge.utils.util import dataloader
from ge.utils.db import Database


def plot_embeddings(nodes, embeddings, labels=None, method="pca", random_state=42, perplexity=5):
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

    if method == 'pca':
        model = PCA(n_components=2, whiten=True, random_state=random_state)
    else:
        """
        perplexity是用来刻画近邻点数量的，如果近邻点多，那么就设置大一点，否则就设置小一点。
        在bell数据集中，有稀疏对称点，数量只有一对，还有稠密对称点，数量有四五对，
        有稀疏又有稠密，所以在bell中无法找到一个很好的值来可视化，一般是1-5。
        """
        model = TSNE(n_components=2,  random_state=random_state, n_iter=5000, perplexity=perplexity, init="pca")
    embeddings = np.array(embeddings)
    node_pos = np.array(model.fit_transform(embeddings))

    # 没有标签
    if not labels:
        dic = dict()
        for i in range(len(node_pos)):
            x, y = node_pos[i, 0], node_pos[i, 1]
            if not labels:
                plt.scatter(x, y, s=88)
            x = round(x, 2)
            y = round(y, 2)
            dic["{} {}".format(x, y)] = dic.get("{} {}".format(x, y), []) + [nodes[i]]
        for k, v in dic.items():
            x, y = k.split(" ")
            #x = max(float(x) - len(v) * 0.02,-2)
            plt.text(float(x), float(y), " ".join(v))

    if labels:
        # 带标签
        markers = ['<', '*', 'x', 'D', 'H', 'x', 'D', '>', '^', "v", '1', '2', '3', '4', 'X', '.']
        color_idx = {}
        for idx, node in enumerate(nodes):
            color_idx.setdefault(labels[idx], [])
            color_idx[labels[idx]].append(idx)
        for c, idx in color_idx.items():
            #plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, marker=markers[int(c)%16])#, s=area[idx])
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=50, marker=markers[int(c)])#, s=area[idx])

    plt.legend()
    plt.show()


def plot_subway_embedding(nodes=None, embeddings=None, labels=None, perplexity=5):
    model = TSNE(n_components=2, random_state=42, n_iter=10000, perplexity=perplexity, init="pca")
    node_pos = model.fit_transform(embeddings)

    markers = ['+', 'o', '<', '*', 'D', 'x', 'H', '>', '^', "v", '1', '2', '3', '4', 'X', '.']
    color_idx = {}
    for i in range(len(nodes)):
        color_idx.setdefault(labels[i], [])
        color_idx[labels[i]].append(i)

    for c, idx in color_idx.items():
        #plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, marker=markers[int(c)%16])#, s=area[idx])\
        c = int(c)
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=(c-1)%3, s=30, marker=markers[(c-1) // 3 + 1])#, s=area[idx])

    """
    for i in range(len(nodes)):
        plt.text(node_pos[i, 0], node_pos[i, 1], nodes[i])
    """
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()


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


if __name__ == '__main__':
    #robustness_knn()
    time_vs()