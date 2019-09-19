# -*- coding:utf-8 -*-

import sys
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import pyecharts
from pyecharts.charts import EffectScatter, Bar, Page
from pyecharts import options as opts
from pyecharts.faker import Collector, Faker
from pyecharts.globals import SymbolType
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def get_node_color(node_community):
    cnames = [item[0] for item in matplotlib.colors.cnames.iteritems()]
    node_colors = [cnames[c] for c in node_community]
    return node_colors


def plot(x_s, y_s, fig_n, x_lab, y_lab, file_save_path, title, legendLabels=None, show=False):
    plt.rcParams.update({'font.size': 16, 'font.weight': 'bold'})
    markers = ['o', '*', 'v', 'D', '<' , 's', '+', '^', '>']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    series = []
    plt.figure(fig_n)
    i = 0
    for i in range(len(x_s)):
        # n_points = len(x_s[i])
        # n_points = int(n_points/10) + random.randint(1,100)
        # x = x_s[i][::n_points]
        # y = y_s[i][::n_points]
        x = x_s[i]
        y = y_s[i]
        series.append(plt.plot(x, y, color=colors[i], linewidth=2, marker=markers[i], markersize=8))
        plt.xlabel(x_lab, fontsize=16, fontweight='bold')
        plt.ylabel(y_lab, fontsize=16, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold')
    if legendLabels:
        plt.legend([s[0] for s in series], legendLabels)
    plt.savefig(file_save_path)
    if show:
        plt.show()


def plot_ts(ts_df, plot_title, eventDates, eventLabels=None, save_file_name=None, xLabel=None, yLabel=None, show=False):
    ax = ts_df.plot(title=plot_title, marker = '*', markerfacecolor='red', markersize=10, linestyle = 'solid')
    colors = ['r', 'g', 'c', 'm', 'y', 'b', 'k']
    if not eventLabels:
        for eventDate in eventDates:
            ax.axvline(eventDate, color='r', linestyle='--', lw=2) # Show event as a red vertical line
    else:
        for idx in range(len(eventDates)):
            ax.axvline(eventDates[idx], color=colors[idx], linestyle='--', lw=2, label=eventLabels[idx]) # Show event as a red vertical line
            ax.legend()
    if xLabel:
        ax.set_xlabel(xLabel, fontweight='bold')
    if yLabel:
        ax.set_ylabel(yLabel, fontweight='bold')
    fig = ax.get_figure()
    if save_file_name:
        fig.savefig(save_file_name, bbox_inches='tight')
    if show:
        fig.show()


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

    node_pos = np.array(model.fit_transform(embeddings))

    # 没有标签
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
        markers = ['<', '8', 's', '*', 'H', 'x', 'D', '>', '^', "v", '1', '2', '3', '4', 'X', '.']
        color_idx = {}
        for idx, node in enumerate(nodes):
            color_idx.setdefault(labels[idx], [])
            color_idx[labels[idx]].append(idx)

        for c, idx in color_idx.items():
            #plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, marker=markers[int(c)%16])#, s=area[idx])
            plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c, s=50, marker=markers[int(c)%16])#, s=area[idx])

    #plt.legend()
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




def effectscatter_splitline() -> EffectScatter:
    c = (
        EffectScatter()
        .add_xaxis(Faker.choose())
        .add_yaxis("", Faker.values())
        .set_global_opts(
            title_opts=opts.TitleOpts(title="EffectScatter-显示分割线"),
            xaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
            yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
        )
    )
    return c



if __name__ == '__main__':
    a = [1, 2, 3]
    b = list(a)
    b[1] = 555
    print(a)
    print(b)
