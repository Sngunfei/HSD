# -*- coding:utf-8 -*-

"""
Visualization for embedding vectors.
"""

from collections import defaultdict

from sklearn.metrics.pairwise import euclidean_distances
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


def plot_2D_points(nodes: list, points: np.ndarray, labels: list):
    category_dict = defaultdict(list)
    for idx in range(len(labels)):
        category_dict[int(labels[idx])].append(idx)

    cm = plt.get_cmap("nipy_spectral")
    cNorm = colors.Normalize(vmin=0, vmax=len(category_dict))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    plt.figure()
    markers = ['o', '*', 'x', '<', '1', 'p', 'D', '>', '^', 'P', 'X', "v", "s", "+", "d"]

    point_set_list = sorted(category_dict.items(), key=lambda item: item[0])
    for category, point_indices in point_set_list:
        plt.scatter(points[point_indices, 0], points[point_indices, 1],
                    #s=100,
                    marker=markers[category % len(markers)],
                    c=[scalarMap.to_rgba(category)],
                    label=category)

        for label, node_idxs in category_dict.items():
            idx = node_idxs[0]
            x, y = points[idx, 0], points[idx, 1]
            string = ",".join([nodes[idx] for idx in node_idxs])
            plt.text(x, y, s=string)

    #plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 2D图，karate，没有结构性标签，直接打出node标号
def plot_node_str(nodes: list, points: np.ndarray):
    plt.figure()
    for idx, node in enumerate(nodes):
        x, y = points[idx, 0], points[idx, 1]
        plt.scatter([x], [y])
        plt.text(x, y, s=node)

    plt.xticks([])
    plt.yticks([])
    plt.show()


# 关系图
def relation_plot(vectors: np.ndarray, labels: list):
    n = len(vectors)
    n_class = len(set(labels))
    corr = np.zeros(shape=(n_class, n_class), dtype=np.float)
    count_matrix = np.zeros(shape=(n_class, n_class))
    distance_matrix = euclidean_distances(vectors, vectors)

    for idx1 in range(n):
        vector1, label1 = vectors[idx1], labels[idx1] - 1
        for idx2 in range(idx1 + 1, n):
            vector2, label2 = vectors[idx2], labels[idx2] - 1
            # 多种metric
            #distance = math.sqrt(np.linalg.norm(vector1 - vector2) / len(vector1))
            #distance = cosine_distances(vector1, vector2)
            distance = distance_matrix[idx1][idx2]
            corr[label1][label2] = (corr[label1][label2] * count_matrix[label1][label2] + distance) / (count_matrix[label1][label2] + 1)
            corr[label2][label1] = corr[label1][label2]
            count_matrix[label1][label2] += 1
            count_matrix[label2][label1] = count_matrix[label1][label2]

    #mask = np.zeros_like(corr)
    #mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(n_class, n_class))
        ax = sns.heatmap(corr, mask=None, vmax=np.max(corr), square=True,
                         cmap="YlGnBu")  #, linewidths=.3 , annot=True, fmt=".3f")
        plt.show()


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
