# -*- encoding:utf-8 -*-

import pandas as pd
import numpy as np
from scipy import stats
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl import Workbook


def calculate_w_distance():
    """
    根据各个节点的N维特征计算它们两两的W距离。
    调用stats.wasserstein_distance函数
    :return:
    """
    #path = u"C:\\Users\86234\Desktop\SR_guojun_katate.xls"
    path = u"C:\\Users\86234\Desktop\SR_guojun_europe.xls"
    data = pd.read_excel(path, sheet_name=0, header=None)
    row, col = data.shape
    distance = np.zeros(shape=(row, row), dtype=np.float)
    for idx1 in range(row):
        print(idx1)
        features_1 = data.loc[idx1, :]
        for idx2 in range(idx1 + 1, row):
            features_2 = data.loc[idx2, :]
            # 计算过程中会将每个元素除以总长度，当做默认权重，下面乘以len就是将权重再乘回来
            distance[idx1, idx2] = distance[idx2, idx1] = stats.wasserstein_distance(features_1, features_2) * len(features_1)

    res = pd.DataFrame(data=distance)

    # to_excel函数只能支持256列，europe数据集需要399*399, 改用openpyxl
    #res.to_excel("europe_distance.xls", encoding="utf-8", header=False, index=False, columns=None, float_format="%.9f")

    wb = Workbook()
    ws = wb.active
    for r in dataframe_to_rows(res, index=False, header=False):
        ws.append(r)
    wb.save("europe_distance.xlsx")


"""
wasserstein_distance的实现细节如下：

"""

def _cdf_distance(p, u_values, v_values, u_weights=None, v_weights=None):
    r"""
    Compute, between two one-dimensional distributions :math:`u` and
    :math:`v`, whose respective CDFs are :math:`U` and :math:`V`, the
    statistical distance that is defined as:

    .. math::

        l_p(u, v) = \left( \int_{-\infty}^{+\infty} |U-V|^p \right)^{1/p}

    p is a positive parameter; p = 1 gives the Wasserstein distance, p = 2
    gives the energy distance.

    Parameters
    ----------
    u_values, v_values : array_like
        Values observed in the (empirical) distribution.
    u_weights, v_weights : array_like, optional
        Weight for each value. If unspecified, each value is assigned the same
        weight.
        `u_weights` (resp. `v_weights`) must have the same length as
        `u_values` (resp. `v_values`). If the weight sum differs from 1, it
        must still be positive and finite so that the weights can be normalized
        to sum to 1.

    Returns
    -------
    distance : float
        The computed distance between the distributions.

    Notes
    -----
    The input distributions can be empirical, therefore coming from samples
    whose values are effectively inputs of the function, or they can be seen as
    generalized functions, in which case they are weighted sums of Dirac delta
    functions located at the specified values.

    References
    ----------
    .. [1] Bellemare, Danihelka, Dabney, Mohamed, Lakshminarayanan, Hoyer,
           Munos "The Cramer Distance as a Solution to Biased Wasserstein
           Gradients" (2017). :arXiv:`1705.10743`.
    """
    u_values, u_weights = _validate_distribution(u_values, u_weights)
    v_values, v_weights = _validate_distribution(v_values, v_weights)

    u_sorter = np.argsort(u_values)
    v_sorter = np.argsort(v_values)

    all_values = np.concatenate((u_values, v_values))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = u_values[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = v_values[v_sorter].searchsorted(all_values[:-1], 'right')

    # Calculate the CDFs of u and v using their weights, if specified.
    if u_weights is None:
        u_cdf = u_cdf_indices / u_values.size
    else:
        u_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(u_weights[u_sorter])))
        u_cdf = u_sorted_cumweights[u_cdf_indices] / u_sorted_cumweights[-1]

    if v_weights is None:
        v_cdf = v_cdf_indices / v_values.size
    else:
        v_sorted_cumweights = np.concatenate(([0],
                                              np.cumsum(v_weights[v_sorter])))
        v_cdf = v_sorted_cumweights[v_cdf_indices] / v_sorted_cumweights[-1]

    # Compute the value of the integral based on the CDFs.
    # If p = 1 or p = 2, we avoid using np.power, which introduces an overhead
    # of about 15%.
    if p == 1:
        return np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))
    if p == 2:
        return np.sqrt(np.sum(np.multiply(np.square(u_cdf - v_cdf), deltas)))
    return np.power(np.sum(np.multiply(np.power(np.abs(u_cdf - v_cdf), p),
                                       deltas)), 1/p)


def f():
    from utils.util import dataloader
    _, labels, _ = dataloader("europe", label="origin")
    a = np.zeros(shape=(len(labels), 2), dtype=int)
    i = 0
    for idx, label in labels.items():
        a[i] = [int(idx), int(label)]
        i += 1
    df = pd.DataFrame(data=a, dtype=int)
    df.to_excel("europe_origin_label.xls", encoding="utf-8", header=False, index=False, columns=None)


def save_laplacian_matrix():
    from utils.util import dataloader
    from model.GraphWave import GraphWave
    graph, _, _ = dataloader("mkarate.edgelist", directed=False, similarity=False, label="SIR")
    wave_machine = GraphWave(graph)
    L = np.asarray(wave_machine.L.toarray())
    nodes = wave_machine.nodes
    idx2node, node2idx = wave_machine.idx2node, wave_machine.node2idx
    x = np.zeros_like(L)

    """
    for idx, node in enumerate(nodes):
        for idx2 in range(idx, wave_machine.n_nodes):
            node2 = idx2node[idx2]
            x[int(node)-1, int(node2)-1] = x[int(node2)-1, int(node)-1] = L[idx, idx2]
            print(int(node), idx)
            print(int(node2), idx2)
    """
    data = pd.DataFrame(data=L)
    data.to_csv(path_or_buf="G:\pyworkspace\graph-embedding\output\\mirror_karate_laplacian1.csv", header=False, index=False)



if __name__ == '__main__':
    #calculate_w_distance()
    #save_laplacian_matrix()
    a = np.array([2, 1, 3])
    b = np.array([6, 5, 4])

    u_sorter = np.argsort(a)
    v_sorter = np.argsort(b)

    all_values = np.concatenate((a, b))
    all_values.sort(kind='mergesort')

    # Compute the differences between pairs of successive values of u and v.
    deltas = np.diff(all_values)

    # Get the respective positions of the values of u and v among the values of
    # both distributions.
    u_cdf_indices = a[u_sorter].searchsorted(all_values[:-1], 'right')
    v_cdf_indices = b[v_sorter].searchsorted(all_values[:-1], 'right')
    print(all_values)
    print(u_cdf_indices)
    print(v_cdf_indices)

    u_cdf = u_cdf_indices / 3
    v_cdf = v_cdf_indices / 3

    dist = np.sum(np.multiply(np.abs(u_cdf - v_cdf), deltas))

    print(u_cdf)
    print(v_cdf)
    print(dist * 3)