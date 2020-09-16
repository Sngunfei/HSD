# -*- encoding: utf-8 -*-

"""
一些关于多尺度分析的处理函数
"""
from collections import defaultdict
import numpy as np

np.set_printoptions(suppress=True)
import pandas as pd
from tools.rw import save_vectors_dict

def cumulate_wavelet_coeffs(coeff_path, max_hop, target_hop) -> dict:
    """
    利用多尺度分析计算距离时，分层结构可以从高层推出低层：
    例如hop=5的小波系数，可以推导出hop={0, 1, 2, 3, 4}的小波系数，只需要逐层累加就可以了
    :param coeff_path:
    :param max_hop:
    :return target_hop:
    """
    if target_hop >= max_hop:
        print("Cumulate wavelet coefficients, target hop >= max hop, break.")
        return {}

    coeffs = defaultdict(list)
    df = pd.read_csv(coeff_path, encoding="utf8", header=None)
    n, dim = df.shape
    seg_length = max_hop + 2
    n_scales = (dim - 1) // seg_length
    for i in range(n):
        vec = df.iloc[i]
        node, coe = str(int(vec[0])), np.asarray(vec[1:])
        for cnt in range(n_scales):
            _old = coe[cnt*seg_length: (cnt+1)*seg_length]
            _new = np.append(_old[:target_hop+1], [np.sum(_old[target_hop+1:])])
            coeffs[node].extend(_new)

    new_path = coeff_path.replace("hop{}".format(max_hop), "hop{}".format(target_hop), 1)
    save_vectors_dict(coeffs, new_path)
    return coeffs


# 开题PPT里需要一张多尺度的小波系数变化图
def plotMultiScalesCoeff():
    from model.GraphWave import GraphWave
    from tools.hierarchy import get_hierarchical_representation
    import networkx as nx
    import matplotlib.pyplot as plt

    graphName = "mkarate"
    scaleNum = 200
    layerCnt = 4
    node1, node2 = "3", "20"

    graph = nx.read_edgelist(path="../data/graph/mkarate.edgelist", create_using=nx.Graph,
                             edgetype=float, data=[('weight', float)])
    hierarchy = get_hierarchical_representation(nx.DiGraph(graph), layerCnt)
    graphWave = GraphWave(graph, graphName)
    idx2node, node2idx = graphWave.idx2node, graphWave.node2idx
    eigenvalues = graphWave.get_eigenvalues()
    scales = np.exp(np.linspace(np.log(0.01), np.log(5.0), scaleNum))
    print("scales: ", scales)
    result1 = [[0] * scaleNum for _ in range(layerCnt)]
    result2 = [[0] * scaleNum for _ in range(layerCnt)]

    for idx, scale in enumerate(scales):
        coeffMat = graphWave.calculate_wavelet_coeff(scale)

        for i in range(layerCnt):
            neighbors = hierarchy[node1][i]
            curLayerSum = 0.0
            for neigh in neighbors:
                curLayerSum += coeffMat[node2idx[neigh]][node2idx[node1]]
            result1[i][idx] = curLayerSum

        for i in range(layerCnt):
            neighbors = hierarchy[node2][i]
            curLayerSum = 0.0
            for neigh in neighbors:
                curLayerSum += coeffMat[node2idx[neigh]][node2idx[node2]]
            result2[i][idx] = curLayerSum

    plt.figure()
    plt.xlabel("scale")
    plt.ylabel("wavelet coefficients sum")

    blues = ["blue", "dodgerblue", "skyblue", "green"]
    for idx in range(layerCnt):
        plt.plot(scales, result1[idx], color=blues[idx], label=f"Node1 {idx}-th layer")
        plt.plot(scales, result2[idx], color=blues[idx], linestyle='-.', label=f"Node2 {idx}-th layer")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plotMultiScalesCoeff()

