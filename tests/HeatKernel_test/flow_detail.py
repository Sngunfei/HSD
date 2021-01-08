# -*- encoding: utf-8 -*-

import networkx as nx
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import copy

from tools.hierarchy import get_hierarchical_representation
from tools.rw import save_vectors_dict
from model.multiscale_HSD import MultiHSD

from collections import defaultdict


def run():
    line = nx.read_edgelist("line.edgelist", create_using=nx.Graph, nodetype=int, edgetype=float,
                             data=[('weight', float)])
    hop = 6
    n_scales = 10000
    target_node = 0
    model = MultiHSD(line, "flow_test", hop, n_scales=n_scales)
    eigenvalues, _ = np.linalg.eigh(model.laplacian)
    model.scales = np.exp(np.linspace(np.log(0.0001), np.log(max(eigenvalues)*2), n_scales))
    #model.scales = np.linspace(0.001, max(eigenvalues)*2, n_scales)
    model.hierarchy = get_hierarchical_representation(line, maxHop=hop)

    coeffs_dict = get_node_wavelet_coefficients(model, model.scales, target_node, model.nodes)
    plot_nodes_coefficients(coeffs_dict, name="line", scales=model.scales)
    #plot_heat_flow(coeffs_dict, name="heat_flow")


# 计算各节点随着尺度变化的小波系数
# 返回结果: dict[node] = [coeff at scale0, coeff at scale1, ...]
def get_node_wavelet_coefficients(model, scales, origin, nodes):
    coeffs_dict = defaultdict(list)
    for scale in scales:
        wavelets = model.calculate_wavelets(scale, approx=False)
        origin_idx = model.node2idx[origin]
        wavelet_vector = np.squeeze(np.array(wavelets[origin_idx]))
        for node in nodes:
            node_idx = model.node2idx[node]
            coeff = wavelet_vector[node_idx]
            coeffs_dict[node].append(coeff)

    return coeffs_dict


# 画出节点随尺度变化的小波系数变化情况，line
def plot_nodes_coefficients(coeffs_dict: dict, name="tmp", scales=None):
    color_names = ['green', 'darkorange', 'aquamarine', 'black', 'blue',
                   'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
                   'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan']

    n_node, n_scale = len(coeffs_dict), len(list(coeffs_dict.values())[0])

    plt.figure()
    plt.title("wavelet coefficients change with scale")
    plt.ylim(0, 0.001)
    plt.xlabel("i-th scale")
    plt.ylabel("wavelet coefficient value")

    xs = list(range(n_scale))
    idx = 0
    for node, coeffs in coeffs_dict.items():
        pos = 0
        for pos, c in enumerate(coeffs):
            if c >= 1e-4:
                break
        print("%.6f" % scales[pos])
        #print(node, pos, scales[pos], coeffs[pos])
        plt.plot(xs, coeffs, '-', label=f"{node}", lw=1, markersize=4, color=color_names[idx], alpha=0.9)
        #plt.xticks(scales)
        idx += 1

    plt.legend()
    plt.savefig(f"{name}.png")
    plt.show()
    plt.close()

if __name__ == '__main__':
    run()
