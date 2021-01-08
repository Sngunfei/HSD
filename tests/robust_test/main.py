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


def get_truncated_graphs():
    deleted_edges = [[(2, 5)],
                     [(3, 6), (3, 7)],
                     [(4, 8), (4, 9), (4, 10)],
                     [(2, 5), (4, 8), (4, 9), (4, 10)],
                     [(3, 6), (3, 7), (4, 8), (4, 9), (4, 10)],
                     [(2, 5), (3, 6), (3, 7), (4, 8), (4, 9), (4, 10)],
                     [(0, 1)],
                     [(0, 2)],
                     [(0, 3)],
                     [(0, 4)],
                     [(0, 1), (0, 2)],
                     [(0, 1), (0, 3)],
                     [(0, 1), (0, 4)],
                     [(0, 2), (0, 3)],
                     [(0, 2), (0, 4)],
                     [(0, 3), (0, 4)],
                     [(0, 2), (0, 3), (0, 4)],
                     [(0, 1), (0, 3), (0, 4)],
                     [(0, 1), (0, 2), (0, 3), (0, 4)]]

    original_graph = nx.read_edgelist("graph.edgelist", create_using=nx.Graph, nodetype=int, edgetype=float, data=[('weight', float)])
    graphs = [original_graph]
    for edges in deleted_edges:
        truncated_graph = nx.Graph(original_graph)
        truncated_graph.remove_edges_from(edges)
        graphs.append(truncated_graph)

    return graphs


def get_variated_graphs():
    original_graph = nx.Graph(nx.read_edgelist("graph.edgelist", create_using=nx.Graph,
                                               nodetype=int, edgetype=float, data=[('weight', float)]))
    original_graph.add_edge(5, 6)

    graphs = [original_graph]
    deleted_edges = [[(0, 2)],
                     [(0, 3)]]

    for edges in deleted_edges:
        truncated_graph = nx.Graph(original_graph)
        truncated_graph.remove_edges_from(edges)
        graphs.append(truncated_graph)

    return graphs


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
def plot_nodes_coefficients(coeffs_dict: dict, name="tmp"):
    color_names = ['green', 'darkorange', 'aquamarine', 'black', 'blue',
                   'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
                   'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan']

    n_node, n_scale = len(coeffs_dict), len(list(coeffs_dict.values())[0])

    label_x_pos = [65, 60, 60, 60, 60, 65, 75, 65, 65, 70, 75]

    plt.figure()
    plt.title("wavelet coefficients change with scale")
    plt.ylim(0, 0.35)
    plt.xlabel("i-th scale")
    plt.ylabel("wavelet coefficient value")
    xs = list(range(n_scale))
    idx = 0
    for node, coeffs in coeffs_dict.items():
        if node not in [0, 1, 2, 3, 4]:
            continue
        plt.plot(xs, coeffs, '-', label=f"{node}", lw=1, markersize=4, color=color_names[idx], alpha=0.9)
        plt.text(label_x_pos[idx], coeffs[label_x_pos[idx]], f"{node}", size=8, ha='left', va='center')
        idx += 1

    plt.legend()
    plt.savefig(f"coeffs/{name}.png")
    plt.close()


def plot_heat_flow(coeffs_dict: dict, name="tmp"):
    NODE_SETS = {
        0: {0},
        1: {1},
        2: {2, 5},
        3: {3, 6, 7},
        4: {4, 8, 9, 10},
        5: {5},
        6: {6},
        7: {7},
        8: {8},
        9: {9},
        10: {10},
    }
    color_names = ['green', 'darkorange', 'aquamarine', 'coral', 'blue', 'brown', 'burlywood', 'cadetblue']
    n_node, n_scale = len(coeffs_dict), len(list(coeffs_dict.values())[0])
    label_x_pos = [75, 70, 75, 65, 70, 75]

    fig = plt.figure()
    #ax = fig.add_subplot(111)

    #ax.spines['top'].set_color('none')
    #ax.spines['right'].set_color('none')
    #ax.xaxis.set_ticks_position('bottom')
    #ax.spines['bottom'].set_position(('data', 0))
    #ax.xaxis.set_ticks_position('left')
    #ax.spines['left'].set_position(('data', 0))

    plt.title("Heat flow")
    plt.ylim(-0.03, 0.01)
    #plt.ylim(0, 0.01)
    plt.xlim(1, n_scale)
    plt.xlabel("i-th scale")
    plt.ylabel("wavelet coefficient value")
    xs = list(range(n_scale))
    idx = 0

    for node, node_group in NODE_SETS.items():
        if node in [5, 6, 7, 8, 9, 10]:
            continue
        coeffs_sum = np.array([0] * n_scale)
        for vertex in node_group:
            coeffs = coeffs_dict.get(vertex)
            coeffs_sum = np.add(coeffs_sum, coeffs)

        flow = coeffs_sum - np.insert(coeffs_sum, 0, 1.0)[:-1]
        print(len(xs), len(flow))
        plt.plot(xs, flow, '-', label=f"{node}", color=color_names[idx])
        plt.text(label_x_pos[idx], flow[label_x_pos[idx]], f"{node}", size=8, ha='left', va='center')
        idx += 1

    plt.plot(xs, [0.0] * n_scale, '-', lw=1, markersize=4, color='black', alpha=0.9)

    plt.legend()
    plt.savefig(f"heat_flow/{name}_hop2.png")
    plt.close()


def process(graphs: list):
    hop = 2
    n_scales = 100
    target_node = 0
    plt.figure()
    for pos, graph in enumerate(graphs):
        model = MultiHSD(graph, "robust_test", hop, n_scales=n_scales)
        eigenvalues, _ = np.linalg.eigh(model.laplacian)
        print(max(eigenvalues))
        model.scales = np.exp(np.linspace(np.log(0.001), np.log(max(eigenvalues)), n_scales))
        #model.scales = np.linspace(0.001, max(eigenvalues), n_scales)
        # print(scales)
        model.hierarchy = get_hierarchical_representation(graph, maxHop=hop)

        #model.scales = [0.01,0.01066347,0.01137095,0.01212538,0.01292986,0.01378771,0.01470248,0.01567794,0.01671812,0.01782731,0.01901009,0.02027135,0.02161629,0.02305045,0.02457978,0.02621056,0.02794955,0.02980391,0.0317813,0.03388988,0.03613836,0.03853602,0.04109276,0.04381913,0.04672638,0.04982652,0.05313235,0.0566575,0.06041654,0.06442498,0.06869936,0.07325733,0.07811772,0.08330057,0.08882728,0.09472068,0.10100508,0.10770644,0.1148524,0.12247248,0.13059812,0.13926287,0.1485025,0.15835515,0.16886149,0.18006489,0.1920116,0.20475093,0.21833548,0.23282131,0.24826823,0.26474001,0.28230463,0.3010346,0.32100725,0.34230502,0.36501582,0.38923341,0.41505776,0.44259547,0.47196021,0.5032732,0.53666371,0.57226957,0.61023776,0.65072501,0.69389846,0.73993632,0.78902864,0.84137807,0.89720072,0.95672701,1.02020268,1.08788974,1.16006762,1.23703426,1.31910738,1.40662579,1.49995074,1.5994675,1.70558687,1.8187469,1.93941473,2.06808847,2.20529928,2.35161358,2.50763534,2.67400863,2.85142024,3.04060252,3.24233642,3.45745469,3.68684534,3.93145531,4.19229434,4.47043917,4.76703798,5.08331515,5.42057626,5.78021352]

        coeffs_dict = get_node_wavelet_coefficients(model, model.scales, target_node, model.nodes)
        #plot_nodes_coefficients(coeffs_dict, name="tmp")
        plot_heat_flow(coeffs_dict, name="heat_flow")
        continue
        embedding_dict = model.embed()
        save_vectors_dict(embedding_dict, "robust.csv")
        target_vector = embedding_dict[target_node]
        # 数据格式：[sum, mean, var, ...], 每个scale占 (hop+1) * 3, 总向量长度为 (hop+1) * 3 * n_scales
        width = (hop + 1) * 3
        length = width * n_scales
        assert length == len(target_vector), f"error: (hop+1) * 3 * n_scales {(hop+1) * 3 * n_scales} != len(embedding vector) {len(target_vector)}"

        # info: [1:hop[], [], [], ]
        indices = [idx * width for idx in range(0, n_scales)]
        description = [defaultdict(list) for _ in range(hop+1)]
        for start in indices:
            for _hop in range(hop+1):
                _start = start + _hop * 3
                description[_hop][0].append(target_vector[_start]) # sum
                description[_hop][1].append(target_vector[_start+1]) # mean
                description[_hop][2].append(target_vector[_start+2])

        xs = list(range(n_scales))
        #plt.subplot(4, 5, pos+1)
        plt.figure()
        plt.plot(xs, description[0][0], '-',label="sum",lw=2, markersize=4, color="blue")
        plt.plot(xs, description[0][1], '-.',label="mean",lw=2,markersize=4, color="blue")
        plt.plot(xs, description[0][2], '--',label="var",lw=2,markersize=4, color="blue")

        plt.plot(xs, description[1][0], '-', label="1-hop sum", lw=2, markersize=4, color="red")
        plt.plot(xs, description[1][1], '-.', label="1-hop mean", lw=2, markersize=4, color="red")
        plt.plot(xs, description[1][2], '--', label="1-hop var", lw=2, markersize=4, color="red")

        plt.plot(xs, description[2][0], '-', label="2-hop sum", lw=2, markersize=4, color="green")
        plt.plot(xs, description[2][1], '-.', label="2-hop mean", lw=2, markersize=4, color="green")
        plt.plot(xs, description[2][2], '--', label="2-hop var", lw=2, markersize=4, color="green")

        description = None
        plt.legend()
        plt.savefig(f"figures2/{pos}.png")


def f(graph: nx.Graph):
    es1, _ = np.linalg.eigh(nx.laplacian_matrix(graph).todense())
    es2, _ = np.linalg.eigh(nx.normalized_laplacian_matrix(graph).todense())
    print(es1)
    print(es2)
    return

if __name__ == '__main__':
    graphs = get_truncated_graphs()
    #graphs = get_variated_graphs()
    #f(graphs[0])
    process([graphs[0]])