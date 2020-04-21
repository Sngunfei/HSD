
from ge.utils.rw import read_vectors
from ge.utils.dataloader import load_data
from ge.utils.distance import calculate_distance
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from collections import defaultdict


def multiscales_analyse(graph_name, n_hop, n_scales, metric="hellinger"):
    graph, _, _ = load_data(graph_name, label_name="origin")
    L = nx.laplacian_matrix(graph)
    e, u = np.linalg.eigh(L.toarray())
    scales = np.exp(np.linspace(np.log(0.01), np.log(max(e)), n_scales))

    print(scales)
    print(scales[40])

    path = "../coeff/{}_hop{}_scales{}.csv".format(graph_name, n_hop, n_scales)
    vectors = read_vectors(path)
    n_node = len(vectors)
    seg_length = len(vectors["1"]) // n_scales

    # 要对比各尺度下的距离，需要事先挑出几对点
    # structrally equivalent cases.
    # mkarate: 1, 37. barbell: 1, 16, europe: 321, 396. usa:1168, 1185
    node1, node2 = "1", "16"
    v1, v2 = vectors[node1], vectors[node2]
    equ_dists = cal_distance(v1, v2, n_scales, seg_length, metric)

    # structurally similar
    # mkarate: 1, 34. barbell: 2, 3. europe: 321, 366, usa: 1168, 1085
    node3, node4 = "2", "4"
    v3, v4 = vectors[node3], vectors[node4]
    sim_dists = cal_distance(v3, v4, n_scales, seg_length, metric)

    # structurally different
    # mkarate: 3, 20. barbell: 0, 15, europe: 321, 20, usa: 1168, 131
    node5, node6 = "0", "15"
    v5, v6 = vectors[node5], vectors[node6]
    diff_dists = cal_distance(v5, v6, n_scales, seg_length, metric)

    xs = [i+1 for i in range(n_scales)]
    max_d = max(max(sim_dists), max(diff_dists))

    plt.figure()

    plt.subplot(131)
    plt.plot(xs, equ_dists)
    plt.ylim(0, max_d * 1.5)
    plt.xlabel("scale")
    plt.ylabel("structural distance")
    plt.title("Strucutrally equivalent nodes: ({}, {})".format(node1, node2))

    plt.subplot(132)
    plt.plot(xs, sim_dists)
    plt.ylim(0, max_d * 1.5)
    plt.xlabel("scale")
    plt.title("Strucutrally similar nodes: ({}, {})".format(node3, node4))


    plt.subplot(133)
    plt.plot(xs, diff_dists)
    plt.ylim(0, max_d * 1.5)
    plt.xlabel("scale")
    plt.title("Strucutrally different nodes: ({}, {})".format(node5, node6))

    plt.show()

def cal_distance(v1, v2, n_scales, seg_length, metric):
    dists = []
    for i in range(n_scales):
        p = v1[i * seg_length: (i + 1) * seg_length]
        q = v2[i * seg_length: (i + 1) * seg_length]
        dists.append(calculate_distance(p, q, metric))
    return dists


def vector2dict(vectors: dict, n_scales: int) -> dict:
    coeff_dict = defaultdict(list)
    seg_length = len(vectors['1']) // n_scales
    for node, v in vectors.items():
        for i in range(n_scales):
            p = v[i*seg_length : (i+1)*seg_length]
            coeff_dict[node].append(p)
    return coeff_dict


def plot_coeff(graph_name, n_hop, n_scales, metric='hellinger'):
    path = "../coeff/{}_hop{}_scales{}.csv".format(graph_name, n_hop, n_scales)
    vectors = read_vectors(path)
    seg_length = len(vectors['1']) // n_scales
    coeff_dict = vector2dict(vectors, n_scales)

    # mkarate: 3, 20
    node1, node2 = '3', '20'
    v1, v2 = vectors[node1], vectors[node2]
    coeff1, coeff2 = coeff_dict[node1], coeff_dict[node2]

    k = 5
    count = k * k
    step = n_scales // count
    plt.figure()
    xs = [i for i in range(seg_length)]
    for i in range(count):
        idx = i * step
        p, q = coeff1[idx], coeff2[idx]
        d = calculate_distance(p, q, metric)
        plt.subplot(k, k, i+1)
        plt.ylim(0, 1.0)
        plt.plot(xs, p, "r")
        plt.plot(xs, q, "g")
        plt.plot(xs, [d]*seg_length, 'b')
    plt.show()


if __name__ == '__main__':
    a = [[2, 2, 2], [2, 2, -2], [2, -2, 2]]
    print(np.linalg.eigvals(a))

    assert False
    graph_name = "barbell"
    n_hop = 4
    n_scales = 100
    #metric = "wasserstein"
    metric = "hellinger"
    #multiscales_analyse(graph_name, n_hop, n_scales, metric)
    #plot_coeff(graph_name, n_hop, n_scales)

    a = 0.04904671223434214
    b = 19.010704066076674
    from ge.model.GraphWave import scale_boundary

    scales = np.exp(np.linspace(np.log(0.01), np.log(19.010704066076674), 100))
    print(scales[20], scales[40])
    print(scale_boundary(a, b))


