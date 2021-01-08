# -*- encoding: utf-8 -*-

# 多尺度小波变化
# 选barbell

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from model import HSD

def func():
    barbell = nx.read_edgelist(f"../../data/graph/barbell.edgelist", create_using=nx.Graph, edgetype=float,
                               data=[('weight', float)])
    hsd = HSD(barbell, "barbell", 0, hop=15, metric="wasserstein")
    hsd.construct_hierarchy()

    origin = "0"
    layers = hsd.hierarchy[origin]
    idx2node, node2idx = hsd.nodes, hsd.node2idx
    plt.figure()
    plt.ylabel("wavelet coefficient")
    dim = 3
    scales = np.exp(np.linspace(0, np.log(100), dim * dim))
    scales[1:] = scales[:len(scales) - 1]
    scales[0] = 0
    for idx, scale in enumerate(scales):
        plt.subplot(dim, dim, idx + 1)
        wavelets = hsd.calculate_wavelets(scale, approx=False)
        xs = list(range(hsd.n_node))
        ys = [0] * hsd.n_node
        for hop in range(len(layers)):
            for neighbor in layers[hop]:
                if neighbor == '':
                    continue
                neighbor_idx = int(neighbor)
                pos = neighbor_idx if neighbor_idx >= 16 else (15 - neighbor_idx)
                ys[pos] = wavelets[node2idx[origin], node2idx[neighbor]]

        plt.bar(x=xs, height=ys, label='wavelets', color='steelblue', alpha=0.8)
        plt.xticks([])
        plt.xlabel(f"scale={round(scale, 2)}")
        plt.ylim(0, 1.0)
        if idx % 3 != 0:
            plt.yticks([])

        plt.tight_layout(pad=2)

    plt.savefig("multiscaleWavelets.svg", format="svg")
    plt.show()

if __name__ == '__main__':
    func()
