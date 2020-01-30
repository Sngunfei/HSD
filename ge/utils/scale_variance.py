# -*- encoding: utf-8 -*-

"""
分析小波系数的方差
"""
from ge.model.GraphWave import GraphWave, scale_boundary
import matplotlib.pyplot as plt
from collections import defaultdict
from ge.utils.util import dataloader
import numpy as np


def delta_a(wavelet_coeff, N):
    """
    |\delta_aa - 1/N|
    :param wavelet_coeff:
    :param N:
    :return:
    """
    return abs(wavelet_coeff - 1.0 / N)


def var(wavelet_coeff_s, wavelet_coeff_2s, N):
    """
    计算var， 忽略常系数
    :param wavelet_coeff_s:
    :param wavelet_coeff_2s:
    :return:
    """
    delta_0 = delta_a(1.0, N)
    delta_s = delta_a(wavelet_coeff_s, N)
    delta_2s = delta_a(wavelet_coeff_2s, N)

    return delta_0 * delta_2s - delta_s * delta_s



def analyse(graph, scales):
    """
    分析图中某个节点a处的，其他位置上小波系数的方差变化趋势
    :return:
    """
    wave_machine = GraphWave(graph)
    eigenvalues = wave_machine._e
    sMin, sMax = scale_boundary(eigenvalues[1], eigenvalues[-1])
    scale = (sMin + sMax) / 2  # 根据GraphWave论文中推荐的尺度进行设置。
    print("recommend scale: ", scale)
    N = wave_machine.n_nodes
    vars = defaultdict(list)
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    var_all = []
    for scale in scales:
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        coeffs_2 = wave_machine.cal_all_wavelet_coeffs(scale * 2.0)
        tmp = 0.0
        for a in nodes:
            vars[a].append(var(coeffs[a][a], coeffs_2[a][a], N))
            tmp += vars[a][-1]
        var_all.append(tmp)


    for idx, node in enumerate(nodes):
        plt.subplot(4, 4, idx+1)
        plt.plot(scales, vars[node], c='r', label=node)
        plt.plot([sMin] * 50, [i*0.002 for i in range(50)], c='b', label="scale_min")
        plt.plot([sMax] * 50, [i*0.002 for i in range(50)], c='k', label="scale_max")
        plt.legend()
    #plt.show()

    plt.figure()
    plt.plot(scales, var_all)
    plt.plot([sMin] * 60, [i * 0.02 for i in range(60)], c='b', label="scale_min")
    plt.plot([sMax] * 60, [i * 0.02 for i in range(60)], c='k', label="scale_max")
    plt.show()


def calc_entropy(arr):
    res = 0.0
    for num in arr:
        res += -num * np.log2(num)
    return res

def entropy(graph, scales):
    """
    分析图中某个节点a处的，其他位置上小波系数的方差变化趋势
    :return:
    """
    wave_machine = GraphWave(graph)
    eigenvalues = wave_machine._e
    sMin, sMax = scale_boundary(eigenvalues[1], eigenvalues[-1])
    scale = (sMin + sMax) / 2  # 根据GraphWave论文中推荐的尺度进行设置。
    print("recommend scale: ", scale)
    N = wave_machine.n_nodes
    entros = defaultdict(list)
    nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    entro_all = []
    for scale in scales:
        coeffs = wave_machine.cal_all_wavelet_coeffs(scale)
        tmp = 0.0
        for a in nodes:
            entros[a].append(calc_entropy(coeffs[a]))
            tmp += entros[a][-1]
        entro_all.append(tmp)

    for idx, node in enumerate(nodes):
        plt.subplot(4, 4, idx+1)
        plt.plot(scales, entros[node], c='r', label=node)
        plt.plot([sMin] * 30, [i*0.2 for i in range(30)], c='b', label="scale_min")
        plt.plot([sMax] * 30, [i*0.2 for i in range(30)], c='k', label="scale_max")
        plt.legend()
    #plt.show()

    plt.figure()
    plt.plot(scales, entro_all)
    plt.plot([sMin] * 60, [i * 0.2 for i in range(60)], c='b', label="scale_min")
    plt.plot([sMax] * 60, [i * 0.2 for i in range(60)], c='k', label="scale_max")
    plt.show()


if __name__ == '__main__':
    graph, _, _ = dataloader(name="mkarate.edgelist")
    scales = [i*0.01 for i in range(0, 2000)]
    #analyse(graph, scales)
    entropy(graph, scales)


