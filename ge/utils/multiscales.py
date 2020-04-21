# -*- encoding: utf-8 -*-

"""
一些关于多尺度分析的处理函数
"""
from collections import defaultdict
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from ge.utils.rw import save_vectors_dict

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


if __name__ == '__main__':
    path = "E:\workspace\py\graph-embedding\coeff\\usa_hop7_scales100.csv"
    cumulate_wavelet_coeffs(path, 7, 6)

