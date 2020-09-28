# -*- encoding: utf-8 -*-

"""
graph data input/output
"""

import os
import numpy as np
import pandas as pd
from configparser import ConfigParser


def save_vectors(nodes: list, vectors: list, path: str):
    """
    save vectors into csv file.
    :param nodes:
    :param vectors:
    :param path:
    :return:
    """
    df = pd.DataFrame(data=vectors, index=nodes, columns=None, dtype=float)
    df.to_csv(path, header=False, float_format="%.8f")


def save_vectors_dict(vectors: dict, path: str):
    vs = []
    ns = []
    for n, v in vectors.items():
        ns.append(n)
        vs.append(v)
    save_vectors(ns, vs, path)


def read_vectors(path: str) -> dict:
    """
    read embedding vectors from csv file.
    :param path:
    :return:
    """
    if not os.path.exists(path):
        raise FileNotFoundError
    df = pd.read_csv(path, header=None)
    row, col = df.shape
    embedding_dict = {}
    for i in range(row):
        embedding_dict[str(int(df.iloc[i][0]))] = list(df.iloc[i][1:])

    return embedding_dict


def read_distance(path: str, n_nodes: int) -> np.ndarray:
    """
    read calculated distance information from edgelist file.
    :param path:
    :param n_nodes:
    :return:
    """
    mat = np.zeros((n_nodes, n_nodes), dtype=np.float)
    with open(path, mode='r', encoding="utf-8") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            u, v, dist = line.strip().split(" ")
            mat[int(u), int(v)] = mat[int(v), int(u)] = float(dist)
    return mat


def save_results(res: dict, path):
    with open(path, mode="a+", encoding="utf-8") as fout:
        for key, value in res.items():
            fout.write("{} : {}\n".format(key, value))
        fout.write("-"*80 + "\n\n")


def save_distance_csv(path:str, nodes:list, mat):
    node_int = [int(node) for node in nodes]
    df = pd.DataFrame(data=mat, dtype=np.float, index=node_int, columns=node_int)
    df.sort_index(axis=1, inplace=True)
    df.sort_index(axis=0, inplace=True)
    df.to_csv(path, mode="w+", encoding="utf-8", index=True, header=True)


def save_distance_edgelist(path: str, nodes: list, mat: np.ndarray):
    """
    将距离矩阵以边的形式存入文件
    :param path: 路径
    :param nodes: 节点集
    :param mat: 距离矩阵
    :return:
    """
    n = len(mat)
    with open(path, mode="w+", encoding="utf-8") as fout:
        for idx1, node1 in enumerate(nodes):
            for idx2 in range(idx1 + 1, n):
                node2 = nodes[idx2]
                distance = mat[idx1, idx2]
                fout.write(f"{node1} {node2} {distance}\n")


def save_edgelist(path: str, edgelist: list):
    """
    保存边，可以有权重也可以无，边的形式为 edge = (u, v, weight)
    :param path:
    :param edgelist:
    :return:
    """
    with open(path, mode="w+", encoding="utf-8") as fout:
        for edge in edgelist:
            edge_str = ""
            if len(edge) < 2:
                continue
            elif len(edge) == 2:
                edge_str = f"{edge[0]} {edge[1]}"
            elif len(edge) == 3:
                edge_str = f"{edge[0]} {edge[1]} {edge[2]}"
            fout.write(f"{edge_str}\n")


if __name__ == '__main__':
    print()