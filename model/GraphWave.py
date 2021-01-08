# -*- encoding: utf-8 -*-

import networkx as nx
import numpy as np
import pygsp
from tqdm import tqdm
from tools import util


class GraphWave(object):

    def __init__(self, graph:nx.Graph):
        """
        Hierarchicall Structural Distance model.
        :param graph: nx.Graph
        """
        self.graph = graph
        self.adjacent = nx.adjacency_matrix(graph).todense()
        self.laplacian = nx.laplacian_matrix(graph).todense()
        self.nodes = list(nx.nodes(graph))

        self.idx2node, self.node2idx = util.build_node_idx_map(graph)
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.laplacian)
        self.wavelets = None


    # caculate wavelet coefficients
    # approx: if True then use chebshev polynomials
    def calculate_wavelets(self, scale, approx=True) -> np.ndarray:
        if approx:
            G = pygsp.graphs.Graph(self.adjacent)
            G.estimate_lmax()
            heat_filter = pygsp.filters.Heat(G, tau=[scale * G._lmax])
            chebyshev = pygsp.filters.approximations.compute_cheby_coeff(heat_filter, m=40)
            wavelets = []
            for idx in range(len(self.nodes)):
                impulse = np.zeros(len(self.nodes), dtype=np.float)
                impulse[idx] = 1.0
                coeff = pygsp.filters.approximations.cheby_op(G, chebyshev, impulse)
                wavelets.append(coeff)
        else:
            assert getattr(self, "eigenvalues", None) is not None, "GraphWave eigenvalues is None!"
            wavelets = np.dot(np.dot(self.eigenvectors, np.diag(np.exp(-1 * scale * self.eigenvalues))),
                                 np.transpose(self.eigenvectors))

        threshold = np.vectorize(lambda x: x if x > 1e-5 * 1.0 / len(self.nodes) else 0)
        processed_wavelets = threshold(wavelets)
        self.wavelets = processed_wavelets
        return processed_wavelets


    # 计算特征函数的采样值
    def calculate_characteristic_value(self, X: np.ndarray, sample_points):
        embedding = []
        for _, t in enumerate(sample_points):
            value = np.mean(np.exp(1j * X * t))
            embedding.append(value.real)
            embedding.append(value.imag)
        return np.array(embedding, dtype=np.float)


    def embed(self, sample_points):
        assert self.wavelets is not None, "GraphWave wavelets is None!"
        embedding_dict = dict()
        for idx, node in tqdm(enumerate(self.nodes)):
            wavelet_coeffs = self.wavelets[idx, :]
            embedding_vector = self.calculate_characteristic_value(wavelet_coeffs, sample_points)
            embedding_dict[node] = embedding_vector
        return embedding_dict


# 根据特征值计算尺度选取范围
def recommend_scale_range(eignvalues: list) -> (float, float):
    eignvalues = sorted(eignvalues)
    e1, en = eignvalues[0], eignvalues[-1]
    for e in eignvalues:
        if e > 0.001:
            e1 = e
            break
    scale_min, scale_max = scale_boundary(e1, en)
    return scale_min, scale_max


# calculate the scale range discussed in GraphWave
def scale_boundary(e1, eN, eta=0.85, gamma=0.95):
    t = np.sqrt(e1 * eN)
    sMax = - np.log(eta) / t
    sMin = - np.log(gamma) / t
    return sMin, sMax
