# -*- coding:utf-8 -*-

"""
GraphWave model
"""
import logging
import numpy as np
from scipy import sparse
import networkx as nx
from tqdm import tqdm
import pygsp
from ge.utils.util import compute_chebshev_coeff_basis, build_node_idx_map
np.set_printoptions(suppress=True, precision=5)


class GraphWave:
    def __init__(self, graph, graph_name, scale=5.0, sample_number=16, step_size=20.0):
        self.graph_name = graph_name
        self.graph = graph
        self.scale = scale
        self.n_nodes = nx.number_of_nodes(graph)
        self.nodes = list(nx.nodes(graph))

        self.A = nx.adjacency_matrix(graph)
        self.L = nx.laplacian_matrix(self.graph)
        self.e, self.u = np.linalg.eigh(self.L.toarray())
        self.idx2node, self.node2idx = build_node_idx_map(self.graph)

        self.sample_number = sample_number
        self.step_size = step_size
        self.sample_points = list(map(lambda x: x * step_size, range(0, sample_number)))
        self.wavelet_coeff = None
        self.embeddings = None


    def calculate_wavelet_coeff_chebyshev(self, scale, order):
        """
        Given the Chebyshev polynomial, graph the approximate wavelet coefficients is calculated.
        :param scale:
        :param order:  the order of chebyshev polynomials.
        :return:
        """
        G = pygsp.graphs.Graph(self.A)
        G.estimate_lmax()
        heat_filter = pygsp.filters.Heat(G, tau=[scale])
        chebyshev = pygsp.filters.approximations.compute_cheby_coeff(heat_filter, m=order)

        wavelet_coeffs = []
        for idx in range(self.n_nodes):
            impulse = np.zeros(self.n_nodes, dtype=np.float)
            impulse[idx] = 1.0
            coeff = pygsp.filters.approximations.cheby_op(G, chebyshev, impulse)
            wavelet_coeffs.append(coeff)

        return np.asarray(wavelet_coeffs, dtype=np.float)


    def _calculate_node_coefficients(self, node_idx, scale):
        """
        计算单个节点的小波系数
        :param node_idx: 节点标号，int
        :param scale: 热系数，即尺度，float
        :return: 该尺度下的该节点对应的小波系数，ndarray(n, 1)
        """
        impulse = np.zeros(self.n_nodes)
        impulse[node_idx] = 1
        coefficients = np.dot(np.dot(np.dot(self.u, np.diag(np.exp(-scale * self.e))),
                             np.transpose(self.u)), impulse)
        return coefficients


    def _calculate_embedding(self, wavelet_coefficients, mode="cha"):
        """
        利用单个节点的小波系数去计算嵌入向量。
        :param wavelet_coefficients:
        :param mode: 'cha': characteristic function,
        :return: a single embedding vector
        """
        if mode not in ["cha", "mog", "mo"]:
            raise ValueError("The embedding mode:{} is not supported.".format(mode))

        embedding = []
        for i, t in enumerate(self.sample_points):
            if mode == "cha":
                value = np.mean(np.exp(1j * wavelet_coefficients * t))
                embedding.append(value.real)
                embedding.append(value.imag)
            elif mode == "mog":
                value = np.mean(np.exp(wavelet_coefficients * t))
                embedding.append(value)
            elif mode == "mo":
                value = np.mean(wavelet_coefficients ** i)
                embedding.append(value)

        return np.array(embedding)


    def single_scale_embedding(self, scale=None, mode="cha") -> dict:
        """
        在单一尺度下计算嵌入向量。
        :param scale: 热系数，即尺度，float类型，默认为热系数参数值，但仍可以临时指定。
        :param mode: characteristic function, moment generating function, moment
        :return: 该尺度下的所有节点对应的嵌入向量。
        """
        if mode not in ["cha", "mog", "mo"]:
            raise ValueError("The embedding mode:{} is not supported.".format(mode))
        if scale is None:
            scale = self.scale

        self.embeddings = {}
        for node_idx in tqdm(range(self.n_nodes)):
            node_wave_coeff = self._calculate_node_coefficients(node_idx, scale)
            node_embedding = self._calculate_embedding(node_wave_coeff, mode)
            self.embeddings[self.nodes[node_idx]] = node_embedding

        return self.embeddings


    def multi_scale_embedding(self, scales, mode="cha"):
        """
        multi-scales embedding
        :param scales: [scale_1, scale_2, ...]
        :param mode:  embedding mode.
        :return:
        """
        multi_embeddings = dict()
        for i in tqdm(range(len(scales))):
            multi_embeddings[scales[i]] = self.single_scale_embedding(scales[i], mode)

        return multi_embeddings


    def calculate_wavelet_coeff(self, scale=None):
        """
        计算某尺度下的小波系数，以供后续针对小波系数进行研究。
        :param scale: 尺度参数，即heat coefficient, float
        :return: 小波系数矩阵，shape=(n, n), ndarray
        """
        logging.info("Start calculate {} wavelet coefficients, scale={}.\n".format(
            self.graph_name, self.scale))
        scale = self.scale if scale is None else scale
        coeff_mat = []
        for node_idx in range(self.n_nodes):
            coeff = self._calculate_node_coefficients(node_idx, scale)
            coeff_mat.append(coeff)
        logging.info("Calculate wavelet coefficients, done.")
        return np.asarray(coeff_mat, dtype=np.float)


def laplacian(adj):
    """
    正则化拉普拉斯矩阵
    :param adj: 邻接矩阵
    :return: 正则拉普拉斯矩阵
    """
    n, _ = adj.shape
    posinv = np.vectorize(lambda x: float(1.0) / np.sqrt(x) if x > 1e-10 else 0.0)
    diag = sparse.diags(np.array(posinv(adj.sum(0))).reshape([-1, ]), 0)
    lap = sparse.eye(n) - diag.dot(adj.dot(diag))
    return lap


def scale_boundary(e1, eN, eta=0.85, gamma=0.95):
    """
    calculate the scale of heat diffusion wavelets.
    :param e1:
    :param eN:
    :param eta:
    :param gamma:
    :return:
    """
    t = np.sqrt(e1 * eN)
    sMax = - np.log(eta) / t
    sMin = - np.log(gamma) / t
    return sMin, sMax









































