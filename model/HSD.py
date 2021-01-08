# -*- encoding: utf-8 -*-

# Hierarchically Structural Distance model

import multiprocessing

import networkx as nx
import numpy as np
import pygsp
from scipy.stats import wasserstein_distance
from tqdm import tqdm

from tools import metrics
from tools import read_hierarchical_representation
from tools import util

class HSD(object):

    def __init__(self, graph, graphName, scale, hop, metric):
        """
        Hierarchicall Structural Distance model.
        :param graph: nx.Graph
        :param graphName: the name of graph
        :param scale: the heat coefficient
        :param metric: 'Wasserstein' or 'Hellinger'
        :param hop: k-hop local neighborhoods
        """
        self.graph = graph
        self.graphName = graphName
        self.scale = scale
        self.hop = hop
        self.metric = metric
        self.A = nx.adjacency_matrix(graph).todense()
        self.L = nx.laplacian_matrix(graph).todense()
        #self.L = nx.normalized_laplacian_matrix(graph).todense()

        self.nodes = list(nx.nodes(graph))
        self.n_node = len(self.nodes)
        self.idx2node, self.node2idx = util.build_node_idx_map(graph)
        self.hierarchy = None

    # init HSD model
    def init(self):
        self.hierarchy = read_hierarchical_representation(self.graphName, self.hop)

    # caculate wavelet coefficients
    # approx: if True then use chebshev polynomials
    def calculate_wavelets(self, scale, approx=True) -> np.ndarray:
        if approx:
            G = pygsp.graphs.Graph(self.A)
            G.estimate_lmax()
            heat_filter = pygsp.filters.Heat(G, tau=[scale * G._lmax])
            chebyshev = pygsp.filters.approximations.compute_cheby_coeff(heat_filter, m=50)
            wavelets = np.empty(shape=(self.n_node, self.n_node))
            for idx in range(self.n_node):
                impulse = np.zeros(self.n_node, dtype=np.float)
                impulse[idx] = 1.0
                coeff = pygsp.filters.approximations.cheby_op(G, chebyshev, impulse)
                wavelets[idx, :] = coeff
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(self.L)
            wavelets = np.dot(np.dot(eigenvectors, np.diag(np.exp(-1 * scale * eigenvalues))),
                                 np.transpose(eigenvectors))

        threshold = np.vectorize(lambda x: x if x > 1e-4 * 1.0 / self.n_node else 0)
        wavelets = threshold(wavelets)
        return wavelets


    # 得到系数的分层表示
    def get_hierarchical_coeffcients(self, wavelets) -> dict:
        coeffs_dict = dict()
        for idx, node in enumerate(self.nodes):
            neighbor_layers = self.hierarchy[node]
            coeffs = []
            for neighbor_set in neighbor_layers:
                tmp = []
                for neighbor in neighbor_set:
                    idx2 = self.node2idx[neighbor]
                    tmp.append(wavelets[idx, idx2])
                coeffs.append(tmp)
            coeffs_dict[node] = coeffs
        return coeffs_dict


    # 作为baseline，统计节点各hop邻居的个数，组成向量
    def get_nodes_hierarchical_degree(self) -> dict:
        hierarchical_degrees = dict()
        for node, layers in self.hierarchy.items():
            hop_degree = [len(level) for level in layers]
            if len(hop_degree) < self.hop:
                hop_degree = hop_degree + [0] * (self.hop - len(hop_degree))
            hierarchical_degrees[node] = hop_degree
        return hierarchical_degrees


    # calculate HSD using a single thread
    def calculate_structural_distance(self, scale, approx=False):
        wavelets = self.calculate_wavelets(scale, approx)
        coeffs_dict = self.get_hierarchical_coeffcients(wavelets)

        dist_mat = np.zeros((self.n_node, self.n_node), dtype=float)
        for idx1, node1 in tqdm(enumerate(self.nodes)):
            for idx2 in range(idx1 + 1, self.n_node):
                node2 = self.nodes[idx2]
                coeffs_layers1, coeffs_layers2 = coeffs_dict[node1], coeffs_dict[node2]
                distance = 0.0
                for hop in range(self.hop+1):
                    # coeffs doesn't have to share same length
                    coeffs1, coeffs2 = coeffs_layers1[hop], coeffs_layers2[hop]
                    distance += wasserstein_distance(coeffs1, coeffs2)
                dist_mat[idx1, idx2] = dist_mat[idx2, idx1] = distance

        return dist_mat


    # calculate HSD parallelly
    def parallel_calculate_HSD(self, n_workers=3):
        distMat = np.zeros((self.n_node, self.n_node), dtype=float)
        pool = multiprocessing.Pool(n_workers)
        states = {}
        for idx in range(self.n_node):
            res = pool.apply_async(HSD._calculate_worker, args=(self, idx))
            states[idx] = res
        pool.close()
        pool.join()

        results = []
        for idx in range(self.n_node):
            results.append(states[idx].get())

        for idx1, dists in enumerate(results):
            for idx2 in range(idx1 + 1, self.n_node):
                distMat[idx1, idx2] = distMat[idx2, idx1] = dists[idx2]

        self.distMat = distMat
        return self.distMat


    def _calculate_worker(self, startIndex: int) -> np.ndarray:
        dists = np.zeros(self.n_node)
        node = self.nodes[startIndex]
        layers = self.hierarchy[node]
        for idx in range(startIndex + 1, self.n_node):
            other = self.nodes[idx]
            _layers = self.hierarchy[other]
            d = 0.0
            for hop in range(self.hop):
                r1, r2 = layers[hop], _layers[hop]
                p, q = [], []
                for neighbor in r1:
                    p.append(self.wavelets[startIndex, self.node2idx[neighbor]])

                for neighbor in r2:
                    q.append(self.wavelets[startIndex, self.node2idx[neighbor]])

                d += metrics.calculate_distance(p, q, self.metric)

            dists[idx] = d

        return dists
