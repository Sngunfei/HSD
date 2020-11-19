# -*- encoding: utf-8 -*-

# Hierarchically Structural Distance model

import multiprocessing

import networkx as nx
import numpy as np
import pygsp
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
        self.adjacent = nx.adjacency_matrix(graph).todense()
        self.laplacian = nx.laplacian_matrix(graph).todense()

        self.eigenvalues = None
        self.eigenvectors = None

        self.nodes = list(nx.nodes(graph))
        self.n_node = len(self.nodes)
        self.idx2node, self.node2idx = util.build_node_idx_map(graph)
        self.wavelets = None
        self.hierarchy = None
        self.wavelets_hierarchy = None
        self.distMat = None

    # init HSD model
    def init(self):
        self.hierarchy = read_hierarchical_representation(self.graphName, self.hop)
        self.wavelets = self.calculate_wavelets(self.scale)

    def construct_hierarchy(self):
        self.hierarchy = read_hierarchical_representation(self.graphName, self.hop)

    # caculate wavelet coefficients
    # approx: if True then use chebshev polynomials
    def calculate_wavelets(self, scale, approx=True) -> np.ndarray:
        if approx:
            G = pygsp.graphs.Graph(self.adjacent)
            G.estimate_lmax()
            heat_filter = pygsp.filters.Heat(G, tau=[scale * G._lmax])
            chebyshev = pygsp.filters.approximations.compute_cheby_coeff(heat_filter, m=50)
            wavelets = []
            for idx in range(self.n_node):
                impulse = np.zeros(self.n_node, dtype=np.float)
                impulse[idx] = 1.0
                coeff = pygsp.filters.approximations.cheby_op(G, chebyshev, impulse)
                wavelets.append(coeff)
        else:
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.laplacian)
            wavelets = np.dot(np.dot(self.eigenvectors, np.diag(np.exp(-1 * scale * self.eigenvalues))),
                                 np.transpose(self.eigenvectors))

        threshold = np.vectorize(lambda x: x if x > 1e-4 * 1.0 / self.n_node else 0)
        processed_wavelets = threshold(wavelets)
        self.wavelets = processed_wavelets
        return processed_wavelets


    def get_hierarchical_wavelet_coefficients(self) -> dict:
        if not self.hierarchy or not self.wavelets:
            raise ValueError("graph hierarchy is None or Wavelets not computed")

        res = dict()
        for idx, node in enumerate(self.nodes):
            layers = self.hierarchy[node]
            coeffs = []
            for layer in layers:
                tmp = []
                for neighbor in layer:
                    idx2 = self.node2idx[neighbor]
                    tmp.append(self.wavelets[idx, idx2])
                coeffs.append(tmp)
            res[node] = coeffs
        return res


    # calculate HSD using a single thread
    def calculate_HSD(self):
        if not self.hierarchy or not self.wavelets:
            raise ValueError("graph hierarchy is None or Wavelets not computed")

        distMat = np.zeros((self.n_node, self.n_node), dtype=float)
        for idx1, node1 in tqdm(enumerate(self.nodes)):
            for idx2 in range(idx1 + 1, self.n_node):
                node2 = self.nodes[idx2]
                layers1, layers2 = self.hierarchy[node1], self.hierarchy[node2]
                distance = 0.0
                for hop in range(self.hop):
                    layer_1, layer_2 = layers1[hop], layers2[hop]
                    p, q = [], []
                    for neighbor in layer_1:
                        if neighbor != '':
                            p.append(self.wavelets[idx1, self.node2idx[neighbor]])

                    for neighbor in layer_2:
                        if neighbor != '':
                            q.append(self.wavelets[idx2, self.node2idx[neighbor]])

                    distance += metrics.calculate_distance(p, q, self.metric)

                distMat[idx1, idx2] = distMat[idx2, idx1] = distance

        self.distMat = distMat
        return self.distMat


    # calculate HSD parallelly
    def parallel_calculate_HSD(self, n_workers=3):
        if self.hierarchy is None or self.wavelets is None:
            raise ValueError("graph hierarchy is None or Wavelet not computed")

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
