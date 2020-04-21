# -*- encoding: utf-8 -*-

"""
Hierarchically Structural Distance model.
"""
from collections import defaultdict
import logging
import networkx as nx
import numpy as np
import copy
import multiprocessing
import pandas as pd
from tqdm import tqdm
from ge.model.GraphWave import GraphWave
from ge.utils.distance import calculate_distance
from ge.utils.rw import save_distance_csv, save_distance_edgelist, save_vectors_dict, read_vectors
import math


class HSD:
    def __init__(self, graph, graph_name, scale, hop, metric, n_workers, **kwargs):
        """
        Hierarchicall Structural Distance model.
        :param graph: nx.Graph
        :param graph_name: the name of graph，e.g. 'barbell' and 'mkarate'
        :param scale: the heat coefficient
        :param metric: 'Wasserstein' or 'Helinger'
        :param hop: k-hop local neighborhoods
        :param kwargs:
        """
        self.graph = graph
        self.graph_name = graph_name
        self.scale = scale
        self.hop = hop
        self.metric = metric
        self.n_workers = n_workers
        self.args = kwargs

        # initialize the GraphWave model
        self.wavelet = GraphWave(graph, graph_name, scale=scale)
        self.nodes = self.wavelet.nodes
        self.node2idx = self.wavelet.node2idx
        self.n_nodes = self.wavelet.n_nodes
        self.waveletCoeff = None
        self.nodeLayers = None
        self.distMat = None


    def initialize(self, multi=False):
        if self.waveletCoeff and self.nodeLayers:
            return
        if not multi:
            self.waveletCoeff = self.wavelet.calculate_wavelet_coeff(self.scale)
        self.nodeLayers = self.constructNodeLayers_BFS()


    def constructHierarchicalRepresentation_dijkstra(self, nodes=None):
        """
        根据节点间的最短路径，将节点局部邻域进行层次划分，以节点为中心的嵌套环状结构，其他节点分布在对应的环上。
        :return: dict(dict()), 嵌套字典结构，第一册key为节点，第二层key为距离。
        """
        if not nodes:
            nodes = self.nodes

        res = defaultdict(dict)
        for i, node1 in enumerate(nodes):
            idx1 = self.node2idx[node1]
            rings = defaultdict(list)
            for j in range(i + 1, len(nodes)):
                node2 = nodes[j]
                idx2 = self.node2idx[node2]
                shortestPathLength = nx.dijkstra_path_length(self.graph, node1, node2)
                rings[shortestPathLength].append(idx2)
            res[idx1] = rings
        return res


    def constructNodeLayers_BFS(self):
        """
        利用广搜BFS构建节点的局部层级拓扑结构。
        :return: dict(dict()), 嵌套字典结构，第0层是节点自身，第k层是节点的k-hop邻居。
        """
        logging.info("Start construct hierarchiy of neighborhoods, graph={}, max-hop={}.\n".format(
            self.graph_name, self.hop))

        hierarchy = defaultdict(dict)
        for idx, node in enumerate(self.nodes):
            rings = defaultdict(list)
            visited = set()
            queue = [node]
            visited.add(node)
            rings[0] = copy.deepcopy(queue)
            # if hop=3, then the hierarchical representation will be {0:node, 1:1-hop nodes, 2:2-hop, 3:3-hop}
            for hop in range(1, self.hop + 1):
                size = len(queue)
                for _ in range(size):
                    curNode = queue.pop(0)
                    neighbors = list(nx.neighbors(self.graph, curNode))
                    for _neighbor in neighbors:
                        if _neighbor not in visited:
                            visited.add(_neighbor)
                            queue.append(_neighbor)
                rings[hop] = copy.deepcopy(queue)

            hierarchy[node] = rings
        return hierarchy


    def calculate_structural_distance(self, metric=None,):
        """
        单线程 两两计算节点之间的结构距离
        :param metric:
        :return:
        """
        if not metric:
            metric = self.metric

        if self.waveletCoeff is None or self.nodeLayers is None:
            self.initialize(multi=False)

        self.distMat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for idx1, node1 in enumerate(self.nodes):
            for idx2 in range(idx1 + 1, self.n_nodes):
                node2 = self.nodes[idx2]
                rings1, rings2 = self.nodeLayers[node1], self.nodeLayers[node2]
                d = 0.0
                for hop in range(self.hop + 1):
                    r1, r2 = rings1[hop], rings2[hop]
                    p, q = [], []
                    for neighbor in r1:
                        _idx = self.node2idx[neighbor]
                        p.append(self.waveletCoeff[idx1, _idx])
                    for neighbor in r2:
                        _idx = self.node2idx[neighbor]
                        q.append(self.waveletCoeff[idx2, _idx])

                    d += calculate_distance(p, q, metric)
                self.distMat[idx1, idx2] = self.distMat[idx2, idx1] = d

        dist_path = "../distance/{}/HSD_{}_scale{}_hop{}".format(
            self.graph_name, metric, self.scale, self.hop)
        save_distance_edgelist(dist_path+".edgelist", self.nodes, self.distMat)
        save_distance_csv(dist_path+".csv", self.nodes, self.distMat)
        return self.distMat


    def parallel_calculate_distance(self, metric=None,):
        """
        Calculate structural distance parallelly in one single scale.
        """
        if not metric:
            metric = self.metric

        if self.waveletCoeff is None or self.nodeLayers is None:
            self.initialize(multi=False)

        print("Start parallelly calculate structural distance, n_workers={}.\n".format(self.n_workers))
        self.distMat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        pool = multiprocessing.Pool(self.n_workers)
        result = {}
        for idx in range(self.n_nodes):
            res = pool.apply_async(self._calculateDistanceWorker,
                                    args=(idx, metric))
            result[idx] = res
        pool.close()
        pool.join()

        for idx in range(self.n_nodes):
            result[idx] = result[idx].get()

        for idx1, dists in result.items():
            for idx2, distance in dists.items():
                self.distMat[idx1, idx2] = self.distMat[idx2, idx1] = distance

        dist_file_path = "../distance/{}/HSD_{}_scale{}_hop{}".format(
            self.graph_name, self.metric, self.scale, self.hop)
        save_distance_edgelist(dist_file_path+".edgelist", self.nodes, self.distMat)
        save_distance_csv(dist_file_path+".csv", self.nodes, self.distMat)

        return self.distMat


    def _calculateDistanceWorker(self, start, metric):
        dist = dict()
        node1 = self.nodes[start]
        rings1 = self.nodeLayers[node1]
        for idx in range(start + 1, self.n_nodes):
            _neighbor = self.nodes[idx]
            rings2 = self.nodeLayers[_neighbor]
            d = 0.0
            for hop in range(self.hop + 1):
                r1, r2 = rings1[hop], rings2[hop]
                p, q = [], []
                for _neighbor in r1:
                    _idx = self.node2idx[_neighbor]
                    p.append(self.waveletCoeff[start, _idx])
                for _neighbor in r2:
                    _idx = self.node2idx[_neighbor]
                    q.append(self.waveletCoeff[idx, _idx])
                d += calculate_distance(p, q, metric)
            dist[idx] = d

        return dist


    def multi_scales_wavelet(self, metric=None, n_scales=200):
        """
        单线程 多尺度分析
        :param n_scales:
        :return:
        """
        eigenvalues = self.wavelet.e
        print("min eigenvalues: {}, max eigenvalues: {}".format(min(eigenvalues), max(eigenvalues)))
        scales = np.exp(np.linspace(np.log(0.01), np.log(max(eigenvalues)), n_scales))
        print("selected scales: [{}]".format(",".join(map(str, scales))))

        if metric is None:
            metric = self.metric

        data = dict()  # 各尺度下的数据汇总
        hierarchy = self.nodeLayers
        vectors = defaultdict(list)
        for scale in tqdm(scales):
            coeffs = self.wavelet.calculate_wavelet_coeff(scale)
            state = dict()
            for idx, node in enumerate(self.nodes):
                p = [coeffs[idx, idx]]
                for k_hop in range(1, self.hop + 1):
                    k_hop_neighbors = hierarchy[node].get(k_hop, [])
                    if len(k_hop_neighbors) == 0:
                        k_hop_sum = 0.0
                    else:
                        k_hop_sum = np.sum([coeffs[idx][self.node2idx[neighbor]] for neighbor in k_hop_neighbors])
                    p.append(k_hop_sum)
                if math.isclose(1.0 - sum(p), 0.0, abs_tol=1e-6):
                    p.append(0.0)
                else:
                    p.append(1.0 - sum(p))
                state[idx] = p
                vectors[node].extend(p)
            data[scale] = state

        save_vectors_dict(vectors, path="../../coeff/{}_hop{}_scales{}.csv")

        # 各尺度下分别计算距离
        dist_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float)
        for scale, state in data.items():
            for idx1 in range(self.n_nodes):
                for idx2 in range(idx1 + 1, self.n_nodes):
                    p, q = state[idx1], state[idx2]
                    d = calculate_distance(p, q, metric)
                    dist_mat[idx1, idx2] += d
                    dist_mat[idx2, idx1] += d
        dist_mat = dist_mat / n_scales  # 取均值

        dist_file_path = "../distance/{}/HSD_multi_{}_hop{}".format(
            self.graph_name, metric, self.hop)
        save_distance_edgelist(dist_file_path+".edgelist", self.nodes, dist_mat)
        save_distance_csv(dist_file_path+".csv", self.nodes, dist_mat)

        return dist_mat


    def parallel_calculate_coeff_sum(self, n_scales) -> dict:
        """
        并行计算多尺度下的小波系数，然后分层求和后，拼接起来，
        :param n_scales:
        :return: dict{node：coeffs}
        """
        eigenvalues = self.wavelet.e
        print("min eigenvalues: {}, max eigenvalues: {}".format(min(eigenvalues), max(eigenvalues)))
        scales = np.exp(np.linspace(np.log(0.01), np.log(max(eigenvalues)), n_scales))

        print("Start compute multi-scales wavelet parallelly.")
        pool = multiprocessing.Pool(self.n_workers)
        result = {}
        for idx, scale in enumerate(scales):
            # res = pool.apply_async(self.wavelet.calculate_wavelet_coeff_chebyshev, args=(scale, 30))
            res = pool.apply_async(self.wavelet.calculate_wavelet_coeff, args=(scale,))
            result[idx] = res
        pool.close()
        pool.join()

        for idx, _ in result.items():
            result[idx] = result[idx].get()

        vectors = defaultdict(list)
        for i, scale in enumerate(scales):
            coeffs = result[i]
            for idx, node in enumerate(self.nodes):
                p = [coeffs[idx, idx]]
                for k_hop in range(1, self.hop + 1):
                    k_hop_neighbors = self.nodeLayers[node].get(k_hop, [])
                    if len(k_hop_neighbors) == 0:
                        k_hop_sum = 0.0
                    else:
                        k_hop_sum = np.sum([coeffs[idx][self.node2idx[neighbor]] for neighbor in k_hop_neighbors])
                    p.append(k_hop_sum)
                if math.isclose(1.0 - sum(p), 0.0, abs_tol=1e-6):
                    p.append(0.0)
                else:
                    p.append(1.0 - sum(p))
                vectors[node].extend(p)

        save_vectors_dict(vectors, path="../coeff/{}_hop{}_scales{}.csv".format(self.graph_name, self.hop, n_scales))
        return vectors


    def parallel_multi_scales_wavelet(self, metric=None, n_scales=200) -> np.ndarray:
        """
        多尺度分析，并行计算距离
        :param metric:
        :param n_scales:
        :return:
        """
        path = "../coeff/{}_hop{}_scales{}.csv".format(self.graph_name, self.hop, n_scales)
        vectors = read_vectors(path)
        if vectors is None:
            vectors = self.parallel_calculate_coeff_sum(n_scales)

        if metric is None:
            metric = self.metric

        pool = multiprocessing.Pool(self.n_workers)
        result = {}
        for idx, node in enumerate(self.nodes):
            res = pool.apply_async(self._calculate_distance_worker, args=(idx, vectors, n_scales))
            result[idx] = res
        pool.close()
        pool.join()
        for idx, _ in result.items():
            result[idx] = result[idx].get()

        dist_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float)
        for idx, dist_info in result.items():
            for idx2, dist in dist_info.items():
                dist_mat[idx, idx2] = dist_mat[idx2, idx] = dist

        dist_mat = dist_mat / n_scales

        dist_file_path = "../distance/{}/HSD_multi_{}_hop{}".format(
            self.graph_name, metric, self.hop)

        save_distance_csv(dist_file_path+".csv", self.nodes, dist_mat)
        save_distance_edgelist(dist_file_path+".edgelist", self.nodes, dist_mat)
        return dist_mat


    def _calculate_distance_worker(self, start, vectors, n_scales,) -> dict:
        """
        多尺度并行计算距离的worker，在多尺度下，每个节点有一个vector，
        :param start:
        :param vectors:
        :return:
        """
        node1 = self.nodes[start]
        v1 = vectors[node1]
        dist = dict()
        seg = len(v1) // n_scales
        for idx in range(start + 1, self.n_nodes):
            node2 = self.nodes[idx]
            v2 = vectors[node2]
            d = 0.0
            for i in range(n_scales):
                p = v1[i * seg : (i + 1) * seg]
                q = v2[i * seg : (i + 1) * seg]
                d += calculate_distance(p, q, self.metric)
            dist[idx] = d

        return dist
