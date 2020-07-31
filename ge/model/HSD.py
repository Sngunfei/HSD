# -*- encoding: utf-8 -*-

"""
Hierarchically Structural Distance model.
"""
import math
import multiprocessing
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from ge.hierarchy.hierarchical import read_hierarchical_representation
from ge.model.GraphWave import GraphWave
from ge.tools.distance import calculate_distance
from ge.tools.rw import save_vectors_dict, read_vectors


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
        self.wavelet_coeff = None
        self.node_layers = None
        self.dist_mat = None


    def initialize(self, multi=False, ):
        if self.wavelet_coeff and self.node_layers:
            return
        if not multi:
            if self.n_nodes > 5000:
                self.wavelet_coeff = self.wavelet.calculate_wavelet_coeff_chebyshev(self.scale, order=8)
            else:
                self.wavelet_coeff = self.wavelet.calculate_wavelet_coeff(self.scale)
        # 为了避免每次运行时都构造层级结构，事先存储，运行的时候读取。
        self.node_layers = read_hierarchical_representation(self.graph_name, layer_cnt=3)
        print("HSD model init done.")

    def initWaveletCoeffByReducedCoeff(self, reduced_coeff: dict):
        # 需要复用之前已经计算好的小波系数，直接从文件里读取精简后的小波系数
        # 若在dict里未出现，则说明小波系数的规模数量级太小，默认为0
        # 为了解耦，IO操作均在本函数外执行
        if self.wavelet_coeff is not None:
            print("Original wavelet coefficient matrix already exists, rewrite...")
        self.wavelet_coeff = np.zeros(shape=(self.n_nodes, self.n_nodes), dtype=np.float)
        for node, neighbors in reduced_coeff.items():
            idx = self.node2idx[node]
            for neighbor, coeff_value in neighbors.items():
                idx2 = self.node2idx[neighbor]
                self.wavelet_coeff[idx, idx2] = coeff_value

    def get_reduced_wavelet_coefficient(self, hop=3, threshold=1e-4) -> dict:
        # 由于在图中，只有近邻节点上小波系数的数量级才比较可观，绝大部分都是0，所以没必要全都存储。
        # 只需要根据层级结构，获取近邻之间
        if not self.node_layers or not self.wavelet_coeff:
            self.initialize(multi=False)
        print("Start filter & get reduced wavelet coefficients.")
        reduced_coeff = defaultdict(dict)
        for idx, node in enumerate(self.nodes):
            neighborhoods = self.node_layers[node]
            for kth_hop in range(hop):
                kth_hop_neighbors = neighborhoods.get(kth_hop, [])
                if len(kth_hop_neighbors) == 0:
                    break
                for neighbor in kth_hop_neighbors:
                    idx2 = self.node2idx[neighbor]
                    coeff_value = self.wavelet_coeff[idx, idx2]
                    if coeff_value < threshold:
                        continue
                    reduced_coeff[node][neighbor] = coeff_value
        self.reduced_wavelet_coeff = reduced_coeff
        return reduced_coeff

    def calculate_structural_distance(self, metric=None, ):
        # 单线程 两两计算节点之间的结构距离
        if not metric:  # 优先使用临时指定的metric
            metric = self.metric

        if self.wavelet_coeff is None or self.node_layers is None:
            self.initialize(multi=False)
        self.dist_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for idx1, node1 in tqdm(enumerate(self.nodes)):
            for idx2 in range(idx1 + 1, self.n_nodes):
                node2 = self.nodes[idx2]
                rings1, rings2 = self.node_layers[node1], self.node_layers[node2]
                distance = 0.0
                for hop in range(self.hop):
                    r1, r2 = rings1[hop], rings2[hop]
                    p, q = [], []
                    for neighbor in r1:
                        if neighbor != '':
                            _idx = self.node2idx[neighbor]
                            p.append(self.wavelet_coeff[idx1, _idx])
                    for neighbor in r2:
                        if neighbor != '':
                            _idx = self.node2idx[neighbor]
                            q.append(self.wavelet_coeff[idx2, _idx])
                    distance += calculate_distance(p, q, metric)
                self.dist_mat[idx1, idx2] = self.dist_mat[idx2, idx1] = distance

        return self.dist_mat


    def parallel_calculate_distance(self, metric=None,):
        """
        Calculate structural distance parallelly in one single scale.
        """
        if not metric:
            metric = self.metric
        if self.wavelet_coeff is None or self.node_layers is None:
            self.initialize(multi=False)

        print("Start parallelly calculate structural distance, n_workers={}.\n".format(self.n_workers))
        self.dist_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        pool = multiprocessing.Pool(self.n_workers)
        result = {}
        for idx in range(self.n_nodes):
            res = pool.apply_async(self._calculate_distance_worker,
                                   args=(idx, metric))
            result[idx] = res
        pool.close()
        pool.join()
        for idx in range(self.n_nodes):
            result[idx] = result[idx].get()
        print("parallelly calculate structural distance done. \n")

        for idx1, dists in result.items():
            for idx2, distance in dists.items():
                self.dist_mat[idx1, idx2] = self.dist_mat[idx2, idx1] = distance

        return self.dist_mat


    def _calculateDistanceWorker(self, start, metric):
        dist = dict()
        node1 = self.nodes[start]
        rings1 = self.node_layers[node1]
        for idx in range(start + 1, self.n_nodes):
            _neighbor = self.nodes[idx]
            rings2 = self.node_layers[_neighbor]
            d = 0.0
            for hop in range(self.hop):
                r1, r2 = rings1[hop], rings2[hop]
                p, q = [], []
                for _neighbor in r1:
                    _idx = self.node2idx[_neighbor]
                    p.append(self.wavelet_coeff[start, _idx])
                for _neighbor in r2:
                    _idx = self.node2idx[_neighbor]
                    q.append(self.wavelet_coeff[idx, _idx])
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
        hierarchy = self.node_layers
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
        self.dist_mat = dist_mat / n_scales  # 取均值

        return self.dist_mat

    def calculate_multi_scales_coeff_sum(self, n_scales):
        self.initialize(multi=True)
        eigenvalues = self.wavelet.e
        print("min eigenvalues: {}, max eigenvalues: {}".format(min(eigenvalues), max(eigenvalues)))
        scales = np.exp(np.linspace(np.log(0.01), np.log(max(eigenvalues)), n_scales))
        print("start calculate multi scales wavelet.")
        vectors = defaultdict(list)
        for _, scale in tqdm(enumerate(scales)):
            #coeffs = self.wavelet.calculate_wavelet_coeff_chebyshev(scale, order=10)
            coeffs = self.wavelet.calculate_wavelet_coeff(scale)
            for idx, node in enumerate(self.nodes):
                p = [coeffs[idx, idx]]
                for k_hop in range(1, self.hop):
                    k_hop_neighbors = self.node_layers[node].get(k_hop, [])
                    if len(k_hop_neighbors) == 0:
                        k_hop_sum = 0.0
                    else:
                        k_hop_sum = np.sum([coeffs[idx][self.node2idx[neighbor]] if neighbor != '' else 0 for neighbor in k_hop_neighbors])
                    p.append(k_hop_sum)
                if math.isclose(1.0 - sum(p), 0.0, abs_tol=1e-6):
                    p.append(0.0)
                else:
                    p.append(1.0 - sum(p))
                vectors[node].extend(p)
            del coeffs
        save_vectors_dict(vectors, path="../coeff/precise_{}_hop{}_scales{}.csv".format(self.graph_name, self.hop, n_scales))
        print("done")
        return vectors

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
            res = pool.apply_async(self.wavelet.calculate_wavelet_coeff_chebyshev, args=(scale, 10))
            #res = pool.apply_async(self.wavelet.calculate_wavelet_coeff, args=(scale,))
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
                    k_hop_neighbors = self.node_layers[node].get(k_hop, [])
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
        print("compute multi-scales coeffs done.")
        save_vectors_dict(vectors, path="../coeff/{}_hop{}_scales{}.csv".format(self.graph_name, self.hop, n_scales))
        return vectors

    def single_multi_scales_wavelet(self, metric=None, n_scales=200, reuse=True) -> np.ndarray:
        # 多尺度分析，并行计算距离
        path = "../coeff/{}_hop{}_scales{}.csv".format(self.graph_name, self.hop, n_scales)
        if metric is None:
            metric = self.metric
        vectors = None
        if reuse:
            vectors = read_vectors(path)
            print(f"reuse multi-scale wavelt coeff. number of scales {(len(vectors))}.")
        if vectors is None:
            print("start calculate multi-scale wavelet coeff.")
            vectors = self.parallel_calculate_coeff_sum(n_scales)

        dist_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float)
        for idx1, node1 in tqdm(enumerate(self.nodes)):
            v1 = vectors[node1]
            seg = len(v1) // n_scales
            for idx2 in range(idx1 + 1, self.n_nodes):
                node2 = self.nodes[idx2]
                v2 = vectors[node2]
                d = 0.0
                for i in range(n_scales):
                    p = v1[i * seg: (i + 1) * seg]
                    q = v2[i * seg: (i + 1) * seg]
                    #wasser_dist = sum([abs(x - y) for x, y in zip(sorted(p), sorted(q))])
                    #d += wasser_dist
                    #print(node1,node2, p, q)
                    d += calculate_distance(p, q, metric)
                dist_mat[idx1, idx2] = dist_mat[idx2, idx1] = d
        print("done.")
        return dist_mat

    def parallel_multi_scales_wavelet(self, metric=None, n_scales=200, reuse=True) -> np.ndarray:
        # 多尺度分析，并行计算距离
        path = "../coeff/{}_hop{}_scales{}.csv".format(self.graph_name, self.hop, n_scales)

        if metric is None:
            metric = self.metric

        vectors = None
        if reuse:
            vectors = read_vectors(path)
            print(f"reuse multi-scale wavelt coeff. number of scales {(len(vectors))}.")
        if vectors is None:
            print("start calculate multi-scale wavelet coeff.")
            vectors = self.parallel_calculate_coeff_sum(n_scales)

        pool = multiprocessing.Pool(self.n_workers)
        result = {}
        for idx, node in enumerate(self.nodes):
            res = pool.apply_async(self._calculate_distance_worker, args=(idx, vectors, n_scales, metric))
            result[idx] = res
        pool.close()
        pool.join()
        for idx, _ in result.items():
            result[idx] = result[idx].get()

        dist_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float)
        for idx, dist_info in result.items():
            for idx2, dist in dist_info.items():
                dist_mat[idx, idx2] = dist_mat[idx2, idx] = dist

        # 求均值，避免结构距离被[多尺度的数目]影响
        dist_mat = dist_mat / n_scales
        return dist_mat


    def _calculate_distance_worker(self, start, vectors, n_scales, metric, ) -> dict:
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
                d += calculate_distance(p, q, metric)
            dist[idx] = d

        return dist
