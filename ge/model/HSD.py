# -*- encoding: utf-8 -*-

"""
Hierarchically Structural Distance model.
"""
from collections import defaultdict
import networkx as nx
import numpy as np
import copy
import multiprocessing
import pandas as pd
from tqdm import tqdm
from ge.model.GraphWave import GraphWave
from ge.utils.distance import calculate_distance
from ge.utils.rw import save_distance_csv, save_distance_edgelist
import math


class HSD:
    def __init__(self, graph, graph_name, scale, hop, metric, **kwargs):
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
        self.args = kwargs

        # initialize the GraphWave model
        self.wavelet = GraphWave(graph, graph_name, scale=scale)
        self.nodes = self.wavelet.nodes
        self.node2idx = self.wavelet.node2idx
        self.n_nodes = self.wavelet.n_nodes
        self.waveletCoeff = None
        self.nodeLayers = None
        self.distMat = None


    def initialize(self):
        if self.waveletCoeff and self.nodeLayers:
            print("HSD already initialized.")
            return

        print("Start calculating wavelet coefficients, scale={}.\n".format(self.scale))
        self.waveletCoeff = self.wavelet.cal_all_wavelet_coeffs(self.scale)
        print("done.")
        print("Start constructing hierarchiy of neighborhoods, maxHop={}.\n".format(self.hop))
        self.nodeLayers = self.constructNodeLayers_BFS()
        print("done.")


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
        res = defaultdict(dict)
        for idx, node in enumerate(self.nodes):
            rings = defaultdict(list)
            visited = set()
            queue = [node]
            visited.add(node)
            for hop in range(self.hop):
                rings[hop] = copy.deepcopy(queue)
                size = len(queue)
                for _ in range(size):
                    curNode = queue.pop(0)
                    neighbors = list(nx.neighbors(self.graph, curNode))
                    for _neighbor in neighbors:
                        if _neighbor not in visited:
                            visited.add(_neighbor)
                            queue.append(_neighbor)
            res[node] = rings
        return res


    def calculateStructuralDistance(self, metric=None, savePath=None):
        """
        单线程 两两计算节点之间的结构距离
        :param metric:
        :param savePath:
        :return:
        """
        if not metric:
            metric = self.metric

        if self.waveletCoeff is None or self.nodeLayers is None:
            print("Not initialize model yet, start init...")
            self.initialize()

        self.distMat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for idx1, node1 in enumerate(self.nodes):
            for idx2 in range(idx1, self.n_nodes):
                node2 = self.nodes[idx2]
                rings1, rings2 = self.nodeLayers[node1], self.nodeLayers[node2]
                dist = 0.0
                for hop in range(self.hop):
                    r1, r2 = rings1[hop], rings2[hop]
                    p, q = [], []
                    for neighbor in r1:
                        _idx = self.node2idx[neighbor]
                        p.append(self.waveletCoeff[idx1, _idx])
                    for neighbor in r2:
                        _idx = self.node2idx[neighbor]
                        q.append(self.waveletCoeff[idx2, _idx])
                    dist += calculate_distance(p, q, metric)

                self.distMat[idx1, idx2] = self.distMat[idx2, idx1] = dist

        if savePath:
            with open(savePath, mode='w+', encoding='utf-8') as fout:
                for idx1 in range(self.n_nodes):
                    node1 = self.nodes[idx1]
                    for idx2 in range(idx1 + 1, self.n_nodes):
                        node2 = self.nodes[idx2]
                        fout.write("{} {} {}\n".format(node1, node2, self.distMat[idx1, idx2]))
        return self.distMat


    def calculateDistanceParallel(self, metric=None, n_workers=4, save=False):
        """
        多进程并行计算结构距离
        :param metric:
        :param n_workers:
        :param save:
        :return:
        """
        if self.distMat:
            return self.distMat

        if not metric:
            metric = self.metric
        if self.waveletCoeff is None or self.nodeLayers is None:
            self.initialize()

        print("Start parallelly calculate structural distance, n_workers={}.\n".format(n_workers))
        self.distMat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        pool = multiprocessing.Pool(n_workers)
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

        if save:
            #　construct new graph using structural distance
            save_distance_edgelist("../distance/HSD_{}_{}_scale{}_hop{}.edgelist"\
                .format(self.graph_name, self.metric, self.scale, self.hop), self.nodes, self.distMat)

            save_distance_csv("../distance/HSD_{}_{}_scale{}_hop{}.csv".format(
                self.graph_name, self.metric, self.scale, self.hop), self.nodes, self.distMat)
            print("Save distance information successfully.\n")
        print("done.")
        return self.distMat


    def _calculateDistanceWorker(self, start, metric):
        dist = dict()
        node = self.nodes[start]
        rings1 = self.nodeLayers[node]
        for idx in range(start+1, self.n_nodes):
            _neighbor = self.nodes[idx]
            rings2 = self.nodeLayers[_neighbor]
            distance = 0.0
            for hop in range(self.hop):
                r1, r2 = rings1[hop], rings2[hop]
                p, q = [], []
                for _neighbor in r1:
                    _idx = self.node2idx[_neighbor]
                    p.append(self.waveletCoeff[start, _idx])
                for _neighbor in r2:
                    _idx = self.node2idx[_neighbor]
                    q.append(self.waveletCoeff[idx, _idx])
                distance += calculate_distance(p, q, metric)
            dist[idx] = distance

        return dist


    def multi_scales_wavelet(self):
        eigenvalues = self.wavelet.e
        print("min eigenvalues: {} \n max eigenvalues: {}".format(min(eigenvalues), max(eigenvalues)))
        scales = np.exp(np.linspace(np.log(0.01), np.log(max(eigenvalues)), 200))
        print("selected scales: [{}]".format(",".join(map(str, scales))))

        data = dict()  # 各尺度下的数据汇总
        hierarchy = self.constructNodeLayers_BFS()
        for scale in tqdm(scales):
            coeffs = self.wavelet.cal_all_wavelet_coeffs(scale)
            state = dict()
            for idx, node in enumerate(self.nodes):
                p = [coeffs[idx, idx]]
                for k_hop in range(1, self.hop + 1):
                    k_hop_neighbors = hierarchy[idx].get(k_hop, [])
                    if not k_hop_neighbors:
                        k_hop_sum = 0.0
                    else:
                        k_hop_sum = np.sum([coeffs[idx][neighbor] for neighbor in k_hop_neighbors])
                    p.append(k_hop_sum)

                if math.isclose(1.0 - sum(p), 0.0, abs_tol=1e-6):
                    p.append(0.0)
                else:
                    p.append(1.0 - sum(p))
                state[idx] = p
            data[scale] = state

        # 各尺度下分别计算距离
        dist_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=np.float)
        for scale, state in data.items():
            for idx1 in range(self.n_nodes):
                for idx2 in range(idx1 + 1, self.n_nodes):
                    p, q = state[idx1], state[idx2]
                    d = calculate_distance(p, q, metric=self.metric)
                    dist_mat[idx1, idx2] += d
                    dist_mat[idx2, idx1] += d
        dist_mat = dist_mat / len(scales)  # 取均值

        save_distance_csv("../distance/HSD_multi_{}_{}_hop{}.csv".format(
            self.graph_name, self.metric, self.hop), self.nodes, dist_mat)
        save_distance_edgelist("../distance/HSD_multi_{}_{}_hop{}.edgelist".format(
            self.graph_name, self.metric, self.hop), self.nodes, dist_mat)

        return dist_mat


