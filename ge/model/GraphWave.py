# -*- coding:utf-8 -*-
from collections import defaultdict
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy import sparse
import networkx as nx
from tqdm import tqdm
import pygsp

from ge.utils.util import compute_cheb_coeff_basis, build_node_idx_map
from .EmbeddingMixin import EmbeddingMixin
from ge.utils.distance import calc_pq_distance

np.set_printoptions(suppress=True, precision=5)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
import multiprocessing as mp


class GraphWave(EmbeddingMixin):

    def __init__(self, graph, heat_coefficient=5.0, sample_number=16, step_size=20.0):
        super().__init__()
        self.name = "GraphWave: Learning Structural Node Embeddings via DiffusionWavelets."
        self.graph = graph
        self.heat_coefficient = heat_coefficient
        self.n_nodes = nx.number_of_nodes(graph)
        self.nodes = list(nx.nodes(graph))
        self.A = nx.adjacency_matrix(graph)
        #self.L = laplacian(self.A)
        #self.L = nx.normalized_laplacian_matrix(self.graph) # 正则拉普拉斯矩阵
        self.L = nx.laplacian_matrix(self.graph)
        self._e, self._u = np.linalg.eigh(self.L.toarray())
        _, self.node2idx = build_node_idx_map(self.graph)
        self.sample_points = list(map(lambda x: x * step_size, range(0, sample_number)))
        self.embeddings = None


    def train(self):
        pass

    """
    heat = {i: sc.sparse.csc_matrix((n_nodes, n_nodes)) for i in range(n_filters) }
        monome = {0: sc.sparse.eye(n_nodes), 1: lap - sc.sparse.eye(n_nodes)}
        for k in range(2, order + 1):
             monome[k] = 2 * (lap - sc.sparse.eye(n_nodes)).dot(monome[k-1]) - monome[k - 2]
        for i in range(n_filters):
            coeffs = compute_cheb_coeff_basis(taus[i], order)
            heat[i] = sc.sum([ coeffs[k] * monome[k] for k in range(0, order + 1)])
            temp = thres(heat[i].A) # cleans up the small coefficients
            heat[i] = sc.sparse.csc_matrix(temp)"""


    def calc_wavelet_coeff_chebyshev(self, scale, order):
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
        for idx in tqdm(range(self.n_nodes)):
            impulse = np.zeros(self.n_nodes, dtype=np.float)
            impulse[idx] = 1.0
            coeff = pygsp.filters.approximations.cheby_op(G, chebyshev, impulse)
        #    self.embeddings[self.nodes[node_idx]] = self._calc_embedding(wavelet_coefficietns, mode)
            wavelet_coeffs.append(coeff)

        return np.asarray(wavelet_coeffs)


    def _check_node(self, node_idx):
        """
        检验节点标号是否有效
        """
        if node_idx < 0 or node_idx >= self.n_nodes:
            raise ValueError("node_idx is not valid: node_idx{}".format(node_idx))


    def _check_wavelet_coefficients(self, coefficients):
        """
        检验小波系数是否有效
        """
        if len(coefficients) != self.n_nodes:
            raise TypeError("The number of coefficients should be {}, error:{}".format(self.n_nodes, len(coefficients)))


    def _calc_node_coefficients(self, node_idx, scale):
        """
        计算单个节点的小波系数
        :param node_idx: 节点标号，int
        :param scale: 热系数，即尺度，float
        :return: 该尺度下的该节点对应的小波系数，ndarray(n, 1)
        """
        impulse = np.zeros(self.n_nodes)
        impulse[node_idx] = 1
        coefficients = np.dot(np.dot(np.dot(self._u, np.diag(np.exp(-scale * self._e))),
                             np.transpose(self._u)), impulse)
        return coefficients


    def _calc_embedding(self, wavelet_coefficients, mode="cha"):
        """
        利用单个节点的小波系数去计算嵌入向量。
        :param wavelet_coefficients: 小波系数
        :param mode: 计算模式，特征函数 or 矩母函数 or k阶矩(和采样点个数有关)
        :return: 嵌入向量，ndarray
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
                # 计算小波系数的第i阶矩，不过其和为1，高阶矩会逼近0，失去效果。
                value = np.mean(wavelet_coefficients ** i)
                embedding.append(value)
        return np.array(embedding)


    def single_scale_embedding(self, scale=None, mode="cha"):
        """
        在单一尺度下计算嵌入向量。
        :param scale: 热系数，即尺度，float类型，默认为热系数参数值，但仍可以临时指定。
        :param mode: characteristic function, moment generating function, moment
        :return: 该尺度下的所有节点对应的嵌入向量。
        """
        if mode not in ["cha", "mog", "mo"]:
            raise ValueError("The embedding mode:{} is not supported.".format(mode))
        if not scale:
            scale = self.heat_coefficient

        logging.info("Start calculate single scale={} embedding， mode={}".format(scale, mode))
        self.embeddings = {}
        for node_idx in tqdm(range(self.n_nodes)):
            node_wave_coeff = self._calc_node_coefficients(node_idx, scale)
            node_embedding = self._calc_embedding(node_wave_coeff, mode)
            self.embeddings[self.nodes[node_idx]] = node_embedding
        return self.embeddings


    def multi_scale_embedding(self, scales, mode="cha"):
        """
        多尺度嵌入，对于输入的各尺度都进行一次单尺度嵌入。
        :param scales: [scale_1, scale_2, ...]
        :param mode:  embedding mode.
        :return: 每个尺度对应着一组嵌入向量， dict
        """
        multi_embeddings = dict()
        for i in tqdm(range(len(scales))):
            multi_embeddings[scales[i]] = self.single_scale_embedding(scales[i], mode)
        return multi_embeddings


    def cal_all_wavelet_coeffs(self, scale):
        """
        计算某尺度下的小波系数，以供后续针对小波系数本身进行研究。
        :param scale: 尺度参数，即heat coefficient, float
        :return: 小波系数矩阵，shape=(n, n), ndarray
        """
        #print("Start calculate wavelet coefficients.\n")
        coeff_mat = []
        for node_idx in range(self.n_nodes):
            coeff = self._calc_node_coefficients(node_idx, scale)
            coeff_mat.append(coeff)
        #print("calculate wavelet coefficients done. \n")
        return np.array(coeff_mat, dtype=np.float32)


    def get_nodes_layers(self):
        """
        根据节点间的最短路径，将节点局部邻域进行层次划分，以节点为中心的嵌套环状结构，其他节点分布在对应的环上。
        :return: dict(dict()), 嵌套字典结构，第一册key为节点，第二层key为距离。
        """
        res = dict()
        for idx1 in range(self.n_nodes):
            rings = defaultdict(list)
            for idx2 in range(self.n_nodes):
                shortest_path_length = nx.dijkstra_path_length(self.graph, self.nodes[idx1], self.nodes[idx2])
                rings[shortest_path_length].append(idx2)
            res[idx1] = rings

        return res


    def get_nodes_layers_bfs(self, max_hop=5):
        """
        根据节点间的最短路径，将节点局部邻域进行层次划分，以节点为中心的嵌套环状结构，其他节点分布在对应的环上。
        :return: dict(dict()), 嵌套字典结构，第一册key为节点，第二层key为距离。
        """
        #print("Start compute node layers. \n")
        res = dict()
        for idx in range(self.n_nodes):
            rings = defaultdict(list)
            origin = self.nodes[idx]
            visited = [origin]
            neibors = nx.neighbors(self.graph, origin)
            queue = list(neibors)
            visited.extend(queue)
            hop = 0   # 因为小波系数在源节点也有值，所以源节点也当做独立的一环参与计算。
            while queue and hop < max_hop:
                cur_layer_nodes = len(queue)
                for _ in range(cur_layer_nodes):
                    _node = queue.pop(0)
                    rings[hop].append(self.node2idx[_node])
                    next_hop_neibors = list(nx.neighbors(self.graph, _node))
                    for _neibor in next_hop_neibors:
                        if _neibor not in visited:
                            queue.append(_neibor)
                            visited.append(_neibor)
                hop += 1
            res[idx] = rings
        #print("Compute node layers done. \n")
        return res


    def calc_wavelet_similarity(self, coeff_mat, method="l1", layers=5, normalized=True, save_path=None):
        """
        计算节点间小波系数的相似性，首先计算出各层的相似性，然后累加求和。
        :param coeff_mat: 小波系数矩阵
        :param method: 相似性衡量标准
        :param layers: 广搜的最大层数
        :param normalized: 是否正则化
        :param save_path: 将计算得到的相似度以csv文件保存
        :return: 相似度矩阵
        """
        #nodes_layers = self.get_nodes_layers()
        nodes_layers = self.get_nodes_layers_bfs(layers)
        method = str.lower(method)
        similarity_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for idx1 in range(self.n_nodes):
            for idx2 in range(idx1, self.n_nodes):
                rings1, rings2 = nodes_layers[idx1], nodes_layers[idx2]
                maxHop = min(max(len(rings1), len(rings2)), 5) + 1
                dist = 0.0
                for hop in range(1, maxHop):
                    # 取出同一层的环
                    ring1, ring2 = rings1[hop], rings2[hop]
                    p, q = [], []
                    for node in ring1:
                        p.append(coeff_mat[idx1, node])
                    for node in ring2:
                        q.append(coeff_mat[idx2, node])
                    if not (p or q):
                        break
                    dist += calc_pq_distance(p, q, method, normalized=normalized)

                #求出距离后，取倒数，用来衡量相似性，但是由于小波系数都很小，取倒数可能会导致数量级爆炸，求其对数
                #similarity_mat[idx1, idx2] = similarity_mat[idx2, idx1] = math.log(min(1.0 / dist, 1e9), math.e)
                similarity_mat[idx1, idx2] = similarity_mat[idx2, idx1] = (1.0 / dist) if dist > 1e-3 else 1e3

        if save_path:
            """
            df = pd.DataFrame(data=similarity_mat, index=self.nodes, columns=self.nodes)
            df.to_csv(save_path, mode="w+")
            """
            with open(save_path, mode='w+', encoding='utf-8') as fout:
                for idx1 in range(self.n_nodes):
                    node1 = self.nodes[idx1]
                    for idx2 in range(idx1 + 1, self.n_nodes):
                        node2 = self.nodes[idx2]
                        fout.write("{} {} {}\n".format(node1, node2, similarity_mat[idx1, idx2]))

        return similarity_mat


    def parallel_calc_similarity(self, coeff_mat, metric="l1", layers=5, workers=5, save_path=None):
        print("Start parallelly calculate similarity......")
        nodes_layers = self.get_nodes_layers_bfs(layers)
        metric = str.lower(metric)
        similarity_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        pool = mp.Pool(workers)
        result = {}

        for idx in range(self.n_nodes):
            _res = pool.apply_async(_worker, args=(coeff_mat, idx, metric, nodes_layers, ))
            result[idx] = _res

        pool.close()
        pool.join()

        for idx, _res in result.items():
            result[idx] = _res.get()

        for idx1, info in result.items():
            for idx2, similarity in info.items():
                similarity_mat[idx1, idx2] = similarity_mat[idx2, idx1] = similarity

        if save_path:
            """
            df = pd.DataFrame(data=similarity_mat, index=self.nodes, columns=self.nodes)
            df.to_csv(save_path, mode="w+")
            """
            with open(save_path, mode='w+', encoding='utf-8') as fout:
                for idx1 in range(self.n_nodes):
                    node1 = self.nodes[idx1]
                    for idx2 in range(idx1 + 1, self.n_nodes):
                        node2 = self.nodes[idx2]
                        fout.write("{} {} {}\n".format(node1, node2, similarity_mat[idx1, idx2]))

        return similarity_mat



def _worker(coeff_mat, idx1, metric="l1", nodes_layers=None):
    similarity = {}
    for idx2 in range(idx1+1, len(coeff_mat)):
        rings1, rings2 = nodes_layers[idx1], nodes_layers[idx2]
        maxHop = min(max(len(rings1), len(rings2)), 5) + 1
        dist = 0.0
        for hop in range(0, maxHop):  # 1 -> 0, 原来的版本不将源节点参与计算，现在从0开始计hop。
            ring1, ring2 = rings1[hop], rings2[hop]
            p, q = [], []
            for node in ring1:
                p.append(coeff_mat[idx1, node])
            for node in ring2:
                q.append(coeff_mat[idx2, node])
            if not (p or q):
                break
            dist += calc_pq_distance(p, q, metric, normalized=False)
        similarity[idx2] =  (1.0 / dist) if dist > 1e-4 else 1e4  # 在这里定义距离和相似度反比。
    #queue.put((idx1, similarity))
    return similarity


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









































