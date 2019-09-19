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

np.set_printoptions(suppress=True, precision=5)
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


class GraphWave:

    def __init__(self, graph, settings):
        self.settings = settings
        self.graph = graph
        self.n_nodes = nx.number_of_nodes(graph)
        self.nodes = list(nx.nodes(graph))
        self.A = nx.adjacency_matrix(graph)
        #self.L = laplacian(self.A)
        #self.L = nx.normalized_laplacian_matrix(self.graph) # 正则拉普拉斯矩阵
        self.L = nx.laplacian_matrix(self.graph)
        self._e, self._u = np.linalg.eigh(self.L.toarray())
        _, self.node2idx = build_node_idx_map(self.graph)
        """
        self.G = pygsp.graphs.Graph(self.L)
        self.G.compute_fourier_basis()
        self.eigenvectors = self.G.U
        self.eigenvalues = self.G.e / max(self.G.e)
        """
        self.sample_points = list(map(lambda x: x * self.settings.step_size, range(0, self.settings.sample_number)))
        self.embeddings = None


    def exact_embedding(self):
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

        return wavelet_coeffs


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


    def single_scale_embedding(self, heat_coefficient=None, mode="cha"):
        """
        在单一尺度下计算嵌入向量。
        :param heat_coefficient: 热系数，即尺度，float类型，默认为热系数参数值，但仍可以临时指定。
        :param mode: characteristic function, moment generating function, moment
        :return: 该尺度下的所有节点对应的嵌入向量。
        """
        if mode not in ["cha", "mog", "mo"]:
            raise ValueError("The embedding mode:{} is not supported.".format(mode))
        if not heat_coefficient:
            heat_coefficient = self.settings.heat_coefficient

        logging.info("Start calculate single scale={} embedding， mode={}".format(heat_coefficient, mode))
        self.embeddings = {}
        for node_idx in tqdm(range(self.n_nodes)):
            node_wave_coeff = self._calc_node_coefficients(node_idx, heat_coefficient)
            node_embedding = self._calc_embedding(node_wave_coeff, mode)
            self.embeddings[self.nodes[node_idx]] = node_embedding
        return self.embeddings


    def _cal_embeddings_distance(self, heat_coefficient=None, mode="cha", sample_points=None):
        """
        在嵌入空间中计算各节点对应的embeddings之间的欧式距离，这也是wavelet文中提到的相似度衡量方法。
        :param heat_coefficient: 热系数，即尺度，默认为model中的热系数，可临时指定。float
        :param mode: 求嵌入向量时使用的函数, str
        :param sample_points: 采样点，默认为model中的采样数组，可临时指定，array like。
        :return: 返回各节点在嵌入空间中的欧式距离， ndarray, (n, n)
        """
        """
        为了能够灵活的重复多次的距离计算过程，函数中可以指定不同的参数，每次都重新计算。
        if self.embeddings is not None:
            return self.embeddings
        """
        if not heat_coefficient:
            heat_coefficient = self.settings.heat_coefficient
        if not sample_points:
            sample_points = self.sample_points

        embeddings = self.single_scale_embedding(heat_coefficient, mode)
        distance_matrix = np.empty(self.n_nodes, self.n_nodes)
        for idx1 in range(self.n_nodes):
            vector1 = embeddings[idx1]
            for idx2 in range(idx1, self.n_nodes):
                vector2 = embeddings[idx2]
                l2 = np.sqrt(np.sum((vector1 - vector2) ** 2))
                distance_matrix[idx1, idx2] = distance_matrix[idx2, idx1] = l2
        return distance_matrix


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
        print("Start calculate wavelet coefficients.\n")
        coeff_mat = []
        for node_idx in tqdm(range(self.n_nodes)):
            coeff = self._calc_node_coefficients(node_idx, scale)
            coeff_mat.append(coeff)
        print("calculate wavelet coefficients done. \n")
        return np.array(coeff_mat, dtype=np.float32)


    def dist_coeff_analyse(self, scale):
        """
        分析小波系数和节点间距离的关系，求最短路径，然后取小波系数均值，直观认为是负相关，距离越远，系数越小。
        但是最短路径只能影响系数大小，不能决定。因为热扩散的路径在全图可以认为有无数条（可以循环扩散），两点间的
        小波系数虽然是刻画这两个节点间的结构特征，但还是会受周围节点的影响。
        只取最短路径进行研究。该函数只是提供一个统计意义上的规律，某些距离较远的节点上的系数是可以比近距离的
        系数要大，这是因为某些地方的结构比较复杂，有些比较简单。
        :param scale: 尺度参数，即热扩散系数。
        :return: dict，key = distance，value = [wavelet coefficient mean, variance]
        """
        coeff_mat = self.cal_all_wavelet_coeffs(scale)
        dist_coeff = dict()
        for idx1 in range(self.n_nodes):
            for idx2 in range(self.n_nodes):
                shortest_path_length = nx.dijkstra_path_length(self.graph, self.nodes[idx1], self.nodes[idx2])
                dist_coeff[shortest_path_length] = dist_coeff.get(shortest_path_length, []) + [coeff_mat[idx1][idx2]]
        dist_coeff_info = dict()
        for length, coeffs in dist_coeff.items():
            mean = np.mean(coeffs)
            var = np.var(coeffs)
            dist_coeff_info[length] = [mean, var]

        return dist_coeff_info


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
        print("Start compute node layers. \n")
        res = dict()
        for idx in tqdm(range(self.n_nodes)):
            rings = defaultdict(list)
            origin = self.nodes[idx]
            visited = [origin]
            neibors = nx.neighbors(self.graph, origin)
            queue = list(neibors)
            visited.extend(queue)
            hop = 1
            while queue and hop < max_hop:
                cur_layer_nodes = len(queue)
                for _ in range(cur_layer_nodes):
                    _node = queue.pop(0)
                    rings[hop].append(self.node2idx[_node])
                    next_hop_neibors = list(nx.neighbors(graph, _node))
                    for _neibor in next_hop_neibors:
                        if _neibor not in visited:
                            queue.append(_neibor)
                            visited.append(_neibor)
                hop += 1
            res[idx] = rings
        print("Compute node layers done. \n")
        return res


    def _calc_pq_distance(self, p, q, metric="l1", normalized=False):
        """
        计算两个分布之间的相似性，p和q分别为两个节点，在同一层环上的小波系数分布，可能并不等长，为了突出节点度数
        对于节点结构性质的重要性，我们用 0 将分布对齐。
        :param metric: 相似性度量标准，默认为L1
        :param normalized: 是否将p，q放缩成正常的概率分布，因为p，q分别求和后的结果不一定相等。
        :return: 两个分布之间的相似性
        """
        length = max(len(p), len(q))
        p = p + (length - len(p)) * [0]
        q = q + (length - len(q)) * [0]
        p = np.sort(p)
        q = np.sort(q)

        if normalized:
            p = 1.0 * p / np.sum(p) if np.sum(p) > 0.0 else p
            q = 1.0 * q / np.sum(q) if np.sum(q) > 0.0 else q

        return calc_distance(p, q, metric)


    def calc_wavelet_similarity(self, coeff_mat, method="l1", weight=None, save_path=None):
        """
        计算节点间小波系数的相似性，首先计算出各层的相似性，然后累加求和。
        :param coeff_mat: 小波系数矩阵
        :param method: 相似性衡量标准
        :param save_path: 将计算得到的相似度以csv文件保存
        :return: 相似度矩阵
        """
        #nodes_layers = self.get_nodes_layers()
        nodes_layers = self.get_nodes_layers_bfs(5)
        method = str.lower(method)
        similarity_mat = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for idx1 in tqdm(range(self.n_nodes)):
            for idx2 in tqdm(range(idx1, self.n_nodes)):
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
                    dist += self._calc_pq_distance(p, q, method, normalized=True)

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


def wasserstein_distance(p, q, dual=False):
    """
    优化求解两个分布之间的2阶wasserstein距离
    """
    length = len(p)
    D = np.zeros((length, length))  # d(x, y)

    for i in range(length):
        for j in range(length):
            D[i, j] = np.abs(p[i] - q[j])

    A_r = np.zeros((length, length, length))
    A_t = np.zeros((length, length, length))

    for i in range(length):
        for j in range(length):
            A_r[i, i, j] = 1
            A_t[i, j, i] = 1

    A = np.concatenate((A_r.reshape((length, length ** 2)), A_t.reshape((length, length ** 2))), axis=0)
    b = np.concatenate((p, q), axis=0)
    c = D.reshape((length ** 2))
    if dual:
        opt_res = linprog(-b, A.transpose(), c, bounds=(None, None))
        emd = -opt_res.fun
    else:
        opt_res = linprog(c, A_eq=A, b_eq=b)
        emd = opt_res.fun
    return emd


def _check_prob_distri(p, q):
    """
    验证两个概率分布是否有效，以供后续计算两者相似性。
    """
    if (not isinstance(p, list) and not isinstance(p, np.ndarray)) \
        or (not isinstance(q, list) and not isinstance(q, np.ndarray)):
        raise TypeError("The probability distribution type must be list or ndarray")
    assert len(p) == len(q), "Length of p({}) must be equal to length of q({})".format(len(p), len(q))
    """
    if not math.isclose(sum(p), 1.0) or not math.isclose(sum(q), 1.0):
        raise ArithmeticError("The sum of probability distribution function is not 1.0")
    """

def wasserstein_guass(p, q):
    """
    在欧式空间中计算两个n维高斯分布之间的2阶Wasserstein距离，解析解如下：
        W(u1，u2)^2 = |m1 - m2|^2 + tr(C1 + C2 - 2(C2^{-1/2}C1C2^{1/2})^{1/2})
    其中m1和m2分别为两个分布的均值即中心，C1和C2是对应的协方差矩阵，对称半正定。
    当维度为1时，就退化到两个正态分布之间的距离，公式可写为：
        W(u1, u2)^2 = |m1 - m2|^2 + sqrt((\sigma1 + \sigma2 - 2(\sigma1\sigma2)))
    这时候就能很好的说明W式距离能够刻画分布之间的几何距离：中心点之间的距离 + 分布形状之间的距离
    :param p: First guass distribution.
    :param q: Second guass distribution
    :return: The 2th Wasserstein distance between two Guass.
    """
    _check_prob_distri(p, q)
    m1 = np.mean(p)
    m2 = np.mean(q)
    sigma1 = np.mean(np.square(p - m1))
    sigma2 = np.mean(np.square(q - m2))
    dist = (m1 - m2) ** 2 + sigma1 + sigma2 - 2 * (sigma1 * sigma2) ** 0.5
    return dist


def KL_divergence(p, q, symmetric=False):
    """
    计算两个分布之间的Kullback–Leibler散度，公式如下：
        KL(p | q) = Σplog(p / q)
    注意到KL散度实际上并不属于距离度量，因为不对称，求KL散度的均值可以解决。
    :param p: First probability distribution
    :param q: Second probability distribution
    :param symmetric: if True, calculate symmetric KL divergence
    :return: The KL divergence between p, q
    """
    _check_prob_distri(p, q)
    kl_pq = np.sum(p * np.log(p / q))
    if symmetric:
        kl_qp = np.sum(q * np.log(q / p))
        return (kl_pq + kl_qp) / 2.0
    return kl_pq


def JS_divergence(p, q):
    """
    计算两个分布之间的Jensen–Shannon散度，定义如下：
        JS(P | Q) = [KL(P | M) + KL(Q | M)] / 2， M = （P + Q) / 2
    可以看到JS散度是基于KL散度定义的，更加平滑，而且对称。
    :param p: First probability distribution
    :param q: Second probability distribution
    :return: The JS divergence between p, q
    """
    _check_prob_distri(p, q)
    m = p + q
    js = (KL_divergence(p, m, False) + KL_divergence(q, m, False)) / 2.0
    return js


def L_1_2_distance(p, q, order):
    """
    计算两个分布之间的 L1 or L2 距离。
    """
    _check_prob_distri(p, q)
    if order == 1:
        return np.sum(np.abs(p - q))
    elif order == 2:
        return np.sum(np.square(p - q))


def Dynamic_Time_Warping(p, q):
    """
    动态时间规整，DTW
    :param p:
    :param q:
    :return:
    """
    raise NotImplementedError("Dynamic_Time_Warping is not implemented yet.")


def calc_distance(p, q, metric=None):
    """
    计算两个概率分布之间的距离。
    :param metric: 距离名称, str
    :return: 距离大小，float
    """
    if not metric or not isinstance(metric, str):
        raise TypeError("Need to specify a metric")
    sup_metrics = ['l1', 'l2', 'kl', 'skl', 'js', 'wguass', 'wasser']
    if str.lower(metric) not in sup_metrics:
        raise NotImplementedError("The {} metric is not supported yet.".format(metric))
    metric = str.lower(metric)
    if metric == 'l1':
        return L_1_2_distance(p, q, 1)
    elif metric == 'l2':
        return L_1_2_distance(p, q, 2)
    elif metric == 'kl':
        return KL_divergence(p, q)
    elif metric == 'skl':
        return KL_divergence(p, q, True)
    elif metric == 'js':
        return JS_divergence(p, q)
    elif metric == 'wguass':
        return wasserstein_guass(p, q)
    elif metric == 'wasser':
        return wasserstein_distance(p, q)


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


if __name__ == "__main__":
    from utils.util import dataloader, evaluate_SVC_accuracy, evaluate_LR_accuracy
    from example import parser
    settings = parser.parameter_parser()

    dataset = "europe"
    scale = 15
    metric = 'L1'

    graph, label_dict, n_class = dataloader(dataset, directed=False)
    wave_machine = GraphWave(graph, settings)
    #wavelet_coeff = wave_machine.cal_all_wavelet_coeffs(10)
    #wave_machine.calc_wavelet_similarity(wavelet_coeff, method='L1', save_path="../../similarity/subway_10_L1.csv")
    #approx_wavelet_coeffs = np.asarray(wave_machine.calc_wavelet_coeff_chebyshev(100, 200), dtype=np.float)
    #exact_wavelet_coeffs = np.array(wave_machine.cal_all_wavelet_coeffs(scale))
    #wave_machine.calc_wavelet_similarity(exact_wavelet_coeffs, metric, save_path="../../similarity/{}_{}_{}.csv".format(dataset, scale, metric))
    embeddings_dict = wave_machine.single_scale_embedding(scale)
    embeddings = []
    nodes = []
    labels = []
    for node, embedd in embeddings_dict.items():
        embeddings.append(embedd)
        nodes.append(node)
        labels.append(label_dict[node])
    evaluate_LR_accuracy(embeddings, labels, random_state=42)
    evaluate_SVC_accuracy(embeddings, labels, random_state=42)







































