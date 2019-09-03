import networkx as nx
import numpy as np
import pygsp
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import math


def wasserstein_distance(p, q, dual=False):
    from scipy.optimize import linprog
    """
    计算两个不等长分布之间的wasserstein距离，用0补齐
    :param p:
    :param q:
    :return:
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
    assert len(p) != len(q), "Length of p({}) must be equal to length of q({})".format(len(p), len(q))
    if not math.isclose(sum(p), 1.0) or not math.isclose(sum(q), 1.0):
        raise ArithmeticError("The sum of probability distribution function is not 1.0")


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

def L1_2_distance(p, q, order):
    """
    计算两个分布之间的 L1 or L2 距离。
    """
    _check_prob_distri(p, q)
    if order == 1:
        return np.sum(np.abs(p - q))
    elif order == 2:
        return np.sum(np.square(p - q))

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
        return L1_2_distance(p, q, 1)
    elif metric == 'l2':
        return L1_2_distance(p, q, 2)
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


class GraphWave:

    def __init__(self, graph, settings):

        self.settings = settings
        self.graph = graph
        self.n_nodes = nx.number_of_nodes(graph)
        self.nodes = list(nx.nodes(graph))

        self.G = pygsp.graphs.Graph(nx.adjacency_matrix(graph))
        self.G.compute_fourier_basis()
        np.set_printoptions(suppress=True, precision=5)
        self.eigenvectors = self.G.U
        self.eigenvalues = self.G.e / max(self.G.e)

        self.sample_points = list(map(lambda x: x * self.settings.step_size, range(0, self.settings.sample_number)))


    def _exact_embedding(self):
        pass

    def _approx_embedding(self, mode="cha"):
        """
        Given the Chebyshev polynomial, graph the approximate embedding is calculated.
        """
        self.G.estimate_lmax()
        self.heat_filter = pygsp.filters.Heat(self.G, tau=[self.settings.heat_coefficient])
        self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter, m=self.settings.approximation)

        self.embeddings = dict()
        for node_idx in tqdm(range(self.n_nodes)):
            impulse = np.zeros((self.n_nodes))
            impulse[node_idx] = 1
            wavelet_coefficietns = pygsp.filters.approximations.cheby_op(self.G, self.chebyshev, impulse)
            self.embeddings[self.nodes[node_idx]] = self._cal_embedding(wavelet_coefficietns, mode)
        return self.embeddings


    def _check_node(self, node_idx):
        if node_idx < 0 or node_idx > self.n_nodes:
            raise ValueError("node_idx is not valid: node_idx{}".format(node_idx))


    def _check_wavelet_coefficients(self, coefficients):
        if len(coefficients) != self.n_nodes:
            raise TypeError("The number of coefficients should be {}, error:{}".format(self.n_nodes, len(coefficients)))


    def _calculate_node_coefficients(self, node_idx, scale):
        impulse = np.zeros(shape=(self.n_nodes))
        impulse[node_idx] = 1
        coefficients = np.dot(np.dot(np.dot(self.eigenvectors, np.diag(np.exp(-scale * self.eigenvalues))),
                             np.transpose(self.eigenvectors)), impulse)
        return coefficients


    def _cal_embedding(self, wavelet_coefficients, mode="cha"):
        """
        用小波系数去计算嵌入。
        :param wavelet_coefficients:
        :param sample_points:
        :param mode:
        :return:
        """
        if mode not in ["cha", "mog", "mo"]:
            raise ValueError("The embedding mode:{} is not supported.".format(mode))
        embedding = []
        for t in self.sample_points:
            if mode == "cha":
                value = np.mean(np.exp(1j * wavelet_coefficients * t))
                embedding.append(value.real)
                embedding.append(value.imag)
            elif mode == "mog":
                value = np.mean(np.exp(wavelet_coefficients * t))
                embedding.append(value)
            elif mode == "mo":
                value = np.mean(wavelet_coefficients ** t)
                embedding.append(value)
        return embedding


    def single_scale_embedding(self, heat_coefficient, mode="cha"):
        """
        :param heat_coefficient:  parameter scale.
        :param mode: characteristic function, moment generating function, moment
        :return:
        """
        if mode not in ["cha", "mog", "mo"]:
            raise ValueError("The embedding mode:{} is not supported.".format(mode))
        #print(heat_coefficient, mode)
        sample_points = list(map(lambda x: x * self.settings.step_size, range(0, self.settings.sample_number)))
        self.embeddings = []
        for node_idx in tqdm(range(self.n_nodes)):
            node_coeff = self._calculate_node_coefficients(node_idx, heat_coefficient)
            #print(node_idx, self.nodes[node_idx], node_coeff)
            embedding = []
            for t in sample_points:
                if mode == "cha":
                    value = np.mean(np.exp(1j * node_coeff * t))
                    embedding.append(value.real)
                    embedding.append(value.imag)
                elif mode == "mog":
                    value = np.mean(np.exp(node_coeff * t))
                    embedding.append(value)
                elif mode == "mo":
                    value = np.mean(node_coeff ** t)
                    embedding.append(value)
            #self.embeddings[node_idx] = np.array(embedding)
            embedding = np.array(embedding)
            self.embeddings.append(np.array(embedding))
        return np.array(self.embeddings)


    def embedding_similarity(self, dataset, scale):
        embedding = self.single_scale_embedding(scale)
        fout = open("G:\pyworkspace\graph-embedding\out\\{}_{}_graphwave.txt".format(dataset, scale), mode="w+", encoding="utf-8")
        for idx1 in range(self.n_nodes):
            e1 = np.array(embedding[idx1])
            for idx2 in range(self.n_nodes):
                e2 = np.array(embedding[idx2])
                s = np.sqrt(np.sum((e1 - e2)**2))
                fout.write("{} {} {}\n".format(self.nodes[idx1], self.nodes[idx2], s))
        fout.close()


    def multi_scale_embedding(self, scales, mode="cha"):
        """
        多尺度嵌入
        :param scales: [scale_1, scale_2, ...]
        :param mode:  embedding mode.
        :return:
        """
        if mode not in ["cha", "mog", "mo"]:
            raise ValueError("The embedding mode:{} is not supported.".format(mode))

        multi_embeddings = dict()
        for i in tqdm(range(len(scales))):
            multi_embeddings[scales[i]] = self.single_scale_embedding(scales[i], mode)
        return multi_embeddings


    def dev_cal_all_wavelet_coeffs(self, scale):
        coeffs = []
        for node_idx in range(self.n_nodes):
            _coeff = self._calculate_node_coefficients(node_idx, scale)
            coeffs.append(_coeff)
        return np.array(coeffs, dtype=np.float32)


    def dev_coeff_research(self, scale):
        """
        The method is used to analyse wavelet coefficients.
        :param scale: parameter : heat coefficient
        :return:
        """
        coeffs = []
        for node_idx in range(self.n_nodes):
            _coeff = self._calculate_node_coefficients(node_idx, scale)
            coeffs.append(_coeff)
        path_len_coeffs = dict()
        for node_idx1 in range(self.n_nodes):
            for node_idx2 in range(self.n_nodes):
                shortest_path_len = nx.dijkstra_path_length(self.graph, self.n_nodes[node_idx1], self.n_nodes[node_idx2])
                path_len_coeffs[shortest_path_len] = path_len_coeffs.get(shortest_path_len, []) + [coeffs[node_idx1][node_idx2]]
        path_average_coeff = dict()
        for _len, _coeffs in path_len_coeffs.items():
            path_average_coeff[_len] = np.mean(np.array(_coeffs))

        return path_average_coeff, path_average_coeff


    def dev_wavlet_KL(self, data_name, scale, save=False, fig=False, top=True):
        """
        研究不同节点的小波系数数组，分析其KL散度，找出分布相似的节点。
        :param data_name: 数据集名字
        :param scale: 嵌入尺度
        :param save: KL散度是否存下来
        :param fig: KL散度分布图是否存下来
        :param top: 找出前几个分布最接近的，即KL散度最小的那些点。
        :return:
        """

        coeffs = []
        for node_idx in tqdm(range(self.n_nodes)):
            wavelet_coeff = self._calculate_node_coefficients(node_idx, scale)
            coeffs.append(np.sort(wavelet_coeff))

        # 计算完小波系数后，排序后计算KL散度
        _KL = np.zeros(shape=(self.n_nodes, self.n_nodes), dtype=float)
        for _idx1 in range(self.n_nodes):
            for _idx2 in range(self.n_nodes):
                _KL[_idx1, _idx2] = np.sum(coeffs[_idx1] * np.log((coeffs[_idx1]+ 2.0**(-15)) / (coeffs[_idx2] + 2.0**(-15))))
                # print(self.nodes[_idx1], self.nodes[_idx2], _KL[_idx1, _idx2])
        # 对称KL散度
        _symm_KL = np.zeros(shape=(self.n_nodes, self.n_nodes), dtype=float)
        for _idx1 in range(self.n_nodes):
            for _idx2 in range(_idx1 + 1, self.n_nodes):
                _symm_KL[_idx1, _idx2] = _symm_KL[_idx2, _idx1] = (_KL[_idx1, _idx2] + _KL[_idx2, _idx1])/2.0

        if save:
            # 以csv格式输出
            df = pd.DataFrame(_KL, index=self.nodes, columns=self.nodes)
            df.to_csv("G:\pyworkspace\graph-embedding\out\\bell_KL.csv", mode="w+")

        if fig:
            self.__save_data_figure(_KL, data_name, scale)

        if top:
            fout = open("../../out/{}-{}-top.txt".format(data_name, scale), mode="w+", encoding="utf-8")
            for _idx1 in range(self.n_nodes):
                t = []
                for _idx2 in range(self.n_nodes):
                    t.append((self.nodes[_idx2], _KL[_idx1, _idx2]))
                t.sort(key=lambda a:a[1])
                res = [t[i][0]+" "+str(t[i][1]) for i in range(10)]
                fout.write(self.nodes[_idx1] + ":" + ", ".join(res) + "\n")
            fout.close()


    def __save_data_figure(self, data, data_name, scale):
        import matplotlib.pyplot as plt

        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']

        xs = [i for i in range(self.n_nodes)]
        for _idx1 in range(self.n_nodes):
            plt.figure()
            _name = self.nodes[_idx1]
            plt.title("节点{}KL散度图".format(_name))
            plt.xlabel("nodes")
            plt.ylabel("KL-divergence")
            plt.scatter(xs, data[_idx1, :])
            for _idx2 in range(self.n_nodes):
                plt.text(_idx2, data[_idx1, _idx2], self.nodes[_idx2], ha='center', va='bottom', fontsize=7)
            plt.savefig(u"G:\KL散度分布图\{}\s{}\{}.png".format(data_name, scale, _name))
            plt.close()
        return

    """
    def dev_coeff_autoencoder(self, scale):
        import tensorflow as tf

        coeffs = np.zeros(shape=(self.n_nodes, self.n_nodes))
        for _idx in range(self.n_nodes):
            coeffs[_idx] = self._calculate_node_coefficients(_idx, scale)

        n_inputs = self.n_nodes
        n_hidden1 = 256
        n_hidden2 = 128

        X = tf.placeholder(tf.float32, shape=[None, n_inputs])

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_inputs, n_hidden1])),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden2, n_hidden1])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden1, n_inputs])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden1])),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden2])),
            'decoder_b1': tf.Variable(tf.random_normal([n_hidden1])),
            'decoder_b2': tf.Variable(tf.random_normal([n_inputs])),
        }

        def encoder(X, weights, biases):
            layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['encoder_h1']), biases['encoder_b1']))
            layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['encoder_h2']), biases['encoder_b2']))
            return layer2

        def decoder(X, weights, biases):
            layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights['decoder_h1']), biases['decoder_b1']))
            layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, weights['decoder_h2']), biases['decoder_b2']))
            return layer2

        encoder_op = encoder(X, weights, biases)
        decoder_op = decoder(encoder_op, weights, biases)

        output = decoder_op
        input = X
        order_1 = tf.constant()

        moment1 = tf.reduce_mean

        learning_rate = 0.01
        batch_size = 300
        n_epochs = 10

        cost = tf.reduce_mean(tf.pow(calculate_moment(input, 10) - calculate_moment(output, 10), 2))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(n_epochs):
                if epoch % 10 == 0:
                    print("Epoch", epoch, "Cost=", cost.eval())
                sess.run([optimizer, cost], feed_dict={X:coeffs})
    """


    def save_coeff_fig(self, dataset, scale, num):
        coeff = self.dev_cal_all_wavelet_coeffs(scale)
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']
        xs = [i for i in range(num)]
        for i in tqdm(range(len(coeff))):
            _coeff = np.array(coeff[i])
            moment2 = np.mean(_coeff ** 2)
            moment3 = np.mean(_coeff ** 3)
            moment4 = np.mean(_coeff ** 4)
            moment5 = np.mean(_coeff ** 5)
            moment6 = np.mean(_coeff ** 6)
            plt.figure()
            plt.xlabel("node")
            plt.ylabel("coefficient value")
            plt.title("{} 小波系数分布".format(self.nodes[i]))
            x = np.argsort(-_coeff)[:num]
            ys = -np.sort(-_coeff)[:num]
            plt.plot(xs, ys)
            plt.text(len(ys), ys[0], "m-2 : %.3e \n\n m-3 : %.3e \n\n m-4 : %.3e \n\n m-5 : %.3e \n\n m-6 : %.3e " % (moment2, moment3, moment4, moment5, moment6),
                     fontsize=10, verticalalignment="top", horizontalalignment="right")
            for _x, _y in zip(xs, ys):
                plt.text(_x, _y, self.nodes[x[_x]] + "\n" + str(round(_y, 5)), ha='center', va='bottom', fontsize=7)
            #plt.savefig(u"G:\小波系数分布图\{}\\noweight\s{}\\{}.png".format(dataset, scale, self.nodes[i]))
            plt.savefig(u"G:\小波系数分布图\{}\s{}\\{}.png".format(dataset, scale, self.nodes[i]))
            plt.close()


    def dist_measure(self, scale, method="L1", save_path=None):

        np.set_printoptions(suppress=True, precision=5)
        """
        计算小波系数的相似性，按照hop数，以源节点为中心的多重环，每个环上都有一些节点。
        两个不同的源节点，计算各层环的相似性，然后累加和，最后作为整体的相似性。
        环上的节点个数可能不同，需要对齐，用0填充。
        计算距离时，可以有多种选择：绝对值距离，欧式距离，Wasserstein距离等等，多多尝试一下。
        :param scale:
        :param mode:
        :return:
        """
        coef = self.dev_cal_all_wavelet_coeffs(scale)

        """
        每个节点都有一个dict，{k-hop: [coef]}
        搞个数组，存起来这些dict。
        """
        dict_list = []
        for node1 in range(self.n_nodes):
            rings = defaultdict(list)
            for node2 in range(self.n_nodes):
                dist = nx.dijkstra_path_length(self.graph, self.nodes[node1], self.nodes[node2])
                rings[dist] += [coef[node1, node2]]
            dict_list.append(rings)

        """
        距离
        """
        dists = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for idx1 in tqdm(range(self.n_nodes)):
            for idx2 in tqdm(range(idx1+1, self.n_nodes)):
                rings1, rings2 = dict_list[idx1], dict_list[idx2]
                maxHop = min(max(len(rings1), len(rings2)), 5)
                res = 0.0
                for hop in range(1, maxHop+1):
                    t1, t2 = rings1[hop], rings2[hop]
                    if not t1 and not t2:
                        break

                    if method == "guass":
                        # 高斯分布不需要对齐
                        res += self._gauss_distance(t1, t2)
                    else:
                        # 对齐
                        length = max(len(t1), len(t2))
                        if method == "L3":
                            t1 = np.sort(np.array(t1 + [-0.5] * (length - len(t1))))
                            t2 = np.sort(np.array(t2 + [-0.5] * (length - len(t2))))
                            res += np.sum(np.abs(t1 - t2))
                        else:
                            t1 = np.sort(np.array(t1 + [0.0] * (length - len(t1))))
                            t2 = np.sort(np.array(t2 + [0.0] * (length - len(t2))))
                            if method == "L1":
                                res += np.sum(np.abs(t1 - t2))
                            elif method == "L2":
                                res += np.sum((t1 - t2) ** 2)
                            elif method == "wasserstein":
                                res += self._wasserstein_distance(t1, t2, dual=False)

                if res < math.exp(-10):
                    res = 0.0
                dists[idx1, idx2] = dists[idx2, idx1] = 1 - res

        if save_path:
            fout = open(save_path, mode="w+", encoding="utf8")
            for i in range(len(dists)):
                for j in range(i+1, len(dists)):
                    fout.write("{} {} {}\n".format(self.nodes[i], self.nodes[j], dists[i, j]))
        return dists


    def _gauss_distance(self, p, q):
        p, q = np.array(p), np.array(q)
        u1 = np.mean(p)
        u2 = np.mean(q)

        sigma1 = np.sqrt(np.mean((p-u1)**2))
        sigma2 = np.sqrt(np.mean((q-u2)**2))

        d = (u1 - u2) ** 2 + (sigma1 + sigma2 - 2 * np.sqrt(sigma1 * sigma2))
        print(p, q, d)
        return d


    def fb1_dist(self, scale):
        np.set_printoptions(suppress=True, precision=5)
        plt.rcParams['font.family'] = ['sans-serif']
        plt.rcParams['font.sans-serif'] = ['SimHei']

        def f(node1, node2, k):
            """
            得到两节点的k层环上的对齐向量
            """
            t1, t2 = node1[k], node2[k]
            length = max(len(t1), len(t2))
            res1 = -np.sort(-np.array(t1 + [0.0] * (length - len(t1))))
            res2 = -np.sort(-np.array(t2 + [0.0] * (length - len(t2))))
            return list(res1), list(res2)

        coef = self.dev_cal_all_wavelet_coeffs(scale)
        dict_rings = dict()
        for node1 in range(self.n_nodes):
            rings = defaultdict(list)
            for node2 in range(self.n_nodes):
                dist = nx.dijkstra_path_length(self.graph, self.nodes[node1], self.nodes[node2])
                if dist > 3:
                    continue
                rings[dist] += [coef[node1, node2]]
            dict_rings[self.nodes[node1]] = rings

        node1 = dict_rings['13']
        node2 = dict_rings['14']
        maxHop = min(max(len(node1), len(node2)), 5)
        print(maxHop)
        print(node1)
        print(node2)
        ring1, ring2 = [], []
        for hop in range(1, maxHop):
            tmp1, tmp2 = f(node1, node2, hop)
            ring1.extend(tmp1)
            ring2.extend(tmp2)
        fig, ax = plt.subplots()
        x = [i for i in range(len(ring1))]
        y1, y2 = np.array(ring1), np.array(ring2)
        rects1 = plt.bar(left=[i - 0.2 for i in x], height=[round(i, 3) for i in y1], width=0.4, alpha=0.8, color='#DC5712', label="node13")
        rects2 = plt.bar(left=[i + 0.2 for i in x], height=[round(i, 3) for i in y2], width=0.4, color='#87CEEB', label="node14")
        plt.xticks(range(0, len(x), 1))
        ax.set_xticklabels(('1', '1', '2', '2', '3','3','3'))
        #ax.set_xticklabels(('1', '1', '2', '2', '2','2','2','2','2', '3','3','3','3','3','3'))

        plt.ylim(0, 0.2)  # y轴取值范围
        plt.ylabel("小波系数")
        plt.legend()
        for rect in rects1:
            height = rect.get_height()
            #plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
        for rect in rects2:
            height = rect.get_height()
            #plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), ha="center", va="bottom")
        plt.show()


































