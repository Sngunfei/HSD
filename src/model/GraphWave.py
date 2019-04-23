import networkx as nx
import numpy as np
import pygsp
from tqdm import tqdm
import pandas as pd

class GraphWave:

    def __init__(self, graph, settings):

        self.settings = settings
        self.graph = graph
        self.n_nodes = nx.number_of_nodes(graph)
        self.nodes = list(nx.nodes(graph))

        self.G = pygsp.graphs.Graph(nx.adjacency_matrix(graph))
        self.G.compute_fourier_basis()

        self.eigenvectors = self.G.U
        self.eigenvalues = self.G.e

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

        sample_points = list(map(lambda x: x * self.settings.step_size, range(0, self.settings.sample_number)))
        self.embeddings = dict()
        for node_idx in tqdm(range(self.n_nodes)):
            node_coeff = self._calculate_node_coefficients(node_idx, heat_coefficient)
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
            self.embeddings[self.nodes[node_idx]] = np.array(embedding)
        return self.embeddings


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



















