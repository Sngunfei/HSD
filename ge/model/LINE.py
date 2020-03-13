from __future__ import print_function
import random
import math

import numpy as np
import tensorflow as tf
import networkx as nx
import torch
import torch.nn as nn

from random_walker.alias_sample import create_alias_table


class _LINE(object):

    def __init__(self, graph, dim=128, batch_size=1000, negative_ratio=5, order=3):
        self.cur_epoch = 0
        self.order = order
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.n_nodes = graph.number_of_nodes()
        self.n_edges = graph.number_of_edges()
        self.dim = dim
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio

        self._add_inverse_edges()
        self._build_lookup_table()
        self._create_sampling_table()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        cur_seed = random.getrandbits(32)
        initializer = tf.initializers.glorot_normal(seed=cur_seed)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_tensorflow_graph()
        self.sess.run(tf.global_variables_initializer())


    def _build_lookup_table(self):
        """
        node与index的映射表
        """
        self.lookup_table = {}
        for idx, node in enumerate(self.nodes):
            self.lookup_table[node] = idx


    def _add_inverse_edges(self):
        """
        有向图，给边加上对称边
        """
        edges = self.graph.edges()
        inv_edges = []
        for edge in edges:
            u, v = edge[0], edge[1]
            inv_edges.append((v, u, {'weight': self.graph[u][v]['weight']}))
        self.graph.add_edges_from(inv_edges)


    def build_torch_graph(self):
        self.embeddings = nn.Embedding.from_pretrained(torch.randn(self.n_nodes, self.dim))


    def build_tensorflow_graph(self):
        self.h = tf.placeholder(tf.int32, [None])
        self.t = tf.placeholder(tf.int32, [None])
        self.sign = tf.placeholder(tf.float32, [None])
        cur_seed = random.getrandbits(32)

        # 初始化各个节点的嵌入向量
        self.embeddings = tf.get_variable(name="embeddings"+str(self.order),
                                          shape=[self.n_nodes, self.dim],
                                          initializer=tf.initializers.glorot_normal(seed=cur_seed))
        # 初始化各个节点作为context的嵌入向量
        self.context_embeddings = tf.get_variable(name="context_embeddings"+str(self.order),
                                                  shape=[self.n_nodes, self.dim],
                                                  initializer=tf.initializers.glorot_normal(seed=cur_seed))

        # self.h_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.h), 1)
        # self.t_e = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.embeddings, self.t), 1)
        # self.t_e_context = tf.nn.l2_normalize(tf.nn.embedding_lookup(self.context_embeddings, self.t), 1)
        self.h_e = tf.nn.embedding_lookup(self.embeddings, self.h)
        self.t_e = tf.nn.embedding_lookup(self.embeddings, self.t)
        self.t_e_context = tf.nn.embedding_lookup(self.context_embeddings, self.t)
        # Loss_2 = - ∑W_ij * log(P_2(vj | vi)), P_2 is softmax
        self.second_loss = -tf.reduce_mean(tf.log_sigmoid(
            self.sign * tf.reduce_sum(tf.multiply(self.h_e, self.t_e_context), axis=1)))
        # Loss_1 = - ∑W_ij * log(P_1(vi, vj)), P_1 is log-sigmoid
        self.first_loss = -tf.reduce_mean(tf.log_sigmoid(
            self.sign * tf.reduce_sum(tf.multiply(self.h_e, self.t_e), axis=1)))

        if self.order == 1:
            self.loss = self.first_loss
        elif self.order == 2:
            self.loss = self.second_loss

        optimizer = tf.train.AdamOptimizer(0.001)
        self.train_op = optimizer.minimize(self.loss)


    def train_one_epoch(self):
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            h, t, sign = batch
            feed_dict = {
                self.h: h,
                self.t: t,
                self.sign: sign,
            }
            _, cur_loss = self.sess.run([self.train_op, self.loss], feed_dict)
            sum_loss += cur_loss
            batch_id += 1
        print('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss))
        self.cur_epoch += 1


    def batch_iter(self):
        table_size = 1e8
        edges = [(self.lookup_table[edge[0]], self.lookup_table[edge[1]]) for edge in self.graph.edges()]
        shuffle_indices = np.random.permutation(np.arange(self.n_edges))

        # positive or negative mode
        mode = 0
        mode_size = 1 + self.negative_ratio
        h = []
        t = []
        start_index = 0
        end_index = min(start_index + self.batch_size, self.n_edges)
        while start_index < self.n_edges:
            if mode == 0:
                # 正样本
                sign = 1.0
                h = []
                t = []
                for i in range(start_index, end_index):
                    if not random.random() < self.edge_prob[shuffle_indices[i]]:
                        shuffle_indices[i] = self.edge_alias[shuffle_indices[i]]
                    cur_h = edges[shuffle_indices[i]][0]
                    cur_t = edges[shuffle_indices[i]][1]
                    h.append(cur_h)
                    t.append(cur_t)
            else:
                # 负采样
                sign = -1.0
                t = []
                for i in range(len(h)):
                    t.append(self.sampling_table[random.randint(0, table_size-1)])

            yield h, t, [sign]
            mode += 1
            mode %= mode_size
            if mode == 0:
                start_index = end_index
                end_index = min(start_index + self.batch_size, self.n_edges)


    def _create_sampling_table(self):
        """
        建立负采样表和别名采样表
        :return:
        """
        table_size = 1e8
        power = 0.75 # 3/4

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = dict()
        for edge in self.graph.edges():
            u, v = edge[0], edge[1]
            weight = self.graph[u][v]['weight']
            node_degree[u] = node_degree.get(u, 0.0) + weight # 只记录出度，无向图中有对称边，所以只考虑u

        norm = sum([math.pow(node_degree[node], power) for node in self.nodes])
        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        # 节点采样表，每个节点被抽中的概率和节点的出度有关。
        p, i = 0, 0
        for idx, node in enumerate(self.nodes):
            p += float(math.pow(node_degree[node], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = idx
                i += 1

        # 边的别名采样
        self.edge_alias = np.zeros(self.n_edges, dtype=np.int32)
        self.edge_prob = np.zeros(self.n_edges, dtype=np.float32)
        total_sum = sum([self.graph[edge[0]][edge[1]]["weight"] for edge in self.graph.edges()])
        norm_prob = [self.graph[edge[0]][edge[1]]["weight"] * self.n_edges / total_sum for edge in self.graph.edges()]
        self.edge_prob, self.edge_alias = create_alias_table(norm_prob)


    def get_embeddings(self):
        vectors = {}
        embeddings = self.embeddings.eval(session=self.sess)
        print("number of nodes:{}".format(self.n_nodes))
        print("number of edges:{}".format(self.n_edges))
        print("shape of embeddings:{}".format(embeddings.shape))
        for idx, embedding in enumerate(embeddings):
            vectors[self.nodes[idx]] = embedding
        return vectors



class LINE(object):
    def __init__(self, graph, d=128, batch_size=1000, epoch=10, negative_ratio=5, order=3):
        """

        :param graph:
        :param d: 目标嵌入维度
        :param batch_size:
        :param epoch:
        :param negative_ratio: 负采样，一个正样本对五个负样本
        :param order: 采用几阶相似性，order==3时，同时考虑一阶和二阶
        :param clf_ratio:
        :param auto_save:
        """
        self.rep_size = d
        self.order = order
        self.best_result = 0
        self.vectors = {}
        if order == 3:
            self.model1 = _LINE(graph, d // 2, batch_size, negative_ratio, order=1)
            self.model2 = _LINE(graph, d // 2, batch_size, negative_ratio, order=2)
            for i in range(epoch):
                self.model1.train_one_epoch()
                self.model2.train_one_epoch()
        else:
            self.model = _LINE(graph, d, batch_size, negative_ratio, order=self.order)
            for i in range(epoch):
                self.model.train_one_epoch()

        self.get_embeddings()


    def get_embeddings(self):
        self.last_vectors = self.vectors
        self.vectors = {}
        if self.order == 3:
            vectors1 = self.model1.get_embeddings()
            vectors2 = self.model2.get_embeddings()
            for node in vectors1.keys():
                self.vectors[node] = np.append(vectors1[node], vectors2[node])
        else:
            self.vectors = self.model.get_embeddings()

        return self.vectors


    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()


if __name__ == '__main__':
    from utils import util
    from utils.visualize import plot_embeddings

    graph = nx.read_edgelist(path="../../similarity/mkarate_10_L1.csv", create_using=nx.DiGraph, nodetype=str, data=[('weight', float)])
    model = LINE(graph, d=64, epoch=50, order=3)
    embeddings_dict = model.get_embeddings()

    labels = util.read_label("../../data/subway.label")
    nodes = []
    embeddings = []
    for node, embedding in embeddings_dict.items():
        nodes.append(node)
        embeddings.append(embedding)

    plot_embeddings(nodes, np.array(embeddings), labels, method='tsne', perplexity=10)
