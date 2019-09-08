# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import Word2Vec
import pandas as pd
import itertools
import math
import random
import numpy as np
from joblib import Parallel, delayed
import logging
import networkx as nx


class Node2Vec:

    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):
        """
        因为在计算转移概率时，需要考虑边的方向，但研究的是无向图，所以就用两条方向相反的边来模拟无向图。
        :param graph: 有向图
        :param walk_length: 随机游走的路径长度
        :param num_walks: 采集的路径条数
        :param p: 返回概率，越小越容易返回，广搜策略
        :param q: 离开概率，越小越容易向外，深搜策略
        :param workers:
        """
        edges = graph.edges()
        inv_edges = []
        for edge in edges:
            u, v = edge[0], edge[1]
            inv_edges.append((v, u, {'weight': graph[u][v]['weight']}))
        graph.add_edges_from(inv_edges)
        self.graph = graph
        self.embeddings = {}
        self.walker = RandomWalker(graph, p=p, q=q)

        logging.info("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)


    def train(self, embed_size=128, window_size=5, workers=3, iter=5, **kwargs):
        """
        :param embed_size: 目标嵌入维度
        :param window_size: word2vec模型中采用的窗口大小
        :param workers:
        :param iter:
        :param kwargs:
        :return:
        """

        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = embed_size
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = workers
        kwargs["window"] = window_size
        kwargs["iter"] = iter

        logging.info("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        logging.info("Learning embedding vectors done!")
        self.w2v_model = model

        return model


    def get_embeddings(self,):
        if self.w2v_model is None:
            print("word2vec model have not been trained.")
            return {}

        self._embeddings = {}
        for word in self.graph.nodes():
            self._embeddings[word] = self.w2v_model.wv[word]

        return self._embeddings


class RandomWalker:
    """
        随机游走
    """
    def __init__(self, graph, p=1.0, q=1.0):
        """
        p和q都无法影响同一层之间的游走，对于无权图，返回的概率为1/p，离开的概率为1/q，而走到同一层的概率为1,
        然后归一化即可得到下一步游走的概率分布，当且仅当p=q=1.0时，node2vec才退化为deepwalk，此时各方向游走
        的概率相等。
        :param graph: 图
        :param p: 返回参数, p越小，越倾向于返回上一个节点
        :param q: 离开参数，q越小，越倾向于向外走
        """
        self.graph = graph
        self.p = p
        self.q = q


    def deepwalk_walk(self, walk_length, start_node):
        """
        完全随机的游走，不考虑广搜和深搜策略，这种游走无法控制。
        :param walk_length:
        :param start_node:
        :return: 游走路径，list
        """
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(self.graph.neighbors(cur))
            if len(cur_nbrs) > 0:
                walk.append(random.choice(cur_nbrs))
            else:
                break
        return walk


    def node2vec_walk(self, walk_length, start_node):
        """
        Node2vec是在deepwalk的基础上发展的，在游走过程中设定了两种策略，广搜和深搜，分别用两个参数p，q控制
        :param walk_length: 游走的路径长度
        :param start_node: 起始节点
        :return: 游走路径，list
        """
        G = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[self._alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[self._alias_sample(alias_edges[edge][0], alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk


    def simulate_walks(self, num_walks, walk_length, workers=1, verbose=0):
        """

        :param num_walks:
        :param walk_length:
        :param workers:
        :param verbose:
        :return:
        """
        G = self.graph
        nodes = list(G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose, )(
            delayed(self._simulate_walks)(nodes, num, walk_length) for num in
            self._partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks


    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == 1.0 and self.q == 1.0: # 1/p = 1/q = 1.0
                    walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks


    def get_alias_edge(self, t, v):
        """
        从节点t走到v，此时状态在v节点，其有一些邻居，计算下一步走到各个邻居的转移概率。
        :param t: 上一个走过的节点
        :param v: 当前所在的节点
        :return: v邻居的转移概率分布
        """
        G = self.graph
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # dist(t, x) = 0
                unnormalized_probs.append(weight / p)
            elif G.has_edge(x, t):  # dist(t, x) = 1
                unnormalized_probs.append(weight)
            else:  # dist(t, x) > 1， or dist(t, x) = 2
                unnormalized_probs.append(weight / q)

        unnormalized_probs = np.array(unnormalized_probs, dtype=np.float)
        normalized_probs = unnormalized_probs / np.sum(unnormalized_probs) * len(unnormalized_probs)

        return self._create_alias_table(normalized_probs)


    def preprocess_transition_probs(self):
        """
        预处理转移概率，计算下一步走哪个节点的概率，有可能往外走，也有可能往回走。
        """

        G = self.graph
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = np.array([G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)])
            normalized_probs = unnormalized_probs / np.sum(unnormalized_probs) * len(unnormalized_probs)
            alias_nodes[node] = self._create_alias_table(normalized_probs)

        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges


    def _create_alias_table(self, area_ratio):
        """
        :param area_ratio: sum(area_ratio)= N
        :return: accept,alias
        """
        N = len(area_ratio)
        accept, alias = [0] * N, [0] * N
        small, large = [], []

        for i, prob in enumerate(area_ratio):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)
        #print("probs: {}".format(str(area_ratio)))
        #print("all:{}, small:{}, large:{}".format(N, str(small), str(large)))
        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = area_ratio[small_idx]
            alias[small_idx] = large_idx
            area_ratio[large_idx] -= (1 - area_ratio[small_idx])
            if area_ratio[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1

        return accept, alias


    def _alias_sample(self, accept, alias):
        """
        采样过程
        :param accept:
        :param alias:
        :return: sample index
        """
        N = len(accept)
        i = int(np.random.random() * N)
        r = np.random.random()
        if r < accept[i]:
            return i
        else:
            return alias[i]


    def _partition_num(self, num, workers):
        if num % workers == 0:
            return [num // workers] * workers
        else:
            return [num // workers] * workers + [num % workers]


if __name__ == '__main__':
    from utils import util
    from utils.visualize import plot_embeddings
    G = nx.read_edgelist(path="../../output/test.csv", create_using=nx.DiGraph, nodetype=str, data=[('weight', float)])
    model = Node2Vec(G, walk_length=15, num_walks=80, p=0.5, q=2.0, workers=1)
    model.train(window_size=7, iter=500)

    embeddings_dict = model.get_embeddings()
    labels = util.read_label("../../data/bell.label")
    nodes = []
    embeddings = []
    for node, embedding in embeddings_dict.items():
        nodes.append(node)
        embeddings.append(embedding)

    plot_embeddings(nodes, np.array(embeddings), labels, method='tsne', perplexity=3)




