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


class Node2Vec:
    """
        实现Node2vec嵌入算法，DeepWalk的进阶版，当p=q的时候，退化为deepwalk。
    """
    def __init__(self, graph, walk_length, num_walks, p=1.0, q=1.0, workers=1):
        """
        :param graph:
        :param walk_length: 随机游走的路径长度
        :param num_walks: 采集的路径条数
        :param p:
        :param q:
        :param workers:
        """
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
        :param graph: 图
        :param p: 返回参数
        :param q: 离开参数
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
            node2vec是在deepwalk的基础上发展的，在游走过程中设定了两种策略，广搜和深搜，分别用两个参数p，q控制
        :param walk_length:
        :param start_node:
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
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0], alias_edges[edge][1])]
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
            partition_num(num_walks, workers))

        walks = list(itertools.chain(*results))

        return walks


    def _simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                if self.p == self.q:
                    walks.append(self.deepwalk_walk(walk_length=walk_length, start_node=v))
                else:
                    walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks


    def get_alias_edge(self, t, v):
        """
        compute unnormalized transition probability between nodes v and its neighbors give the previous visited node t.
        :param t:
        :param v:
        :return:
        """
        G = self.graph
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)


    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self.graph

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr].get('weight', 1.0)
                                  for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}

        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []

    for i, prob in enumerate(area_ratio):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio[small_idx]
        alias[small_idx] = large_idx
        area_ratio[large_idx] = area_ratio[large_idx] - \
            (1 - area_ratio[small_idx])
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


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]


