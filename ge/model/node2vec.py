# -*- coding:utf-8 -*-
import itertools
import math
import random
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
import numpy as np
from joblib import Parallel, delayed
import logging
import networkx as nx
from ge.utils.walker import RandomWalker


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

        self.graph = graph
        self._add_inverse_edges()
        self.embeddings = {}
        self.walker = RandomWalker(graph, p=p, q=q)

        logging.info("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()
        self.sentences = self.walker.simulate_walks(num_walks=num_walks, walk_length=walk_length, workers=workers, verbose=1)


    def _add_inverse_edges(self):
        edges = self.graph.edges()
        inv_edges = []
        for edge in edges:
            u, v = edge[0], edge[1]
            inv_edges.append((v, u))
        self.graph.add_edges_from(inv_edges)


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





