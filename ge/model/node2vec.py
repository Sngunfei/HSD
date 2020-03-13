# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import Word2Vec
import logging
from ge.random_walker.walker import RandomWalker


class Node2Vec:

    def __init__(self, graph, graph_name, dim=64, walk_length=15,
                 walk_num=10, window_size=10, p=1.0, q=1.0, workers=1,
                 iter=5):
        """
        :param graph: 有向图
        :param walk_length: 随机游走的路径长度
        :param num_walks: 采集的路径条数
        :param p: 返回概率，越小越容易返回，广搜策略
        :param q: 离开概率，越小越容易向外，深搜策略
        :param workers:
        """
        self.graph = graph
        self.graph_name = graph_name
        self.p = p
        self.q = q
        self.window_size = window_size
        self.workers = workers
        self.dim = dim
        self.iter = iter
        self.walk_length = walk_length
        self.walk_num = walk_num

        # 因为在计算转移概率时，需要考虑边的方向，但研究的是无向图，所以就用两条方向相反的边来模拟无向图。
        self.embeddings = None
        self.walker = RandomWalker(graph, p=p, q=q)
        print("Preprocess transition probs...")
        self.walker.preprocess_transition_probs()
        self.sentences = self.walker.simulate_walks(num_walks=walk_num, walk_length=walk_length, workers=workers, verbose=1)


    def train(self, **kwargs):
        kwargs["sentences"] = self.sentences
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["size"] = self.dim
        kwargs["sg"] = 1
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = self.workers
        kwargs["window"] = self.window_size
        kwargs["iter"] = self.iter

        logging.info("Learning embedding vectors...")
        model = Word2Vec(**kwargs)
        logging.info("Learning embedding vectors done!")
        self.w2v_model = model

        return model


    def get_embeddings(self,):
        if self.w2v_model is None:
            print("word2vec model have not been trained.")
            return {}

        self.embeddings = {}
        for word in self.graph.nodes():
            self.embeddings[word] = self.w2v_model.wv[word]

        return self.embeddings





