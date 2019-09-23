# -*- coding: utf-8 -*-


class EmbeddingMixin(object):
    """
    Graph Embedding Method mixin.
    """

    def __init__(self):
        self.embeddings = None


    def train(self):
        raise NotImplementedError("111")


    def getEmbedding(self):
        return self.embeddings


    def getParams(self):
        return None

