# -*- encoding: utf-8 -*-

# Multi-scales HSD implementataion

import multiprocessing
from collections import defaultdict

import networkx as nx
import numpy as np
import pygsp
from tqdm import tqdm

from model import HSD
from tools import hierarchy

class MultiHSD(HSD):

    def __init__(self, graph: nx.Graph, graphName: str, hop: int, n_scales: int, metric="euclidean"):
        super(MultiHSD, self).__init__(graph, graphName, 0, hop, metric)

        self.n_scales = n_scales
        self.scales = None
        self.embeddings = {}

    def init(self):
        G = pygsp.graphs.Graph(self.adjacent)
        G.estimate_lmax()
        # 如何取scales?
        self.scales = np.exp(np.linspace(np.log(0.01), np.log(G._lmax), self.n_scales))
        self.hierarchy = hierarchy.read_hierarchical_representation(self.graphName, self.hop)

    def init2(self):
        G = pygsp.graphs.Graph(self.adjacent)
        G.estimate_lmax()
        # 如何取scales?
        self.scales = np.exp(np.linspace(np.log(0.01), np.log(G._lmax), self.n_scales))

    # embed nodes into vectors using multi-scale wavelets
    def embed(self) -> dict:
        embeddings = defaultdict(list)
        for scale in tqdm(self.scales):
            wavelets = self.calculate_wavelets(scale, approx=True)
            for node in self.nodes:
                embeddings[node].extend(self.get_layer_sum(wavelets, node))
        return embeddings


    def get_layer_sum(self, wavelets: np.ndarray, node:str) -> list:
        layers_sum = [0] * (self.hop + 1)
        neighborhoods = self.hierarchy[node]
        node_idx = self.node2idx[node]
        for hop, level in enumerate(neighborhoods):
            for neighbor in level:
                if neighbor == '':
                    continue
                layers_sum[hop] += wavelets[node_idx, self.node2idx[neighbor]]
        return layers_sum


    def parallel_embed(self, n_workers) -> dict:
        pool = multiprocessing.Pool(n_workers)
        states = {}
        for idx, scale in enumerate(self.scales):
            res = pool.apply_async(self.calculate_wavelets, args=(scale, True))
            states[idx] = res
        pool.close()
        pool.join()

        results = []
        for idx in range(self.n_scales):
            results.append(states[idx].get())

        embeddings = defaultdict(list)
        for idx, _ in enumerate(self.scales):
            wavelets = results[idx]
            for node in self.nodes:
                # 每一层简单求和
                #embeddings[node].extend(self.get_layer_sum(wavelets, node))
                # 每一层用三元组作为描述符
                embeddings[node].extend(self.get_triple(wavelets, node))
        self.embeddings = embeddings
        return embeddings

    def get_triple(self, wavelets: np.ndarray, node: str) -> list:
        descriptor = []
        neighborhoods = self.hierarchy[node]
        node_idx = self.node2idx[node]
        for hop, level in enumerate(neighborhoods):
            coeffs = []
            for neighbor in level:
                if neighbor == '':
                    continue
                coeffs.append(wavelets[node_idx, self.node2idx[neighbor]])

            if len(coeffs) > 0:
                triple = [np.sum(coeffs), np.mean(coeffs), np.var(coeffs)]
            else:
                triple = [0.0, 0.0, 0.0]

            descriptor.extend(triple)

        return descriptor

# plot wavelet changes in multi scales
def multiscale_plot_wavelets():
    pass
