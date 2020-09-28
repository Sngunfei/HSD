# -*- encoding: utf-8 -*-

# Multi-scales HSD implementataion

import numpy as np
import networkx as nx
import pygsp
import multiprocessing
from collections import defaultdict
from tqdm import tqdm
from .HSD import HSD
from tools import hierarchy


class MultiHSD(HSD):

    def __init__(self, graph: nx.Graph, graphName: str, hop: int, n_scales:int, metric="euclidean"):
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
                embeddings[node].extend(self.get_layer_sum(wavelets, node))

        self.embeddings = embeddings
        return embeddings


# plot wavelet changes in multi scales
def multiscale_plot_wavelets():
    pass

