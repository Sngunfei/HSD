# -*- encoding: utf-8 -*-

# Dynamic Graph Embedding based on HSD

import networkx as nx
import copy
from model import MultiHSD


class DynamicHSD(MultiHSD):

    def __init__(self, graph: nx.Graph, graphName: str, hop: int, n_scales: int, metric="euclidean"):
        super(DynamicHSD, self).__init__(graph, graphName, hop, n_scales, metric)
        self.embeddings = {}
    
    
    def init(self):
        super(DynamicHSD, self).init()


    # dynamic node
    # newNode
    def dynamic_add_node(self, newNode: str, edges: list):
        pass


    # explore the local neighborhoods of node
    def explore_neighborhoods(self, node, maxHop=5) -> set:
        neighborhoods = {node}
        curLayer = {node}
        layers = [curLayer]
        for hop in range(self.hop):
            nextLayer = set()
            for curNode in curLayer:
                nextHopNeighbors = set(nx.neighbors(self.graph, curNode))
                nextLayer = nextLayer.union(nextHopNeighbors - neighborhoods)
            curLayer = nextLayer

            layers.append(copy.deepcopy(curLayer))
            neighborhoods.union(curLayer)

        self.hierarchy[node] = layers
        return neighborhoods


    def convert_neighborhoods_to_subgraph(self, neighbors) -> nx.Graph:
        candidate_edges = nx.edges(self.graph, neighbors)
        edges = set()
        for edge in candidate_edges:
            if edge[0] in neighbors and edge[1] in neighbors:
                edges.add(edge)

        subGraph = nx.Graph()
        subGraph.add_edges_from(edges)
        return subGraph
