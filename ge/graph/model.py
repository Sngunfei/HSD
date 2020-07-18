# -*- encoding: utf-8 -*-

"""
对nx.graph进行一层封装
"""

class Graph:

    def __init__(self, G, graph_name: str):
        self.G = G
        self.name = graph_name
