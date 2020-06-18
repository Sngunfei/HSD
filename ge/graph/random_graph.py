
"""
自动生成小规模图，GraphWave
"""

import networkx as nx
from utils.rw import save_edgelist
import random

class VariedGraph:

    def __init__(self):
        self.graph = None
        self.labels = None
        self.n_label = 0
        self.nodes = None
        pass


    def basic_graph(self, type: str):
        """
        The graphs are given by basic shapes of one of diﬀerent types
        (“house”, “fan”, “star”) that are regularly placed along a cycle of length 30.
        :param type:
        :return:
        """
        self.type = type

        n_cycle_nodes = 30
        n_plant = 5

        self.nodes = []
        self.labels = {}
        self.graph = nx.Graph()
        for i in range(n_cycle_nodes):
            self.graph.add_edge(i, (i+1) % n_cycle_nodes)
            self.nodes.append(i)
            self.labels[i] = 0

        for i in range(n_plant):
            node = (n_cycle_nodes // n_plant) * i
            self.labels[node] = 1
            base = len(self.nodes)
            if type == "house":
                self.add_house(node, base, 2)
            elif type == "fan":
                self.add_fan(node, base, 2)
            elif type == "star":
                self.add_star(node, base, 2)
            else:
                raise ValueError("Auto-generate graph, type {} is unsupported.".format(type))


    def add_house(self, junction: int, base: int, label_base: int):
        """
        生成house，一共5个点，按顺序赋label
        :param junction: 圆环上的连接点
        :param base:
        :param label_base:
        :return:
        """
        cnt = 5
        # 贴标签
        nodes = [(base+i) for i in range(cnt)]
        labels = {nodes[0]:label_base,
                  nodes[1]:label_base+1,
                  nodes[2]:label_base+1,
                  nodes[3]:label_base+2,
                  nodes[4]:label_base+2}

        # 边
        edges = [(junction, nodes[0])]
        edges.extend([(nodes[0], nodes[1]), (nodes[0], nodes[2])])
        for i in range(1, cnt):
            for j in range(i+1, cnt):
                edges.append((nodes[i], nodes[j]))

        # 添加到圆环上
        self.graph.add_edges_from(edges)
        self.nodes.extend(nodes)
        self.labels.update(labels)


    def add_fan(self, junction, base, label_base):
        """

        :param junction:
        :param base:
        :return:
        """
        cnt = 7
        nodes = [(base + i) for i in range(cnt)]
        labels = {}
        edges = []

        # 贴标签
        labels[nodes[0]] = label_base
        labels[nodes[1]] = label_base + 1
        for i in range(2, cnt):
            labels[nodes[i]] = label_base + 2

        # 边
        edges.append((junction, nodes[0]))
        edges.append((nodes[0], nodes[1]))
        for i in range(2, cnt):
            edges.append((nodes[1], nodes[i]))
        for i in range(2, cnt - 1):
            edges.append((nodes[i], nodes[i+1]))

        # 添加到圆环上
        self.graph.add_edges_from(edges)
        self.nodes.extend(nodes)
        self.labels.update(labels)


    def add_star(self, junction, base, label_base):
        """

        :param junction:
        :param base:
        :return:
        """
        cnt = 7
        nodes = [(base + i) for i in range(cnt)]
        labels = {base: label_base}
        for i in range(1, cnt):
            labels[nodes[i]] = label_base + 1
        edges = [(nodes[0], nodes[i]) for i in range(1, cnt)]
        edges.append((junction, nodes[0]))
        self.graph.add_edges_from(edges)
        self.nodes.extend(nodes)
        self.labels.update(labels)


    def save(self):
        edge_path = f"../../data/graph/basic_{self.type}.edgelist"
        label_path = f"../../data/label/basic_{self.type}.label"
        save_edgelist(edge_path, nx.edges(self.graph))
        save_edgelist(label_path, self.labels.items())


    def varied_graph(self, save=True, random_seed=42, simplify_label=False):
        """
        自动生成一个随机图
        :return:
        """

        random.seed(random_seed)
        n_cycle_nodes = 40
        n_plant = 8

        self.nodes = []
        self.labels = {}
        self.graph = nx.Graph()
        for i in range(n_cycle_nodes):
            self.graph.add_edge(i, (i + 1) % n_cycle_nodes)
            self.nodes.append(i)
            self.labels[i] = 0

        plant_places = []
        for i in range(n_plant * 3):
            pos = random.randint(0, n_cycle_nodes)
            while pos in plant_places:
                pos = random.randint(0, n_cycle_nodes)
            plant_places.append(pos)
            if i < n_plant:
                if simplify_label:
                    self.labels[pos] = 1
                    self.add_house(pos, len(self.nodes), 2)
                else:
                    self.labels[pos] = 1
                    self.add_house(pos, len(self.nodes), 4)
                #print(f"add house: pos={pos}, number of nodes={len(self.nodes)}")
            elif i < n_plant * 2:
                if simplify_label:
                    self.labels[pos] = 1
                    self.add_star(pos, len(self.nodes), 5)
                else:
                    self.labels[pos] = 2
                    self.add_star(pos, len(self.nodes), 7)
                    #print(f"add star: pos={pos}, number of nodes={len(self.nodes)}")
            elif i < n_plant * 3:
                if simplify_label:
                    self.labels[pos] = 1
                    self.add_fan(pos, len(self.nodes), 7)
                else:
                    self.labels[pos] = 3
                    self.add_fan(pos, len(self.nodes), 9)
                #print(f"add fan: pos={pos}, number of nodes={len(self.nodes)}")

        if save:
            save_edgelist(f"../../data/graph/varied_graph.edgelist", nx.edges(self.graph))
            save_edgelist(f"../../data/label/varied_graph.label", self.labels.items())

        return self.graph

    def get_graph(self, random_seed):
        g = self.varied_graph(save=False, random_seed=random_seed, simplify_label=True)
        return nx.Graph(g), self.labels


def _test():
    from collections import defaultdict
    dic = defaultdict(dict)
    basic = open("E:\workspace\py\graph-embedding\data\graph\\basic_house.edgelist", mode="r", encoding="utf-8")
    house = open("E:\workspace\py\graph-embedding\data\graph\house.edgelist", mode="r", encoding="utf-8")

    while True:
        line = house.readline()
        if not line:
            break
        u, v = line.strip().split(" ")
        u, v = int(u) - 1, int(v) - 1
        dic[u][v] = dic[v][u] = 1

    while True:
        line = basic.readline()
        if not line:
            break
        u, v = line.strip().split(" ")
        u, v = int(u), int(v)
        del dic[u][v]
        del dic[v][u]

    print(dic)

    for k, v in dic.items():
        if len(v) != 0:
            print(k, v)


if __name__ == '__main__':
    m = VariedGraph()
    m.varied_graph()
    #m.basic_graph("house")
    #m.save()

