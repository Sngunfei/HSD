# -*- encoding: utf-8 -*-

"""
    Susceptible - Infected - Recover Model

    随机选取某个节点为origin，向外传播疾病，感染概率为p，恢复概率为r，每次恢复后就免疫后续
感染。在给定的时间t内，其能传染的节点数量（感染 + 恢复），可以代表节点的influence，不考虑
节点属性和边属性，只考虑连接情况，能反映出节点的结构信息。
"""

import networkx as nx
from utils.util import build_node_idx_map, connect_graph
import copy, random
from collections import defaultdict

class SIR():

    def __init__(self, graph, p=1.0, r=None, t=25, random_state=42):
        """
        模型初始化
        :param graph: 网络图
        :param p: 传染概率
        :param r: 恢复概率， default：1 / (average degree)
        :param t: 传播时间
        """

        self.G = graph
        self.p = p
        if not r:
            self.r = 1.0 / self._get_average_degree()
        else:
            self.r = r
        self.t = t
        self.random_state = random_state
        self.idx2node, self.node2idx = build_node_idx_map(graph)
        self.influence = defaultdict(int)


    def _get_average_degree(self):
        total = 0
        for _, degree in nx.degree(self.G):
            total += degree
        return total / nx.number_of_nodes(self.G)


    def start(self):
        """
        Susceptible - Infected - Recover Model
        :return:
        """
        for idx, node in self.idx2node.items():
            for _ in range(5): # 重复5次消除随机因素
                self.influence[node] += self._diffuse_from_node(node)
            #print(node, self.influence[node])


    def _diffuse_from_node(self, node):
        """
        以node为起始点，向外传播
        :param node:
        :return:
        """
        infected = set() # 传染集合
        infected.add((node, 1))
        recoverd = set() # 恢复集合

        t = self.t
        while t > 0:
            #print(node, t, infected)
            current_infectd = copy.deepcopy(infected)
            for _node, _weight in current_infectd:
                for neighbor in nx.neighbors(self.G, _node):
                    # 传染
                    if neighbor not in recoverd and neighbor not in infected and random.random() <= self.p:
                        infected.add((neighbor, _weight-0.1))
                # 恢复
                if random.random() <= self.r:
                    recoverd.add((_node, _weight))
                    infected.remove((_node, _weight))
            t -= 1

        # 时间t内传染到的节点数量，即节点的影响力
        influence = sum([_weight for _, _weight in infected]) + sum([_weight for _, _weight in recoverd])
        return influence


    def label_nodes(self, n_class, mode=0):
        """
        根据node的influence来贴标签，n_class表示分为多少个类别

        按位置划分还是按score的区间划分？ 怎么分都不均匀= =
        :param n_class:
        :return:
        """
        scores = sorted(self.influence.items(), key=lambda x: x[1])
        min_score, max_score = scores[0][1], scores[-1][1]
        interval = (max_score - min_score) / n_class
        labels = {}

        cnt = defaultdict(int)
        i = 1

        if mode == 0:
            for node, score in scores:
                if score <= min_score + interval * i:
                    labels[node] = i - 1
                    cnt[i-1] += 1
                else:
                    labels[node] = i
                    cnt[i] += 1
                    i += 1
            #print(cnt)
            return labels
        elif mode == 1:
            interval = len(scores) / n_class
            for idx, info in enumerate(scores):
                node, score = info[0], info[1]
                if idx <= interval * i:
                    labels[node] = i - 1
                    cnt[i - 1] += 1
                else:
                    labels[node] = i
                    cnt[i] += 1
                    i += 1
            return labels


if __name__ == '__main__':
    from utils.util import dataloader

    name = "usa"
    data, _, _ = dataloader(name, directed=False)
    #print("radius", nx.radius(data))
    #print("diameter", nx.diameter(data))
    model = SIR(data, p=0.95, r=None, t=4, random_state=42)
    #print(model._get_average_degree())
    model.start()
    labels = model.label_nodes(5, mode=1)
    fout = open("../../data/{}_SIR_3.label".format(name), mode="w+", encoding="utf8")
    for node, label in labels.items():
        fout.write("{} {} \n".format(node, label))
    fout.close()




















