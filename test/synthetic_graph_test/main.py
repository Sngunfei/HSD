# -*- encoding: utf-8 -*-

# 分形图，本质和层级二叉树类似，每一层的节点都是结构对称的

import networkx as nx

class TreeNode(object):

    def __init__(self, node_id, node_label):
        self.id = node_id
        self.label = node_label

        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"id:{self.id}, label:{self.label}, num of children:{len(self.children)}, children: {self.children}"

    def __str__(self):
        return self.__repr__()

# 创建新图
def construct_graph():
    n_branch = 2
    root = TreeNode(0, 0)
    root.add_child(TreeNode(1, 1))
    root.add_child(TreeNode(2, 1))
    root.add_child(TreeNode(3, 1))

    def id_incrementor():
        data = {"id": 3}

        def number():
            data["id"] += 1
            return data["id"]

        return number

    next_id = id_incrementor()

    node_label = 2
    queue = [node for node in root.children]
    for iteration in range(5):
        cap = len(queue)
        for _ in range(cap):
            cur_node = queue.pop(0)
            for _ in range(n_branch):
                child = TreeNode(next_id(), node_label)
                cur_node.add_child(child)
                queue.append(child)
        node_label += 1

    graph = nx.Graph()
    labels = dict()
    queue = [root]
    while queue:
        cap = len(queue)
        for _ in range(cap):
            cur_node = queue.pop()
            for child in cur_node.children:
                graph.add_edge(cur_node.id, child.id)
                queue.append(child)
                labels[child.id] = child.label
    return graph, labels

def write2file(graph: nx.Graph, labels: dict):
    edges = list(graph.edges())
    edges.sort(key=lambda x: x[0])
    with open("tree.edgelist", mode="w+", encoding="utf-8") as fout:
        for edge in edges:
            fout.write(f"{edge[0]} {edge[1]}\n")

    with open("tree.label", mode="w+", encoding="utf-8") as fout:
        for node, label in labels.items():
            fout.write(f"{node} {label}\n")

if __name__ == '__main__':
    graph, labels = construct_graph()
    write2file(graph, labels)
