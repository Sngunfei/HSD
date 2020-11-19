# -*- encoding: utf-8 -*-

from functools import cmp_to_key

def reindex(filePath):
    edges = set()
    with open(filePath, mode="r", encoding="utf-8") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            node1, node2 = map(int, line.strip().split(" "))
            edge = (min(node1, node2), max(node1, node2))
            if edge[0] == edge[1]:
                continue
            if edge in edges:
                print(edge)
            edges.add(edge)

    edges = list(edges)
    print(f"number of edges:{len(edges)}")

    def cmp(edge1, edge2):
        if edge1[0] == edge2[0]:
            return edge1[1] - edge2[1]
        return edge1[0] - edge2[0]
    edges = sorted(edges, key=cmp_to_key(cmp))

    node_idx = 0
    node2idx = {}
    for edge in edges:
        node1, node2 = edge[0], edge[1]
        if node1 not in node2idx:
            node2idx[node1] = node_idx
            node_idx += 1
        if node2 not in node2idx:
            node2idx[node2] = node_idx
            node_idx += 1

    for idx, edge in enumerate(edges):
        edges[idx] = (node2idx[edge[0]], node2idx[edge[1]])
    edges = sorted(edges, key=cmp_to_key(cmp))

    with open(filePath.replace(".edgelist", "_reindex.edgelist"), mode="w+", encoding="utf-8") as fout:
        for edge in edges:
            fout.write("{} {}\n".format(edge[0], edge[1]))

    print("done.")

if __name__ == '__main__':
    files = ["../data/graph/cora.edgelist"]
    for filePath in files:
        reindex(filePath)
    # labels = {}
    # with open("../data/label/mkarate.label", mode="r", encoding="utf-8") as fin:
    #     while True:
    #         line = fin.readline().strip()
    #         if not line:
    #             break
    #         node, label = line.split(" ")
    #         labels[int(node) - 1] = label
    #
    # with open("../data/label/mkarate_new.label", mode="w+", encoding="utf-8") as fout:
    #     for node, label in labels.items():
    #         fout.write("{} {}\n".format(node, label))
