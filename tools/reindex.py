# -*- encoding: utf-8 -*-

from functools import cmp_to_key

def reindex(edgelist_path):
    edges = set()
    line_count = 0
    with open(edgelist_path, mode="r", encoding="utf-8") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line_count += 1
            node1, node2 = map(int, line.strip().split(" "))
            edge = (min(node1, node2), max(node1, node2))
            if edge[0] == edge[1]:
                continue
            edges.add(edge)

    edges = list(edges)
    print(f"line: {line_count}, number of edges:{len(edges)}")

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

    with open( edgelist_path.replace(".edgelist", "_reindex.edgelist"), mode="w+", encoding="utf-8") as fout:
        for edge in edges:
            fout.write("{} {}\n".format(edge[0], edge[1]))

    print("done.")

if __name__ == '__main__':
    files = ["../data/graph/facebook.edgelist"]
    for edgelist_path in files:
        reindex(edgelist_path)
