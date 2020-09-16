# -*- encoding: utf-8 -*-

from functools import cmp_to_key

def reindex(filePath):
    edges = []
    sets = set()
    with open(filePath, mode="r", encoding="utf-8") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = line.strip().split(" ")
            node1, node2 = int(line[0]), int(line[1])
            node1, node2 = min(node1, node2), max(node1, node2)
            if f"{node1}#{node2}" in sets:
                continue
            edges.append([node1, node2])
            sets.add(f"{node1}#{node2}")
    print(f"number of edges:{len(edges)}")

    def cmp(edge1, edge2):
        if edge1[0] == edge2[0]:
            return edge1[1] - edge2[1]
        return edge1[0] - edge2[0]

    edges = sorted(edges, key=cmp_to_key(cmp))
    if edges[0][0] == 0:
        for idx, edge in enumerate(edges):
            edges[idx] = "{},{}\n".format(edge[0], edge[1])
    elif edges[0][0] == 1: # 以1为起点，统一归成0
        for idx, edge in enumerate(edges):
            edges[idx] = "{},{}\n".format(edge[0]-1, edge[1]-1)
    else:
        print(edges[0])
        raise ValueError

    with open(filePath.replace("edgelist", "csv"), mode="w+", encoding="utf-8") as fout:
        fout.write("node1,node2\n")
        fout.writelines(edges)

    print("done.")

if __name__ == '__main__':
    files = ["../data/graph/bio_dmela.edgelist"]
    for filePath in files:
        reindex(filePath)
