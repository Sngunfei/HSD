

def read_label(path):
    """
    读取节点的标签信息, 返回字典。
    """
    with open(path, mode="r", encoding="utf-8") as fin:
        labels = dict()
        while True:
            line = fin.readline()
            if not line:
                break
            node, label = line.strip().split(" ")
            labels[node] = label

    return labels


def write_label(data_path):
    import networkx as nx
    graph = nx.read_edgelist(path=data_path, create_using=nx.Graph, nodetype=str, edgetype=float, data=[('weight', float)])

    fout = open("G:\pyworkspace\graph-embedding\out\subway_label_2.txt", mode="w+", encoding="utf-8")

    nodes = list(nx.nodes(graph))
    rings = dict()
    for node1 in nodes:
        hop1, hop2 = 0, 0
        for node2 in nodes:
            length = nx.dijkstra_path_length(graph, node1, node2)
            if length > 2:
                continue
            elif length == 1:
                hop1 += 1
            elif length == 2:
                hop2 += 1
        rings[node1] = [hop1, hop2]

    for node, hop in rings.items():
        hop1 = min(hop[0], 4)
        hop2 = min(hop[1], 6) // 3 + 1
        label = (hop1 - 1) * 3 + hop2
        fout.write("{} {}\n".format(node, label))

    fout.close()


if __name__ == '__main__':
    #write_label("G:\pyworkspace\graph-embedding\data\subway.edgelist")
    a = [1, 2, 3]
    c, d, e = a
    print(c, d, e)



