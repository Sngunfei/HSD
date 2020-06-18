"""
综合对比各算法之间的性能
"""

from example import graphwave, HSD, node2vec, struc2vec, RolX
from ge.graph.random_graph import VariedGraph
from ge.utils.robustness import random_add_edges
from ge.SIR.SIR import SIR_labels
import networkx as nx
from tqdm import tqdm
from sklearn import metrics


def _single_run(graph, labels, fout, idx, perp=10):
    label_set = set()
    for _node, _label in labels.items():
        label_set.add(_label)
    n_class = len(label_set)

    methods = {"HSD-single": HSD,
               "HSD-multi": HSD,
               "graphwave": graphwave,
               "struc2vec": struc2vec,
               "node2vec": node2vec,
               "rolx": RolX}

    for i in range(1):
        g = random_add_edges(graph, ratio=0.1)
        sir_labels = SIR_labels(nx.Graph(g), n_class, 5, 1.0, 0.0)
        multi_labels = {"SIR": sir_labels, "origin": labels}
        #multi_labels = {"SIR": sir_labels}
        #n_class = 5
        for name, method in methods.items():
            print(name, method)
            if name == "HSD-single":
                res = method.exec(nx.Graph(g), multi_labels, n_class, mode=0, perp=perp)
            elif name == "HSD-multi":
                res = method.exec(nx.Graph(g), multi_labels, n_class, mode=1, perp=perp)
            else:
                res = method.exec(nx.Graph(g), multi_labels, n_class, perp=perp)

            for label_name, _res in res.items():
                h, c, v, s, a, M, m = _res
                fout.write(f"{i}-{name}-{label_name},h:{h},c:{c},v:{v},s:{s},a:{a},M:{M},m:{m}\n")
                fout.flush()


def compare():
    data_generator = VariedGraph()
    fout = open("E:\workspace\py\graph-embedding\\results\\res_hop2.txt", mode="a+", encoding="utf-8")
    for idx in tqdm(range(5)):
        graph, labels = data_generator.get_graph(random_seed=idx)
        sir_labels = SIR_labels(nx.Graph(graph), 10, 5, 1.0, 0.0)
        label1 = [labels[node] for node in nx.nodes(graph)]
        label2 = [sir_labels[node] for node in nx.nodes(graph)]
        h, c, v = metrics.homogeneity_completeness_v_measure(label1, label2)
        print(idx, "-origin : ", h, c, v)

        g = random_add_edges(graph, ratio=0.1)
        sir_labels_1 = SIR_labels(nx.Graph(g), 10, 5, 1.0, 0.0)
        label3 = [sir_labels_1[node] for node in nx.nodes(graph)]
        h, c, v = metrics.homogeneity_completeness_v_measure(label1, label3)
        print(idx, "-noise : ", h, c, v)
        h, c, v = metrics.homogeneity_completeness_v_measure(label2, label3)
        print(idx, "-sir : ", h, c, v)
        print(label2)
        print(label3)
        #_single_run(graph, labels, fout, idx, perp=2)

    fout.close()


if __name__ == '__main__':

    #a = [1, 2, 1, 3, 1, 3, 1, 2, 1, 2]
    #b = [20, 30, 20, 40, 20, 40, 20, 30, 20, 30]
    #acc = metrics.accuracy_score(a, b)
    #h, c, v = metrics.homogeneity_completeness_v_measure(a, b)
    #print(h, c, v, acc)
    compare()



