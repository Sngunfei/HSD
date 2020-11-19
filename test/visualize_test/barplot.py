# -*- encoding: utf-8 -*-

"""
mkarate度数横着的柱状图
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns

from model.multiscale_HSD import MultiHSD
from tools.hierarchy import read_hierarchy

def fetch_data():
    # 获取mkarate节点度数
    # 邻域（4-hop）节点总个数 vs 直接邻居个数

    graph = nx.read_edgelist(f"../../data/graph/mkarate.edgelist", create_using=nx.Graph, edgetype=float,
                             data=[('weight', float)])

    model = MultiHSD(graph, "mkarate", 2, 100)
    model.init2()
    model.hierarchy = read_hierarchy(f"../../data/hierarchy/mkarate2.layers", 3)

    all_nums = [0] * 34
    nei_nums = [0] * 34
    for node, layers in model.hierarchy.items():
        node = int(node)
        if node > 33:
            continue
        nei_cnt = len(layers[1])
        all_cnt = sum([len(layers[i]) for i in range(0, 3)])
        nei_nums[node] = float(nei_cnt)
        all_nums[node] = float(all_cnt)

    df = pd.DataFrame(data={'nei': nei_nums,
                            'all': all_nums,
                            'node': [str(i + 1) for i in range(34)]})
    #df = df.sort_values('nei', ascending=False)

    return df

def plot():
    sns.set_style(style="whitegrid")

    # Initialize the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 30))

    # Load the example car crash dataset
    df = fetch_data()

    """
    sort_index = list(range(35))
    def mycmp(i, j):
        if neighbor_cnt[i] == neighbor_cnt[j]:
            return all_cnt[i] - all_cnt[j]
        return neighbor_cnt[i] - neighbor_cnt[j]
    sort_index.sort(key=cmp_to_key(mycmp))
    sort_index = sort_index[::-1]
    """

    # Plot the total crashes
    sns.set_color_codes("pastel")
    ax = sns.barplot(x='all', y='node', data=df,
                     label="all", color="b")

    # Plot the crashes where alcohol was involved
    sns.set_color_codes("muted")
    ax = sns.barplot(x='nei', y='node', data=df,
                     label="nei", color="b")

    # Add a legend and informative axis label
    #ax.legend(ncol=1, loc="lower right", frameon=True)
    ax.set(xlim=(0, 60))
    plt.xlabel(xlabel="Number of nodes in local neighborhood.",
               fontsize=30)
    plt.ylabel(ylabel="node", fontsize=30)
    plt.yticks(fontsize=30)
    #sns.despine(left=True, bottom=True)
    plt.savefig("mkarate_barplot.png")
    plt.show()

if __name__ == '__main__':
    data = fetch_data()
    print(data)
    plot()
