
"""
statistic

2-HSD-multi,h:0.5221621136917932,c:0.6590885384632814,v:0.5826893110723829,s:0.2084582270602763,a:0.828125,M:0.6844969624067458,m:0.828125

"""

def statistic():
    fin = open("E:\workspace\py\graph-embedding\example\\res_all_label10.txt", mode="r", encoding="utf-8")

    methods = ['graphwave', 'HSD-single', 'HSD-multi', 'struc2vec', 'node2vec', 'rolx']
    res = {}
    for m in methods:
        res[m] = dict()

    while True:
        line = fin.readline()
        if not line:
            break
        infos = line.strip().split(",")
        metadata = infos[0]
        method = ""
        for x in methods:
            if metadata.find(x) > 0:
                method = x
                break

        for idx, item in enumerate(infos):
            if idx == 0:
                continue
            metric, value = item.strip().split(':')
            value = float(value)
            res[method][metric] = res[method].get(metric, list()) + [value]

    for method, result in res.items():
        s = method + ": "
        for metric, values in result.items():
            s += metric + "=" + str(sum(values) / len(values)) + ","
        print(s)


if __name__ == '__main__':
    statistic()
