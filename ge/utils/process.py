# -*- encoding: utf-8 -*-

fin = open("../../data/usa2.edgelist", mode="r", encoding="utf8")
fin1 = open("../../data/usa2_SIR.label", mode="r", encoding="utf8")
fout = open("../../data/usa.edgelist", mode="w+", encoding="utf8")
fout1 = open("../../data/usa_SIR.label", mode="w+", encoding="utf8")


dic = dict()
idx = 0
while True:
    line = fin.readline()
    if not line:
        break
    left, right = line.strip().split(" ")
    if left not in dic:
        dic[left] = idx
        idx += 1
    if right not in dic:
        dic[right] = idx
        idx += 1

    fout.write("{} {}\n".format(dic[left], dic[right]))

while True:
    line = fin1.readline()
    if not line:
        break
    node, label = line.strip().split(" ")
    fout1.write("{} {}\n".format(dic[node], label))

fin.close()
fin1.close()
fout.close()
fout1.close()