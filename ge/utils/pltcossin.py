# -*- coding:utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def f(p, t):
    length = len(p)
    cos = 0.0
    sin = 0.0
    for i in range(length):
        cos += np.cos(t*p[i])
        sin += np.sin(t*p[i])

    cos /= length
    sin /= length
    return cos, sin


def compare(p, q, scale):
    pcos, psin, qcos, qsin = [], [], [], []
    x = [i for i in range(scale)]
    for i in range(scale):
        cos, sin = f(p, i)
        pcos.append(cos)
        psin.append(sin)
        cos, sin = f(q, i)
        qcos.append(cos)
        qsin.append(sin)

    plt.figure()
    plt.plot(x, pcos, label='节点7 实部',color='red')
    plt.plot(x, psin, label='节点7 虚部',color="green")
    plt.plot(x, qcos, label='节点6 实部',color="blue")
    plt.plot(x, qsin, label='节点6 虚部',color="pink")
    for t in range(0, scale, 20):
        x1 = [t] * 4
        y1 = [pcos[t], psin[t], qcos[t], qsin[t]]
        plt.scatter(x1, y1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    p = np.array([0.013064725,0.03608222,0.1196094,0.06859553,0.035123084,0.14102638,0.22639008,0.11952891,0.001000638,0.0009936608,0.0065047955,0.034392595,0.007391006,0.0065869656,0.14094475,0.03528574,0.007479533
])
    q = np.array([0.046537716,0.0146652805,0.036082223,0.013064725,0.17013028,0.43254536,0.14102638,0.035123084,0.008106486,0.0011707481,0.0010086695,0.0064984076,0.0009931505,0.045644682,0.039815243,0.006586964,0.0010006028
])
    compare(p, q, 500)