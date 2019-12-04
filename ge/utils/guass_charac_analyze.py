# -*- encoding:utf-8 -*-

"""

分析两个高斯分布之间的特征函数，之间的距离。

"""
import numpy as np
import matplotlib.pyplot as plt

def guass_charac():

    first = np.random.normal(20, 5, 100)
    second = np.random.normal(50, 10, 100)

    first = first / np.sum(first)
    second = second / np.sum(second)

    kl_divergence = 0.0
    print(first)
    print(second)
    for p, q in zip(first, second):
        print(p, q)
        kl_divergence += p * np.log(p / q)

    print(kl_divergence)

    x = [i for i in range(0, 5000, 10)]

    real_1, imag_1 = charac(first, x)
    real_2, imag_2 = charac(second, x)

    #plt.plot(x, real_1, "bo")
    #plt.plot(x, real_2, "ro")
    #plt.plot(x, imag_1, "bx")
    #plt.plot(x, imag_2, "rx")

    real_diff = [abs(i-j) for i, j in zip(real_1, real_2)]
    imag_diff = [abs(i-j) for i, j in zip(imag_1, imag_2)]

    diff = [i+j for i, j in zip(real_diff, imag_diff)]

    plt.plot(x, real_diff, "b-")
    plt.plot(x, imag_diff, "r-")
    plt.plot(x, diff, "k-")
    plt.plot(x, [kl_divergence]*len(x), "g-")


    plt.legend()
    plt.show()

    return


def charac(p:np.array, samples:list):
    """
    特征函数，= mean（e^(itx)）
    :param p:  概率分布
    :return:
    """
    res_real = []
    res_imag = []
    for t in samples:
        value = np.mean(np.exp(1j * p * t))
        res_real.append(value.real)
        res_imag.append(value.imag)

    return res_real, res_imag


if __name__ == '__main__':
    guass_charac()



