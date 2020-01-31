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

    x = [i for i in range(0, 10000, 10)]

    real_1, imag_1 = charac(first, x)
    real_2, imag_2 = charac(second, x)

    #plt.plot(x, real_1, "bo")
    #plt.plot(x, real_2, "ro")
    #plt.plot(x, imag_1, "bx")
    #plt.plot(x, imag_2, "rx")

    real_diff = [abs(i-j) for i, j in zip(real_1, real_2)]
    imag_diff = [abs(i-j) for i, j in zip(imag_1, imag_2)]

    diff = [np.sqrt(i**2+j**2) for i, j in zip(real_diff, imag_diff)]

    #plt.plot(x, real_diff, "b-")
    #plt.plot(x, imag_diff, "r-")
    plt.plot(x, diff, "k-", label="real^2 + imag^2")
    plt.plot(x, [kl_divergence]*len(x), "g-", label="KL")


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
        print(p)
        value = np.mean(np.exp(1j * p * t))
        res_real.append(value.real)
        res_imag.append(value.imag)

    return res_real, res_imag


def mkarate_wavelet_analyse(scale, node1, w1:list, node2, w2:list, node3, w3:list, s1:float, s2:float):
    """
    以mirror-karate network中的节点举例，其中w1和w2是结构对称的小波系数，而w3是不对称的，
    分析一下它们的特征函数差异
    :param w1:
    :param w2:
    :param w3:
    :return:
    """
    w1, w2, w3 = np.sort(w1),  np.sort(w2),  np.sort(w3)
    samples = [i for i in range(0, 2000, 5)]
    real_1, imag_1 = charac(w1, samples)
    real_2, imag_2 = charac(w2, samples)
    real_3, imag_3 = charac(w3, samples)

    fout=open("../../output/samples.txt", mode="a+", encoding="utf-8")
    fout.write("scale = {}, node{}, node{}, node{}\n".format(scale, node1, node2, node3))
    fout.write("node{}: \n".format(node1))
    fout.write(",".join(map(str, samples)))
    fout.write("\n")
    fout.write(",".join(map(str, real_1)))
    fout.write("\n")
    fout.write(",".join(map(str, imag_1)))
    fout.write("\n")

    fout.write("node{}: \n".format(node2))
    fout.write(",".join(map(str, samples)))
    fout.write("\n")
    fout.write(",".join(map(str, real_2)))
    fout.write("\n")
    fout.write(",".join(map(str, imag_2)))
    fout.write("\n")

    fout.write("node{}: \n".format(node3))
    fout.write(",".join(map(str, samples)))
    fout.write("\n")
    fout.write(",".join(map(str, real_3)))
    fout.write("\n")
    fout.write(",".join(map(str, imag_3)))
    fout.write("\n\n")

    font = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
    }

    plt.figure()
    #plt.subplot(221)
    plt.plot(samples, real_1, 'b-', label="Node{} - Real".format(node1))
    plt.plot(samples, imag_1, 'r-', label="Node{} - Imag".format(node1))
    plt.plot(samples, real_2, 'g-', label="Node{} - Real".format(node2))
    plt.plot(samples, imag_2, 'k-', label="Node{} - Imag".format(node2))
    plt.xlabel("Sample points", fontdict=font)
    plt.ylabel("Value", fontdict=font)
    #plt.plot(samples, real_3, 'g-', label="real_3")
    #plt.plot(samples, imag_3, 'g-', label="imag_3")
    plt.legend(prop=font)

    #plt.subplot(222)
    plt.figure()

    plt.plot(samples, real_3, 'g-', label="Node{} - Real".format(node3))
    plt.plot(samples, imag_3, 'k-', label="Node{} - Imag".format(node3))
    plt.plot(samples, real_1, 'b-', label="Node{} - Real".format(node1))
    plt.plot(samples, imag_1, 'r-', label="Node{} - Imag".format(node1))
    plt.xlabel("Sample points", fontdict=font)
    plt.ylabel("Value", fontdict=font)

    #plt.plot(samples, real_3, 'g-', label="real_3")
    #plt.plot(samples, imag_3, 'g-', label="imag_3")
    plt.legend(prop=font)

    #plt.subplot(223)
    plt.figure()

    real_diff = [abs(i - j) for i, j in zip(real_1, real_2)]
    imag_diff = [abs(i - j) for i, j in zip(imag_1, imag_2)]

    diff = [np.sqrt(i ** 2 + j ** 2) for i, j in zip(real_diff, imag_diff)]
    plt.plot(samples, diff, 'b-', label="Nodes ({}, {}) Euclidean distance".format(node1, node2))
    plt.plot(samples, [s1]*len(samples), 'k-', label="Nodes ({}, {}) hierachical wasserstein distance".format(node1, node2))

    #plt.subplot(224)
    real_diff = [abs(i - j) for i, j in zip(real_1, real_3)]
    imag_diff = [abs(i - j) for i, j in zip(imag_1, imag_3)]
    diff = [np.sqrt(i ** 2 + j ** 2) for i, j in zip(real_diff, imag_diff)]
    plt.plot(samples, diff, 'r-',  label="Nodes ({}, {}) Euclidean distance".format(node1, node3))
    plt.plot(samples, [s2] * len(samples), 'y-', label="Nodes ({}, {}) hierachical wasserstein distance".format(node1, node3))
    plt.xlabel("Sample points", fontdict=font)
    plt.ylabel("Distance", fontdict=font)
    plt.legend(prop=font)
    plt.show()


def mkarate_wavelet_analyse_2(w1:list, w2:list, w3:list):
    """
    以mirror-karate network中的节点举例，其中w1和w2是结构对称的小波系数，而w3是不对称的，
    分析一下它们的特征函数差异
    :param w1:
    :param w2:
    :param w3:
    :return:
    """
    w1, w2, w3 = np.sort(w1),  np.sort(w2),  np.sort(w3)
    samples = [i for i in range(0, 1000, 5)]
    real_1, imag_1 = charac(w1, samples)
    real_2, imag_2 = charac(w2, samples)
    real_3, imag_3 = charac(w3, samples)

    plt.subplot(221)
    plt.plot(real_1, imag_1, 'b-', label="1")
    plt.plot(real_2, imag_2, 'r-', label="2")
    plt.plot(real_3, imag_3, 'k-', label="3")

    plt.subplot(222)
    real_diff = [abs(i - j) for i, j in zip(real_1, real_2)]
    imag_diff = [abs(i - j) for i, j in zip(imag_1, imag_2)]
    KL_12 = 0.0
    for p, q in zip(w1, w2):
        KL_12 += p * np.log(p / q)

    diff = [np.sqrt(i ** 2 + j ** 2) for i, j in zip(real_diff, imag_diff)]
    plt.plot(samples, diff, 'b-')
    plt.plot(samples, [KL_12]*len(samples), 'k-')

    plt.subplot(223)
    real_diff = [abs(i - j) for i, j in zip(real_1, real_3)]
    imag_diff = [abs(i - j) for i, j in zip(imag_1, imag_3)]
    KL_13 = 0.0
    for p, q in zip(w1, w3):
        KL_13 += p * np.log(p / q)
    plt.plot(samples, [KL_13] * len(samples), 'k-')
    diff = [np.sqrt(i ** 2 + j ** 2) for i, j in zip(real_diff, imag_diff)]
    plt.plot(samples, diff, 'r-')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    guass_charac()



