# -*- coding:utf-8 -*-

"""
Calculate different distances between distributions.
"""

import numpy as np
from numpy import linalg
from scipy import stats


def calc_pq_distance(p, q, metric="l1", normalized=False):
    """
    计算两个分布之间的相似性，p和q分别为两个节点，在同一层环上的小波系数分布，可能并不等长，为了突出节点度数
    对于节点结构性质的重要性，我们用 0 将分布对齐。
    :param metric: 相似性度量标准，默认为L1
    :param normalized: 是否将p，q放缩成正常的概率分布，因为p，q分别求和后的结果不一定相等。
    :return: 两个分布之间的相似性
    """
    length = max(len(p), len(q))
    p = p + (length - len(p)) * [0]
    q = q + (length - len(q)) * [0]
    p = np.sort(p)
    q = np.sort(q)

    if normalized:
        p = 1.0 * p / np.sum(p) if np.sum(p) > 0.0 else p
        q = 1.0 * q / np.sum(q) if np.sum(q) > 0.0 else q

    return calc_distance(p, q, metric)


def _check_prob_distri(p, q):
    """
    验证两个概率分布是否有效，以供后续计算两者相似性。
    """
    if not (isinstance(p, (list, np.ndarray)) and isinstance(q, (list, np.ndarray))):
        raise TypeError("The probability distribution type must be list or ndarray")
    assert len(p) == len(q), "Length of p({}) must be equal to length of q({})".format(len(p), len(q))
    """
    if not math.isclose(sum(p), 1.0) or not math.isclose(sum(q), 1.0):
        raise ArithmeticError("The sum of probability distribution function is not 1.0")
    """


def wasserstein_distance(p, q, dual=False):
    """
    优化求解两个分布之间的2阶wasserstein距离
    """
    length = len(p)
    D = np.zeros((length, length))  # d(x, y)

    for i in range(length):
        for j in range(length):
            D[i, j] = np.abs(p[i] - q[j])

    A_r = np.zeros((length, length, length))
    A_t = np.zeros((length, length, length))

    for i in range(length):
        for j in range(length):
            A_r[i, i, j] = 1
            A_t[i, j, i] = 1

    A = np.concatenate((A_r.reshape((length, length ** 2)), A_t.reshape((length, length ** 2))), axis=0)
    b = np.concatenate((p, q), axis=0)
    c = D.reshape((length ** 2))
    if dual:
        opt_res = linprog(-b, A.transpose(), c, bounds=(None, None))
        emd = -opt_res.fun
    else:
        opt_res = linprog(c, A_eq=A, b_eq=b)
        emd = opt_res.fun
    return emd


def wasserstein_guass(p, q):
    """
    在欧式空间中计算两个n维高斯分布之间的2阶Wasserstein距离，解析解如下：
        W(u1，u2)^2 = |m1 - m2|^2 + tr(C1 + C2 - 2(C2^{-1/2}C1C2^{1/2})^{1/2})
    其中m1和m2分别为两个分布的均值即中心，C1和C2是对应的协方差矩阵，对称半正定。
    当维度为1时，就退化到两个正态分布之间的距离，公式可写为：
        W(u1, u2)^2 = |m1 - m2|^2 + sqrt((\sigma1 + \sigma2 - 2(\sigma1\sigma2)))
    这时候就能很好的说明W式距离能够刻画分布之间的几何距离：中心点之间的距离 + 分布形状之间的距离
    :param p: First guass distribution.
    :param q: Second guass distribution
    :return: The 2th Wasserstein distance between two Guass.
    """
    _check_prob_distri(p, q)
    m1 = np.mean(p)
    m2 = np.mean(q)
    sigma1 = np.mean(np.square(p - m1))
    sigma2 = np.mean(np.square(q - m2))
    dist = (m1 - m2) ** 2 + sigma1 + sigma2 - 2 * (sigma1 * sigma2) ** 0.5
    return dist


def KL_divergence(p, q, symmetric=False):
    """
    计算两个分布之间的Kullback–Leibler散度，公式如下：
        KL(p | q) = Σplog(p / q)
    注意到KL散度实际上并不属于距离度量，因为不对称，求KL散度的均值可以解决。
    :param p: First probability distribution
    :param q: Second probability distribution
    :param symmetric: if True, calculate symmetric KL divergence
    :return: The KL divergence between p, q
    """
    _check_prob_distri(p, q)
    kl_pq = np.sum(p * np.log(p / q))
    if symmetric:
        kl_qp = np.sum(q * np.log(q / p))
        return (kl_pq + kl_qp) / 2.0
    return kl_pq


def JS_divergence(p, q):
    """
    计算两个分布之间的Jensen–Shannon散度，定义如下：
        JS(P | Q) = [KL(P | M) + KL(Q | M)] / 2， M = （P + Q) / 2
    可以看到JS散度是基于KL散度定义的，更加平滑，而且对称。
    :param p: First probability distribution
    :param q: Second probability distribution
    :return: The JS divergence between p, q
    """
    _check_prob_distri(p, q)
    m = p + q
    js = (KL_divergence(p, m, False) + KL_divergence(q, m, False)) / 2.0
    return js


def L_1_2_distance(p, q, order):
    """
    计算两个分布之间的 L1 or L2 距离。
    """
    _check_prob_distri(p, q)
    if order == 1:
        return np.sum(np.abs(p - q))
    elif order == 2:
        return np.sum(np.square(p - q))


def Dynamic_Time_Warping(p, q):
    """
    动态时间规整，DTW
    :param p:
    :param q:
    :return:
    """
    raise NotImplementedError("Dynamic_Time_Warping is not implemented yet.")


def calc_distance(p, q, metric=None):
    """
    计算两个概率分布之间的距离。
    :param metric: 距离名称, str
    :return: 距离大小，float
    """
    if not metric or not isinstance(metric, str):
        raise TypeError("Need to specify a metric")
    sup_metrics = ['l1', 'l2', 'kl', 'skl', 'js', 'wguass', 'wasserstein']
    if str.lower(metric) not in sup_metrics:
        raise NotImplementedError("The {} metric is not supported yet.".format(metric))
    metric = str.lower(metric)
    if metric == 'l1':
        return L_1_2_distance(p, q, 1)
    elif metric == 'l2':
        return L_1_2_distance(p, q, 2)
    elif metric == 'kl':
        return KL_divergence(p, q)
    elif metric == 'skl':
        return KL_divergence(p, q, True)
    elif metric == 'js':
        return JS_divergence(p, q)
    elif metric == 'wguass':
        return wasserstein_guass(p, q)
    elif metric == 'wasserstein':
        #return wasserstein_distance(p, q)
        return stats.wasserstein_distance(p, q) * len(p)


def Hellinger_distance(p, q):
    """
    计算两个概率分布之间的hellinger distance
    wiki：https://en.wikipedia.org/wiki/Bhattacharyya_distance
    :param p:
    :param q:
    :return:
    """
    import math
    assert len(p) == len(q)
    assert math.isclose(sum(p), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(q), 1.0, abs_tol=1e-6)
    BC = 0.0
    for px, qx in zip(p, q):
        BC += math.sqrt(px * qx)
    if math.isclose(BC, 0.0, abs_tol=1e-6):
        BC = 0.0
    elif math.isclose(BC, 1.0, abs_tol=1e-6):
        BC = 1.0

    hellinger_distance = math.sqrt(1.0 - BC)
    return hellinger_distance
