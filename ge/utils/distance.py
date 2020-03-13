# -*- coding:utf-8 -*-

"""
Distance metric functions:
- L1
- L2
- KL divergence
- JS divergence
- Wasserstein distance
- Hellinger distance
"""

import numpy as np
import math
from scipy import stats


def align_probablity_distribution(p, q, normalized=False):
    """
    为了突出节点度数, 用 0 将两个分布对齐。
    :param p:
    :param q:
    :param normalized: 是否将p，q放缩成正常的概率分布。
    :return:
    """
    length = max(len(p), len(q))
    p = p + (length - len(p)) * [0.0]
    q = q + (length - len(q)) * [0.0]
    p = np.sort(p)
    q = np.sort(q)

    if normalized:
        p = 1.0 * p / np.sum(p) if np.sum(p) > 0.0 else p
        q = 1.0 * q / np.sum(q) if np.sum(q) > 0.0 else q

    return p, q


def check_probablity_distribution(p, q):
    """
    check probablity distribution.
    """
    if not (isinstance(p, (list, np.ndarray)) and isinstance(q, (list, np.ndarray))):
        raise TypeError("The probability distribution must be list or ndarray.")

    if len(p) != len(q):
        raise TypeError("Length of p({}) must be equal to length of q({})".format(len(p), len(q)))

    if not math.isclose(np.sum(p) - 1.0, 0.0, abs_tol=1e-6) or \
        not math.isclose(np.sum(q) - 1.0, 0.0, abs_tol=1e-6):
        raise ValueError("The sum of probability distribution must be 1.0.")


def wasserstein_guass_distance(p, q):
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
    u1 = np.mean(p)
    u2 = np.mean(q)
    sigma1 = np.mean(np.square(p - u1))
    sigma2 = np.mean(np.square(q - u2))
    distance = (u1 - u2) ** 2 + sigma1 + sigma2 - 2 * (sigma1 * sigma2) ** 0.5
    return distance


def KL_divergence(p, q, symmetric=False):
    """
    计算两个分布之间的Kullback–Leibler散度，公式如下：
        KL(p | q) = Σplog(p / q)
    :param p: probability distribution. list[float]
    :param q: probability distribution, list[float]
    :param symmetric: if True, calculate symmetric KL divergence
    :return: KL divergence between p, q
    """
    check_probablity_distribution(p, q)
    KL_pq = np.sum(p * np.log(p / q))
    if symmetric:
        KL_qp = np.sum(q * np.log(q / p))
        return (KL_pq + KL_qp) / 2.0
    return KL_pq


def JS_divergence(p, q):
    """
    计算两个分布之间的Jensen–Shannon散度，定义如下：
        JS(P | Q) = [KL(P | M) + KL(Q | M)] / 2， M = （P + Q) / 2
    可以看到JS散度是基于KL散度定义的，更加平滑，而且对称。
    :param p: First probability distribution
    :param q: Second probability distribution
    :return: The JS divergence between p, q
    """
    check_probablity_distribution(p, q)
    m = p + q
    js = (KL_divergence(p, m, False) + KL_divergence(q, m, False)) / 2.0
    return js


def L_distance(p, q, order):
    """
    计算两个分布之间的 L1 or L2 距离。
    """
    check_probablity_distribution(p, q)
    if order == 1: # L1
        return np.sum(np.abs(p - q))
    elif order == 2: # L2
        return np.sum(np.square(p - q))


def hellinger_distance(p, q):
    """
    计算两个概率分布之间的hellinger distance
    wiki：https://en.wikipedia.org/wiki/Bhattacharyya_distance
    :param p:
    :param q:
    :return:
    """
    check_probablity_distribution(p, q)

    BC = 0.0
    for px, qx in zip(p, q):
        BC += math.sqrt(px * qx)
    if math.isclose(BC, 0.0, abs_tol=1e-6):
        BC = 0.0
    elif math.isclose(BC, 1.0, abs_tol=1e-6):
        BC = 1.0

    distance = math.sqrt(1.0 - BC)
    return distance


def Dynamic_Time_Warping(p, q):
    """
    动态时间规整，DTW
    :param p:
    :param q:
    :return:
    """
    raise NotImplementedError("Dynamic_Time_Warping is not implemented yet.")


def calculate_distance(p, q, metric):
    """
    calculate distance between probabilities (p ,q)
    :param p:
    :param q:
    :param metric: str
    :return: distance，float
    """

    if not metric or not isinstance(metric, str):
        raise TypeError("Need to specify a metric.")

    supported_metrics = ['l1', 'l2', 'kl', 'symmetric_kl', 'js', 'wasserstein_guass', 'wasserstein', 'hellinger']
    metric = str.lower(metric)

    if metric not in supported_metrics:
        raise NotImplementedError("{} metric is not implemented.".format(metric))

    p, q = align_probablity_distribution(p, q)
    if len(p) == 0 and len(q) == 0:
        return 0.0

    metric = str.lower(metric)
    if metric == 'l1':
        return L_distance(p, q, order=1)
    elif metric == 'l2':
        return L_distance(p, q, order=2)
    elif metric == 'kl':
        return KL_divergence(p, q, symmetric=False)
    elif metric == 'symmetric_kl':
        return KL_divergence(p, q, symmetric=True)
    elif metric == 'js':
        return JS_divergence(p, q)
    elif metric == 'wasserstein_guass':
        return wasserstein_guass_distance(p, q)
    elif metric == 'wasserstein':
        return stats.wasserstein_distance(p, q) * len(p)
    elif metric == 'hellinger':
        return hellinger_distance(p, q)


