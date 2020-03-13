# -*- coding:utf-8 -*-

import numpy as np
import math


def create_alias_table(probs):
    """
    calculate the alias sample table.
    :param probs: sum(area_ratio)= 1 or N
    :return: accept, alias
    """
    if not isinstance(probs, (list, np.ndarray)):
        raise TypeError("The probs type must be list or np.ndarray.")
    probs = np.asarray(probs)
    N = len(probs)
    probs_sum = probs.sum()
    if math.isclose(probs_sum, 1.0):
        probs *= N
    elif math.isclose(probs_sum, N):
        pass
    else:
        raise ValueError("The probs sum must be 1.0 (normalized) or N (for alias), get{}.".format(probs_sum))


    accept, alias = [0] * N, [0] * N
    small, large = [], []

    for i, prob in enumerate(probs):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = probs[small_idx]
        alias[small_idx] = large_idx
        probs[large_idx] -= (1 - probs[small_idx])
        if probs[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """
    
    :param accept: probability table
    :param alias: alias table
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random() * N)
    prob = np.random.random()
    if prob < accept[i]:
        return i
    else:
        return alias[i]
