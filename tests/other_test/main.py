# -*- encoding: utf-8 -*-

# Student-t分布

from scipy.stats import wasserstein_distance
import numpy as np


def wasserstein_distance_test():
    p = [1, 2, 3, 4, 5]
    q = [15]
    dist = wasserstein_distance(p, q)
    print(dist)


def powers_of_laplacian():
    A = np.array(
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]])

    D = np.array(
        [[2, 0, 0, 0],
         [0, 2, 0, 0],
         [0, 0, 2, 0],
         [0, 0, 0, 2]])

    L = np.array(
        [[2, -1, -1],
         [-1, 1, 0],
         [-1, 0, 1]])
    all_paths = 32
    print(np.dot(L, L))


if __name__ == '__main__':
    #wasserstein_distance_test()
    powers_of_laplacian()
