# -*- coding:utf-8 -*-
### 多进程

import multiprocessing as mp
import numpy as np
from multiprocessing import Queue, Pipe

def worker(arr, start, q):
    origin = arr[start]
    dis = {}
    for i in range(start+1, len(arr)):
        d = np.sum(np.square(origin - arr[i]))
        dis[i] = d
    q.put((start, dis))
    return

if __name__ == '__main__':

    array = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    distance = np.zeros(shape=(len(array), len(array)), dtype=np.float)
    jobs = []
    queue = Queue()
    for i in range(4):
        p = mp.Process(target=worker, args=(array, i, queue, ), name="row{}".format(i))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()
    for _ in jobs:
        start, dis = queue.get()
        for k, v in dis.items():
            print(start, k, v)
            distance[start, k] = distance[k, start] = v

    print(distance)