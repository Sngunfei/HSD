# -*- coding:utf-8 -*-

import pymongo
import logging

"""
连接MongoDB，存储实验结果
"""

class Mongo:

    def __init__(self, host=None, db_name='ge'):
        if host is None:
            host = "mongodb://localhost:27017/"
        self.client = pymongo.MongoClient(host)
        self.db = self.client[db_name]


    def insert_time(self, dataset, n_nodes, n_edges, time):
        col = self.db['time']
        doc = {"name": dataset,
               "n_nodes": n_nodes,
               "n_edges": n_edges,
               "time": time}
        res = col.insert_one(doc)
        logging.info("MongoDB insert time: {}".format(doc))
        return res.inserted_id


    def insert(self, info, collection):
        col = self.db[collection]
        res = col.insert_one(info)
        logging.info("MongoDB insert {}: {}".format(collection, info))
        print("MongoDB insert {}: {}\n".format(collection, info))
        return res.inserted_id


    def find(self, collection, filters):
        col = self.db[collection]
        res = col.find(filters, {"method": 1, "scores":1, "prob": 1, "evaluate model": 1, "accuracy": 1, "micro f1": 1, "macro f1": 1})
        return res

    def close(self):
        self.client.close()
