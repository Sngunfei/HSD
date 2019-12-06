# -*- coding:utf-8 -*-

import pymongo
import logging


class Database(object):

    def __init__(self, host=None, db_name='ge'):
        if not host:
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


    def insert_score(self, info):
        col = self.db['scores']
        res = col.insert_one(info)
        logging.info("MongoDB insert scores: {}".format(info))
        return res.inserted_id


    def find(self, collection, filters):
        col = self.db[collection]
        res = col.find(filters, {"prob": 1, "scores": 1, "evaluate": 1})
        return res

    def close(self):
        self.client.close()
