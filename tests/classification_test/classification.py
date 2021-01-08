# -*- encoding: utf-8 -*-

from tools.evaluate import KNN_evaluate
from tools.rw import read_vectors
from tools.dataloader import read_label
from tools.util import merge_dicts_to_lists


def classfication_test():
    vector_path = "output/struc2vec_cora_64_2.csv"
    label_path = "../../data/label/cora_EigenCentrality.label"

    vector_dict = read_vectors(vector_path)
    label_dict = read_label(label_path)

    vectors, labels = merge_dicts_to_lists(vector_dict, label_dict)
    KNN_evaluate(vectors, labels, cv=5, n_neighbor=10)


if __name__ == '__main__':
    classfication_test()
