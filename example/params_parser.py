# -*- encoding: utf-8 -*-

import argparse
from texttable import Texttable

"""
不同嵌入方法的命令行参数parser
"""

def GraphWaveParameterParser():
    """
    GraphWave参数
    """
    parser = argparse.ArgumentParser(prog="GraphWave", description = "Run WaveletMachine.")

    parser.add_argument('--graph', type=str, default="bio_grid_human",
	                    help = 'Network graph name.')

    parser.add_argument('--scale', type = float, default = 2,
	                    help = 'Heat-coefficent, i.e, the scale parameter. Default is 2')

    parser.add_argument('--sample-number', type = int, default = 50,
	                    help = 'Number of characteristic function sample points. Default is 50.')

    parser.add_argument('--step-size', type=int, default=10,
                        help='Step size, default is 20.')

    parser.add_argument('--approximation', type = int, default = 100,
	                    help = 'Number of Chebyshev approximation. Default is 100.')

    parser.add_argument('--tsne', type=int, default=30,
                        help='TSNE perplexity, default is 10.')

    parser.add_argument('--random', type=int, default=42,
                        help='Random seed, default is 42.')

    return parser.parse_args()


def HSDParameterParser():
    """
    HSD命令行参数
    """
    parser = argparse.ArgumentParser(prog="HSD", description="Hierarchically Structural Distance")
    parser.add_argument("--graph", type=str, default="bio_dmela",
                        help='Network graph name: barbell, mkarate, europe, usa')

    parser.add_argument("--scale", type=float, default=1.0,
                        help='Heat-coefficent, i.e, the scale, default is 2.')

    parser.add_argument("--metric", type=str, default="hellinger",
                        help='Distance metric: wasserstein, hellinger, default is wasserstein.')

    parser.add_argument("--hop", type=int, default=3,
                        help="Max hop of local neighborhood, default is 5.")

    parser.add_argument("--tsne", type=int, default=30,
                        help="Perplexity of TSNE model, default is 30.")

    parser.add_argument("--test_size", type=float, default=0.3,
                        help="Test set size, default is 0.3.")

    parser.add_argument("--neighbors", type=int, default=10,
                        help="Number of neighbors in KNN evaluate model.")

    parser.add_argument('--cv', type=int, default=5,
                        help="Cross validation in evaluate process, default is 5.")

    parser.add_argument('--random', type=int, default=42,
                        help="Random seed, default is 42.")

    parser.add_argument('--workers', type=int, default=3,
                        help="Number of process workers.")

    parser.add_argument('--multi_scales', type=str, default='yes',
                        help="Employ multi scales analysis, default is No.")

    parser.add_argument('--embedding_method', type=str, default='LLE',
                        help="Embedding methods, LE - Laplacian Eigenmaps, LLE - Locally Linear Embedding")

    parser.add_argument('--dim', type=int, default=64,
                        help="Embedding vector dimension, default is 64.")

    parser.add_argument('--sparse', type=float, default=0.9,
                        help="Remove how many redundant edges from new graph, default is 0.9.")

    parser.add_argument('--reuse', type=str, default="no",
                        help="Reuse the distance output computed before.")


    return parser.parse_args()


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    tab = Texttable()
    tab.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(tab.draw())