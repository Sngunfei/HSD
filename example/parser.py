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

    parser.add_argument('--graph', type=str,
	                    help = 'Network graph name.')

    parser.add_argument('--scale', type = float, default = 2,
	                    help = 'Heat-coefficent, i.e, the scale parameter. Default is 2')

    parser.add_argument('--sample-number', type = int, default = 50,
	                    help = 'Number of characteristic function sample points. Default is 50.')

    parser.add_argument('--step-size', type=int, default=20,
                        help='Step size, default is 20.')

    parser.add_argument('--approximation', type = int, default = 100,
	                    help = 'Number of Chebyshev approximation. Default is 100.')

    parser.add_argument('--tsne', type=int, default=10,
                        help='TSNE perplexity, default is 10.')

    parser.add_argument('--random', type=int, default=42,
                        help='Random seed, default is 42.')

    return parser.parse_args()


def HSDParameterParser():
    """
    HSD命令行参数
    """
    parser = argparse.ArgumentParser(prog="HSD", description="Hierarchically Structural Distance")
    parser.add_argument("--graph", type=str,
                        help='Network graph name: barbell, mkarate, europe, usa')

    parser.add_argument("--scale", type=float, default=2.0,
                        help='Heat-coefficent, i.e, the scale, default is 2.')

    parser.add_argument("--metric", type=str, default="wasserstein",
                        help='Distance metric: wasserstein, hellinger, default is wasserstein.')

    parser.add_argument("--hop", type=int, default=5,
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

    parser.add_argument('--multi_scales', type=str, default='no',
                        help="Employ multi scales analysis, default is No.")

    parser.add_argument('--embedding_method', type=str, default='no',
                        help="Embedding methods, LE - Laplacian Eigenmaps, LLE - Locally Linear Embedding")

    parser.add_argument('--dim', type=int, default=64,
                        help="Embedding vector dimension, default is 64.")

    parser.add_argument('--sparse', type=float, default=0.9,
                        help="Remove how many redundant edges from new graph, default is 0.9.")

    parser.add_argument('--reuse', type=str, default="yes",
                        help="Reuse the distance results computed before.")


    return parser.parse_args()


def Struc2vecParameterParser():
    """
    struc2vec
    """
    parser = argparse.ArgumentParser(prog="Struc2vec", description="")
    parser.add_argument('--graph', type=str, default='barbell',
                        help='Network graph name: barbell, mkarate, europe, usa')

    parser.add_argument('--walk_length', type=int, default=15,
                        help='Random walk length, default is 10')

    parser.add_argument('--walk_num', type=int, default=10,
                        help='Number of random walks, default is 10.')

    parser.add_argument('--stay_prob', type=float, default=0.3,
                        help="Stay probability")

    parser.add_argument('--dim', type=int, default=64,
                        help="Dimension of embedding vectors.")

    parser.add_argument('--window_size', type=int, default=10,
                        help="Skip-gram model, window size for training.")

    parser.add_argument('--tsne', type=int, default=30,
                        help="Perplexity of TSNE model, default is 30.")

    parser.add_argument('--random', type=int, default=42,
                        help="Random seed, default is 42.")

    parser.add_argument('--workers', type=int, default=2,
                        help="Number of process workers.")

    parser.add_argument('--iter', type=int, default=5,
                        help="Number of iteration in word2vec.")

    return parser.parse_args()


def Node2vecParameterParser():
    """
    node2vec
    """
    parser = argparse.ArgumentParser(prog="Node2vec", description="")
    parser.add_argument('--graph', type=str, default='barbell',
                        help='Network graph name: barbell, mkarate, europe, usa')

    parser.add_argument('--walk_length', type=int, default=15,
                        help='Random walk length, default is 10')

    parser.add_argument('--walk_num', type=int, default=10,
                        help='Number of random walks, default is 10.')

    parser.add_argument('--p', type=float, default=1.0,
                        help="Stay probability")

    parser.add_argument('--q', type=float, default=2.0,
                        help="Stay probability")

    parser.add_argument('--dim', type=int, default=64,
                        help="Dimension of embedding vectors.")

    parser.add_argument('--window_size', type=int, default=10,
                        help="Skip-gram model, window size for training.")

    parser.add_argument('--tsne', type=int, default=30,
                        help="Perplexity of TSNE model, default is 30.")

    parser.add_argument('--random', type=int, default=42,
                        help="Random seed, default is 42.")

    parser.add_argument('--workers', type=int, default=2,
                        help="Number of process workers.")

    parser.add_argument('--iter', type=int, default=5,
                        help="Number of iteration in word2vec.")

    return parser.parse_args()


def RolxParameterParser():
    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook tvshow network.
    The default hyperparameters give a good quality representation and good candidate cluster means without grid search.
    """

    parser = argparse.ArgumentParser(description="Run RolX.")

    parser.add_argument("--graph", type=str, default="mkarate",
                        help="Network graph name: barbell, mkarate, europe, usa")

    parser.add_argument("--recursive-features-output",
                        nargs="?",
                        default="./rolX/features/{}_features.csv",
                        help="Embeddings path.")

    parser.add_argument("--embedding-output",
                        nargs="?",
                        default="./rolX/embeddings/{}_embedding.csv",
                        help="Embeddings path.")

    parser.add_argument("--log-output",
                        nargs="?",
                        default="./rolX/logs/{}_log.json",
                        help="Log path.")

    # -----------------------------------------------------------------------
    # Recursive feature extraction parameters.
    # -----------------------------------------------------------------------

    parser.add_argument("--recursive-iterations", type=int, default=5,
                        help="Number of recursions.")

    parser.add_argument("--aggregator", nargs="?", default="simple",
                        help="Aggregator statistics extracted.")

    parser.add_argument("--bins", type=int, default=4,
                        help="Number of quantization bins.")

    parser.add_argument("--pruning-cutoff", type=float, default=0.5,
                        help="Absolute correlation for feature pruning.")

    # ------------------------------------------------------------------
    # Factor model parameters.
    # ------------------------------------------------------------------

    parser.add_argument("--dim", type=int, default=16,
                        help="Number of dimensions. Default is 16.")

    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of edges in batch. Default is 128.")

    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs. Default is 50.")

    parser.add_argument("--initial-learning-rate", type=float, default=0.01,
                        help="Initial learning rate. Default is 0.01.")

    parser.add_argument("--minimal-learning-rate", type=float, default=0.001,
                        help="Minimal learning rate. Default is 0.001.")

    parser.add_argument("--annealing-factor", type=float, default=1,
                        help="Annealing factor. Default is 1.0.")

    # ------------------------------------------------------------------
    # Evaluate parameters.
    # ------------------------------------------------------------------

    parser.add_argument('--tsne', type=int, default=30,
                        help="Perplexity of TSNE model, default is 30.")

    parser.add_argument('--test_size', type=float, default=0.3,
                        help="Test set size, default is 0.3.")

    parser.add_argument("--neighbors", type=int, default=10,
                        help="Number of neighbors in KNN evaluate model.")

    parser.add_argument('--cv', type=int, default=5,
                        help="Cross validation in evaluate process, default is 5.")

    parser.add_argument('--random', type=int, default=42,
                        help="random seed.")

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