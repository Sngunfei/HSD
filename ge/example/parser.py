import argparse
from texttable import Texttable

def parameter_parser():
    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description = "Run WaveletMachine.")

    parser.add_argument('--mechanism',
                        nargs = '?',
                        default = 'exact',
	                help = 'Eigenvalue calculation method. Default is exact.')

    parser.add_argument('--input',
                        nargs = '?',
                        default = '../data/bell/bell.edgelist',
	                help = 'Path to the graph edges. Default is food_edges.csv.')

    parser.add_argument('--output',
                        nargs = '?',
                        default = '../bell/bell_output.csv',
	                help = 'Path to the structural embedding. Default is embedding.csv.')

    parser.add_argument('--heat-coefficient',
                        type = float,
                        default = 5,
	                help = 'Heat kernel exponent. Default is 1000.0.')

    parser.add_argument('--moment',
                        type=bool,
                        dest="moment",
                        default=False,
                        help='use wavelet moment to embed')

    parser.add_argument('--sample-number',
                        type = int,
                        default = 50,
	                help = 'Number of characteristic function sample points. Default is 50.')

    parser.add_argument('--approximation',
                        type = int,
                        default = 100,
	                help = 'Number of Chebyshev approximation. Default is 100.')

    parser.add_argument('--step-size',
                        type = int,
                        default = 10,
	                help = 'Number of steps. Default is 20.')

    parser.add_argument('--switch',
                        type = int,
                        default = 128,
	                help = 'Number of dimensions. Default is 100.')

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