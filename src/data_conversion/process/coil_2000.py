# Authors: Joan Massich and Guillaume Lemaitre
# License: MIT

from os.path import join, exists

import numpy as np


_TARGET_FILENAME = ["ticdata2000.txt", "ticeval2000.txt", "tictgts2000.txt"]

# Grab the module-level docstring to use as a description of the
# dataset
MODULE_DOCS = __doc__


def process_coil_2000():
    """Process data of the CoIL 2000 dataset.

    it generates a npz according to... #TODO

    #TODO: check if files exist
    #TODO: a generic file managing using get_data_home
    #TODO:
    """

    parse = lambda src: np.loadtxt(join('../data/raw/coil_2000/', src), dtype=int)
    ticdata  = parse("ticdata2000.txt")
    ticeval  = parse("ticeval2000.txt")
    tictgts  = parse("tictgts2000.txt")

    return (np.append(ticdata[:,:-1], ticeval, axis=0),
            np.append(ticdata[:,-1], tictgts, axis=0))

def convert_coil_2000():
    d, l = process_coil_2000()
    np.savez('../data/clean/coil_2000.npz', data=d, label=l)

if __name__ == '__main__':
    convert_coil_2000()
