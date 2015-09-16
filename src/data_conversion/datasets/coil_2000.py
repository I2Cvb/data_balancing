"""CoIL Challenge 2000 dataset.

The original dataset and further information can be found here:

    http://www.liacs.nl/~putten/library/cc2000/

Brief description
-----------------

The data contains 9,822 observations on 86 variables describing information on
customers of an insurance company. The data was collected to answer the
following real question:

    - Can you predict who would be interested in buying a caravan insurance
    policy and give an explanation why?

This dataset contains `Nr NAME Description Domain`
#TODO: add variables without corrupting the comment

Original Owner and Donor
------------------------

Peter van der Putten
Sentient Machine Research
Baarsjesweg 224
1058 AA Amsterdam
The Netherlands
+31 20 6186927
pvdputten@hotmail.com, putten@liacs.nl


References
----------
P. van der Putten and M. van Someren (eds), CoIL Challenge 2000: The Insurance
Company Case. Published by Sentient Machine Research, Amsterdam.
Also a Leiden Institute of Advanced Computer Science Technical Report 2000-09. June 22, 2000.

"""

# Authors: Joan Massich and Guillaume Lemaitre
# License: MIT

import numpy as np
from ..base import *

# Grab the module-level docstring to use as a description of the
# dataset
MODULE_DOCS = __doc__

DATA_URL = "http://kdd.ics.uci.edu/databases/tic/"
TARGET_FILENAME_ = ["ticdata2000.txt", "ticeval2000.txt", "tictgts2000.txt"]
RAW_DATA_LABEL = 'coil_2000'


def fetch_coil_2000(download_if_missing=True):
    """Fetcher for the CoIL 2000 dataset.

    Parameters
    ----------
    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    """
    dataset_raw_home = get_dataset_home(dir=RAW_DATA_LABEL)
    check_fetch_data(dataset_raw_home=dataset_raw_home,
                     base_url=DATA_URL,
                     target_filenames=TARGET_FILENAME_,
                     dataset_name=RAW_DATA_LABEL,
                     download_if_missing=download_if_missing
                     )


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
            np.append(ticdata[:,-1], tictgts, axis=0).astype(int))

def check_coil_2000_data(data):
    check_data(data)

def check_coil_2000_label(label):
    check_label(label)

def convert_coil_2000():
    fetch_coil_2000(download_if_missing=True)
    d, l = process_coil_2000()
    check_coil_2000_data(d)
    check_coil_2000_label(l)
    #TODO: change this hardcoded file
    np.savez('../data/clean/coil_2000.npz', data=d, label=l)

def load_coil_2000():
    raise NotImplementedError()

if __name__ == '__main__':
    convert_coil_2000()
