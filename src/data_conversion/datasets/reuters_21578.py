""" 

The original dataset and further information can be found here:

    https:

Brief description
-----------------

#TODO https:

Original Owner and Donor
------------------------


References
----------
#TODO
"""

# Authors: Joan Massich and Guillaume Lemaitre
# License: MIT

from os.path import join, exists
from os import makedirs
try:
    # Python 2
    from urllib2 import urlretrieve
except ImportError:
    # Python 3+
    from urllib import urlretrieve

import numpy as np

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/"
TARGET_FILENAME_ = ["reuters21578.tar.gz"]
RAW_DATA_LABEL = 'reuters_21578'

def get_dataset_home(data_home=None, dir=RAW_DATA_LABEL):
    return join(get_data_home(data_home=data_home), dir)

def fetch_reuters_21578(data_home=None, download_if_missing=True):
    """Fetcher for xxxxxxxxxxxxxxxxxxxxx.

    Parameters
    ----------
    data_home : optional, default: None
        Specify another download and cache folder for the datasets. By default
        the original datasets for this `data_balance` study are stored at
        `../data/raw/` subfolders.

    download_if_missing: optional, True by default
        If False, raise a IOError if the data is not locally available
        instead of trying to download the data from the source site.

    """
    data_home = get_dataset_home(data_home=data_home)
    if not exists(data_home):
        makedirs(data_home)
    for target in TARGET_FILENAME_:
        url = join(DATA_URL, target)
        path = join(data_home, target)
        print('downloading %s data from %s to %s' %
              (RAW_DATA_LABEL, url, data_home))
        urlretrieve(url, path)

def process_reuters_21578(target=None):
    """Process data of the solar flare dataset.

    Parameters
    ----------
    target: the target class #TODO

    Returns
    -------
    (data, label)

    #TODO: check if files exist
    #TODO: a generic file managing using get_data_home
    #TODO:
    """
    def parse(src):
        f = join(get_dataset_home() , src)
        return np.loadtxt(f, delimiter=',', dtype=str, skiprows=1)

    #TODO: assert target
    # tmp_input = parse(TARGET_FILENAME_[0])

    # label = np.array([1 if x==0 else 0 for tmp_input[:,-2]], dtype=int)
    # return (tmp_input[:, :-4], tmp_input[:, -1])
    return ([0], [0])

def convert_reuters_21578():
    d, l = process_reuters_21578()
    np.savez('../data/clean/uci-reuters_21578.npz', data=d, label=l)

if __name__ == '__main__':
    fetch_reuters_21578()
    convert_reuters_21578()
