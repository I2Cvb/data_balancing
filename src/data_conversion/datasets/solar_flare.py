""" Solar Flare Database

The original dataset and further information can be found here:

    https://archive.ics.uci.edu/ml/datasets/Solar+Flare

Brief description
-----------------

This data contains a single observation on 10 variables for 1,389 active
regions of the Sun. The data was collected to predict solar flare in 24h
period. The original dataset was composed of two subsets with different
level of error correction on the data, however this distinction is not
mantained here.

Attribute Information:
    1. Code for class (modified Zurich class)  (A,B,C,D,E,F,H)
    2. Code for largest spot size              (X,R,S,A,H,K)
    3. Code for spot distribution              (X,O,I,C)
    4. Activity                                (1 = reduced, 2 = unchanged)
    5. Evolution                               (1 = decay, 2 = no growth,
                                                3 = growth)
    6. Previous 24 hour flare activity code    (1 = nothing as big as an M1,
                                                2 = one M1,
                                                3 = more activity than one M1)
    7. Historically-complex                    (1 = Yes, 2 = No)
    8. Did region become historically complex  (1 = yes, 2 = no)
       on this pass across the sun's disk
    9. Area                                    (1 = small, 2 = large)
    10. Area of the largest spot                (1 = <=5, 2 = >5)

From all these predictors three classes of flares are predicted, which are
represented in the last three columns.

11. C-class flares production by this region    Number
in the following 24 hours (common flares)
12. M-class flares production by this region    Number
in the following 24 hours (moderate flares)
13. X-class flares production by this region    Number
in the following 24 hours (severe flares)

8. Missing values: None

9. Class Distribution:

                  0         1     2    3   4  4  5  6  7  8  Total
C-class flares 287+884   129+12  7+33  20  0  9  4  3  0  1   1389
M-class flares 291+1030   24+29  6+ 3  2   2  1  0  1  0  0   1389
X-class flares 316+1061    7+ 4  0+ 1  0   0  0  0  0  0  0   1389

Original Owner and Donor
------------------------
Gary Bradshaw

Email: gbradshaw@clipr.colorado.edu


References
----------

#TODO: explain that we use class=M>0
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

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/solar-flare/"
TARGET_FILENAME_ = ["flare.data1", "flare.data2"]
RAW_DATA_LABEL = 'solar_flare'

def get_dataset_home(data_home=None, dir=RAW_DATA_LABEL):
    return join(get_data_home(data_home=data_home), dir)

def fetch_solar_flare(data_home=None, download_if_missing=True):
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

def process_solar_flare(target=None):
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
        return np.loadtxt(f, delimiter=' ', dtype=str, skiprows=1)

    #TODO: assert target
    tmp_input = np.append(parse(TARGET_FILENAME_[0]),
                          parse(TARGET_FILENAME_[1]),
                          axis=0)
    label = np.array([1 if x==0 else 0 for tmp_input[:,-2]], dtype=int)
    return (tmp_input[:, :-4], tmp_input[:, -1])

def convert_solar_flare_Mgreatthan0():
    d, l = process_solar_flare(target='M>0')
    np.savez('../data/clean/uci-solar_flare_Mgth0.npz', data=d, label=l)

if __name__ == '__main__':
    fetch_solar_flare()
    convert_solar_flare_Mgreatthan0()
