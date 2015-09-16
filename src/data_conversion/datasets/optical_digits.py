# In order to manipulate the array
import numpy as np

# In order to load mat file
from scipy.io import loadmat

# In order to import the libsvm format dataset
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Binarizer

from collections import Counter

from fetch.coil_2000 import fetch_coil_2000
from process.coil_2000 import convert_coil_2000


def optical_digits():
    # optical digits dataset

    filename = '../../data/raw/mldata/optdigits'

    # We are loading only the 7th continous attributes
    data = np.loadtxt(filename, delimiter = ',', usecols = tuple(range(64)), dtype=float)

    # Get the label
    tmp_label = np.loadtxt(filename, delimiter = ',', usecols = (64, ), dtype=int)
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 8)] = 1

    np.savez('../../data/clean/uci-optical-digits.npz', data=data, label=label)

