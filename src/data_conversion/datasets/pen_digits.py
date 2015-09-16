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


def pen_digits():
    # sat image dataset

    filename = '../../data/raw/mldata/pendigits'

    tmp_data, tmp_label = load_svmlight_file(filename)
    data = tmp_data.toarray()
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 5)] = 1

    np.savez('../../data/clean/uci-pen-digits.npz', data=data, label=label)

