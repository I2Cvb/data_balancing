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

def adult():
    # Adult dataset

    filename = '../../data/raw/mldata/adult'

    tmp_input = np.loadtxt(filename, delimiter = ',', usecols = (0, 2, 4, 10, 11, 12, 14))

    data = tmp_input[:, :-1]
    label = tmp_input[:, -1].astype(int)

    np.savez('../../data/clean/uci-adult.npz', data=data, label=label)
