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

def yeast():
    # yeast dataset

    filename = '../../data/raw/mldata/yeast.svm'

    tmp_data, tmp_label = load_svmlight_file(filename, multilabel=True)
    data = tmp_data.toarray()
    label = np.zeros(len(tmp_label), dtype=int)

    # Get only the value with label 8.
    for idx in range(len(tmp_label)):
        if 8.0 in tmp_label[idx]:
            label[idx] = 1

    np.savez('../../data/clean/libsvm-yeast.npz', data=data, label=label)


