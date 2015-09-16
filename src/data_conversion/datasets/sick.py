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

def sick():
    # sick dataset

    filename = '../../data/raw/mldata/sick.data'

    # Read the data for the 26th first dimension
    tmp_data = np.loadtxt(filename, delimiter = ',', usecols = tuple(range(26)), dtype=str)
    tmp_data_3 = np.loadtxt(filename, delimiter = ',', usecols = (29, ), dtype=str)

    tmp_data_2 = []
    tmp_data_4 = []
    for s_idx in range(tmp_data.shape[0]):
        if not np.any(tmp_data[s_idx, :] == '?'):
            tmp_data_2.append(tmp_data[s_idx, :])
            tmp_data_4.append(tmp_data_3[s_idx])

    tmp_data_2 = np.array(tmp_data_2)

    data = np.zeros(tmp_data_2.shape, dtype=float)
    data[:, 0] = tmp_data_2[:, 0].astype(float)
    
    # encode the category
    f_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24]
    for i in f_idx:
        le = LabelEncoder()
        data[:, i] = le.fit_transform(tmp_data_2[:, i]).astype(float)

    f_idx = [17, 19, 21, 23]
    for i in f_idx:
        data[:, i] = tmp_data_2[:, i].astype(float)

    # Create the label
    label = np.zeros((len(tmp_data_4), ), dtype=int)
    for s_idx in range(len(tmp_data_4)):
        if "sick" in tmp_data_4[s_idx]:
            label[s_idx] = 1

    np.savez('../../data/clean/uci-sick.npz', data=data, label=label)

def sick():
    # sick dataset

    filename = '../../data/raw/mldata/sick.data'

    # Read the data for the 26th first dimension
    tmp_data = np.loadtxt(filename, delimiter = ',', usecols = tuple(range(26)), dtype=str)
    tmp_data_3 = np.loadtxt(filename, delimiter = ',', usecols = (29, ), dtype=str)

    tmp_data_2 = []
    tmp_data_4 = []
    for s_idx in range(tmp_data.shape[0]):
        if not np.any(tmp_data[s_idx, :] == '?'):
            tmp_data_2.append(tmp_data[s_idx, :])
            tmp_data_4.append(tmp_data_3[s_idx])

    tmp_data_2 = np.array(tmp_data_2)

    data = np.zeros(tmp_data_2.shape, dtype=float)
    data[:, 0] = tmp_data_2[:, 0].astype(float)
    
    # encode the category
    f_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24]
    for i in f_idx:
        le = LabelEncoder()
        data[:, i] = le.fit_transform(tmp_data_2[:, i]).astype(float)

    f_idx = [17, 19, 21, 23]
    for i in f_idx:
        data[:, i] = tmp_data_2[:, i].astype(float)

    # Create the label
    label = np.zeros((len(tmp_data_4), ), dtype=int)
    for s_idx in range(len(tmp_data_4)):
        if "sick" in tmp_data_4[s_idx]:
            label[s_idx] = 1

    np.savez('../../data/clean/uci-sick.npz', data=data, label=label)


