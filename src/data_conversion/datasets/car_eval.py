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

def car_eval_34():
    # car eval dataset

    filename = '../../data/raw/mldata/car.data'
    
    tmp_data = np.loadtxt(filename, delimiter = ',', dtype=str)
    tmp_label = tmp_data[:, -1]
    tmp_data_2 = np.zeros((tmp_data.shape), dtype=int)

    # Encode each label with an integer
    for f_idx in range(tmp_data.shape[1]):
        le = LabelEncoder()
        tmp_data_2[:, f_idx] = le.fit_transform(tmp_data[:, f_idx])

    # initialise the data
    data = np.zeros((tmp_data.shape[0], tmp_data.shape[1] - 1), dtype=float)
    label = np.zeros((tmp_data.shape[0], ), dtype=int)
    
    # Push the data
    data = tmp_data_2[:, :-1]
    label[np.nonzero(tmp_label == 'good')] = 1
    label[np.nonzero(tmp_label == 'v-good')] = 1
    
    np.savez('../../data/clean/uci-car-eval-34.npz', data=data, label=label)

def car_eval_4():
    # car eval dataset

    filename = '../../data/raw/mldata/car.data'
    
    tmp_data = np.loadtxt(filename, delimiter = ',', dtype=str)
    tmp_label = tmp_data[:, -1]
    tmp_data_2 = np.zeros((tmp_data.shape), dtype=int)

    # Encode each label with an integer
    for f_idx in range(tmp_data.shape[1]):
        le = LabelEncoder()
        tmp_data_2[:, f_idx] = le.fit_transform(tmp_data[:, f_idx])

    # initialise the data
    data = np.zeros((tmp_data.shape[0], tmp_data.shape[1] - 1), dtype=float)
    label = np.zeros((tmp_data.shape[0], ), dtype=int)
    
    # Push the data
    data = tmp_data_2[:, :-1]
    label[np.nonzero(tmp_label == 'v-good')] = 1
    
    np.savez('../../data/clean/uci-car-eval-4.npz', data=data, label=label)

