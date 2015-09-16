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

def us_crime():
    # US crime dataset

    filename = '../../data/raw/mldata/communities.data'

    # The missing data will be consider as NaN
    # Only use 122 continuous features
    tmp_data = np.genfromtxt(filename, delimiter = ',')
    tmp_data = tmp_data[:, 5:]

    # replace missing value by the mean
    imp = Imputer(verbose = 1)
    tmp_data = imp.fit_transform(tmp_data)

    # extract the data to be saved
    data = tmp_data[:, :-1]
    bn = Binarizer(threshold=0.65)
    label = np.ravel(bn.fit_transform(tmp_data[:, -1]))

    np.savez('../../data/clean/uci-us-crime.npz', data=data, label=label)


