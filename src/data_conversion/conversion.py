# In order to manipulate the array
import numpy as np

# In order to load mat file
from scipy.io import loadmat

# Abalone dataset - Convert the ring = 19 to class 1 and the other to class 0

filename = '../../data/raw/mldata/uci-20070111-abalone.mat'
matfile = loadmat(filename)

sex_array = np.zeros(np.ravel(matfile['int1']).shape[0])
sex_array[np.nonzero(np.ravel(matfile['Sex']) == 'M')] = 0
sex_array[np.nonzero(np.ravel(matfile['Sex']) == 'F')] = 1
sex_array[np.nonzero(np.ravel(matfile['Sex']) == 'I')] = 2

data = np.zeros((np.ravel(matfile['int1']).shape[0], 8))
data[:, 0] = sex_array
data[:, 1::] = matfile['double0'].T

label = np.zeros((np.ravel(matfile['int1']).shape[0], ), dtype=(int))
label[np.nonzero(np.ravel(matfile['int1']) == 19)] = 1

np.savez('../../data/raw/mldata/uci-20070111-abalone.npz', data=data, label=label)
