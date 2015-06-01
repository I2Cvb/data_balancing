# In order to manipulate the array
import numpy as np

# In order to load mat file
from scipy.io import loadmat

# In order to import the libsvm format dataset
from sklearn.datasets import load_svmlight_file

def abalone_19():
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

    np.savez('../../data/clean/uci-abalone-19.npz', data=data, label=label)

def abalone_7():
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
    label[np.nonzero(np.ravel(matfile['int1']) == 7)] = 1

    np.savez('../../data/clean/uci-abalone-7.npz', data=data, label=label)


def adult():
    # Adult dataset

    filename = '../../data/raw/mldata/adult'
    
    tmp_input = np.loadtxt(filename, delimiter = ',', usecols = (0, 2, 4, 10, 11, 12, 14))

    data = tmp_input[:, :-1]
    label = tmp_input[:, -1].astype(int)

    np.savez('../../data/clean/uci-adult.npz', data=data, label=label)

def ecoli():
    # ecoli dataset

    filename = '../../data/raw/mldata/ecoli.data'

    # We are loading only the 7th continous attributes
    data = np.loadtxt(filename, usecols = (1, 2, 3, 4, 5, 6, 7), dtype=float)
    
    # Get the label
    tmp_label = np.loadtxt(filename, usecols = (8, ), dtype=str)
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 'imU')] = 1
    
    np.savez('../../data/clean/uci-ecoli.npz', data=data, label=label)

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

def sat_image():
    # sat image dataset
 
    filename = '../../data/raw/mldata/satimage.scale'
 
    tmp_data, tmp_label = load_svmlight_file(filename)
    data = tmp_data.toarray()
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 4)] = 1

    np.savez('../../data/clean/uci-sat-image.npz', data=data, label=label)

def pen_digits():
    # sat image dataset
 
    filename = '../../data/raw/mldata/pendigits'
 
    tmp_data, tmp_label = load_svmlight_file(filename)
    data = tmp_data.toarray()
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 5)] = 1

    np.savez('../../data/clean/uci-pen-digits.npz', data=data, label=label)

def spectrometer():
    # spectrometer dataset

    filename = '../../data/raw/mldata/lrs.data'

    # We are loading only the 7th continous attributes
    data = np.loadtxt(filename, usecols = tuple(range(10, 103)), dtype=float)
    
    # Get the label
    tmp_label = np.loadtxt(filename, usecols = (1, ), dtype=int)
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 44)] = 1
    
    np.savez('../../data/clean/uci-spectrometer.npz', data=data, label=label)

if __name__ == "__main__":
    
    #abalone_19()
    #adult()
    #ecoli()
    #optical_digits()    
    #sat_image()
    #pen_digits()
    #abalone_7()
    spectrometer()
