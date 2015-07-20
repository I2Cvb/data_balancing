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

def balance():
    # balance dataset

    filename = '../../data/raw/mldata/balance-scale.data'

    # We are loading only the 7th continous attributes
    data = np.loadtxt(filename, delimiter= ',', usecols = tuple(range(1, 5)), dtype=float)
    
    # Get the label
    tmp_label = np.loadtxt(filename, delimiter = ',', usecols = (0, ), dtype=str)
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 'B')] = 1
        
    np.savez('../../data/clean/uci-balance.npz', data=data, label=label)

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
    label[np.nonzero(tmp_label == 'vgood')] = 1
    
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
    label[np.nonzero(tmp_label == 'vgood')] = 1
    
    np.savez('../../data/clean/uci-car-eval-4.npz', data=data, label=label)

def isolet():
    # isolet dataset

    filename = '../../data/raw/mldata/isolet.data'
    
    data = np.loadtxt(filename, delimiter = ',', usecols = tuple(range(617)), dtype=float)
    tmp_label = np.loadtxt(filename, delimiter = ',', usecols = (617, ), dtype=float)

    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 1.)] = 1
    label[np.nonzero(tmp_label == 2.)] = 1
    
    np.savez('../../data/clean/uci-isolet.npz', data=data, label=label)

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

def yeast_ml8():
    # yeast dataset

    filename = '../../data/raw/mldata/yeast.svm'

    tmp_data, tmp_label = load_svmlight_file(filename, multilabel=True)
    data = tmp_data.toarray()
    label = np.zeros(len(tmp_label), dtype=int)

    # Get only the value with label 8.
    for idx in range(len(tmp_label)):
        if 8.0 in tmp_label[idx]:
            label[idx] = 1

    np.savez('../../data/clean/libsvm-yeast-ml8.npz', data=data, label=label)

def yeast_me2():
    # yeast dataset

    filename = '../../data/raw/mldata/yeast.data'
    
    # read the first column and convert it into float
    tmp_data = np.atleast_2d(np.loadtxt(filename, usecols=(0,), dtype=str)).T
    # encode this feature to get a float
    le = LabelEncoder()
    tmp_data = np.ravel(le.fit_transform(tmp_data[:, 0]).astype(float))
    data = np.hstack((np.atleast_2d(tmp_data).T, np.loadtxt(filename, usecols=tuple(range(1, 9)), dtype=float)))

    # read the label as string
    tmp_label = np.ravel(np.loadtxt(filename, usecols=(9, ), dtype=str))
    label = np.zeros(tmp_label.shape, dtype=int)
    # Get only the label equal to ME2
    label[np.nonzero(tmp_label == 'ME2')] = 1

    np.savez('../../data/clean/uci-yeast-me2.npz', data=data, label=label)

def scene():
    # scene dataset

    filename = '../../data/raw/mldata/scene.svm'

    tmp_data, tmp_label = load_svmlight_file(filename, multilabel=True)
    data = tmp_data.toarray()
    label = np.zeros(len(tmp_label), dtype=int)

    # Get only the value with label 8.
    for idx in range(len(tmp_label)):
        if len(tmp_label[idx]) > 1:
            label[idx] = 1

    np.savez('../../data/clean/libsvm-scene.npz', data=data, label=label)

def movement_libras():
    # movement libras dataset

    filename = '../../data/raw/mldata/movement_libras.data'
    
    data = np.loadtxt(filename, delimiter = ',', usecols = tuple(range(90)), dtype=float)
    tmp_label = np.loadtxt(filename, delimiter = ',', usecols = (90, ), dtype=float)

    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 1.)] = 1
    
    np.savez('../../data/clean/uci-movement-libras.npz', data=data, label=label)

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

def glass():
    # glass dataset

    filename = '../../data/raw/mldata/glass.data'

    data = np.loadtxt(filename, delimiter=',', usecols = tuple(range(1, 10)), dtype=float)
    
    # Get the label
    tmp_label = np.loadtxt(filename, delimiter=',', usecols = (10, ), dtype=int)
    count_l = Counter(tmp_label)
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == min(count_l, key=count_l.get))] = 1
    
    np.savez('../../data/clean/uci-glass.npz', data=data, label=label)

def ionosphere():
    # ionosphere dataset

    filename = '../../data/raw/mldata/ionosphere.data'

    data = np.loadtxt(filename, delimiter=',', usecols = tuple(range(0, 34)), dtype=float)
    
    # Get the label
    tmp_label = np.loadtxt(filename, delimiter=',', usecols = (34, ), dtype=str)
    label = np.zeros(tmp_label.shape, dtype=int)
    label[np.nonzero(tmp_label == 'b')] = 1
    
    np.savez('../../data/clean/uci-ionosphere.npz', data=data, label=label)


def phoneme():
    # phoneme dataset

    filename = '../../data/raw/mldata/phoneme.dat'

    data = np.loadtxt(filename, usecols = tuple(range(0, 5)), dtype=float)
    
    # Get the label
    label = np.loadtxt(filename, usecols = (5, ), dtype=int)
        
    np.savez('../../data/clean/elena-phoneme.npz', data=data, label=label)

def arrhythmia():
    # arrhythmia dataset

    filename = '../../data/raw/mldata/arrhythmia.data'
    
    # The missing data will be consider as NaN
    tmp_data = np.genfromtxt(filename, delimiter = ',')

    # replace missing value by the mean
    imp = Imputer(verbose = 1)
    tmp_data = imp.fit_transform(tmp_data)

    # extract the data to be saved
    data = tmp_data[:, :-1]
    label = np.zeros(tmp_data.shape[0], dtype=int)
    label[np.nonzero(tmp_data[:, -1] == 6)] = 1

    np.savez('../../data/clean/uci-arrhythmia.npz', data=data, label=label)

def solar_flare():
    # solar dataset
    
    filename = '../../data/raw/mldata/flare.data'
    
    tmp_data = np.loadtxt(filename, delimiter = ' ', dtype=str)
    tmp_label = tmp_data[:, -2].astype(int)
    # Only the first ten columns are attributes
    tmp_data_2 = np.zeros((tmp_data.shape[0], 10), dtype=int)

    # Encode each label with an integer
    for f_idx in range(tmp_data_2.shape[1]):
        le = LabelEncoder()
        tmp_data_2[:, f_idx] = le.fit_transform(tmp_data[:, f_idx])

    # initialise the data
    data = np.zeros(tmp_data_2.shape, dtype=float)
    label = np.zeros(tmp_label.shape, dtype=int)
            
    # Push the data
    data = tmp_data_2[:, :]
    label[np.nonzero(tmp_label > 0)] = 1
    
    np.savez('../../data/clean/uci-solar-flare-m0.npz', data=data, label=label)

def wine_quality_white():
    # white wine quality dataset

    filename = '../../data/raw/mldata/winequality-white.csv'

    # The data corresponds to the 11 first column of the csv file
    data = np.loadtxt(filename, usecols=tuple(range(11)), delimiter=';', dtype=float)
    # Read the label
    # We need to binarise the label using a threshold at 4
    bn = Binarizer(threshold=4)
    label = bn.fit_transform(np.loadtxt(filename, usecols=(11,), delimiter=';', dtype=int))
    # We need to inverse the label -> 1=0 and 0=1
    label = np.ravel(np.abs(label - 1))
    
    np.savez('../../data/clean/uci-wine-quality-white.npz', data=data, label=label)

def letter_recognition():
    # letter recognition dataset

    filename = '../../data/raw/mldata/letter-recognition.data'

    # The data to read are from the second to the last 
    data = np.loadtxt(filename, usecols=tuple(range(1, 17)), delimiter=',', dtype=float)

    # The label is the first column of the data file and need to be converted to integer
    tmp_label = np.ravel(np.loadtxt(filename, usecols=(0,), delimiter=',', dtype=str))
    label = np.zeros(tmp_label.shape, dtype=int)
    # Find only the Z letter
    label[np.nonzero(tmp_label == 'Z')] = 1

    np.savez('../../data/clean/uci-letter-recognition-z.npz', data=data, label=label)

def convert(convert_func, out_file_name, force):
    path = '../data/clean/' + out_file_name
    if force or not exist(path):
        convert_func(path)

if __name__ == "__main__":
    
    abalone_19()
    adult()
    ecoli()
    optical_digits()    
    sat_image()
    pen_digits()
    abalone_7()
    spectrometer()
    balance()
    car_eval_34()
    car_eval_4()
    isolet()
    us_crime()
    yeast_ml8()
    yeast_me2()
    scene()
    movement_libras()
    sick()
    glass()
    ionosphere()
    phoneme()
    # force_convertion=False
    # convert(convert_coil2000, 'coil_2000.npz', force_convertion)
    arrhythmia()
    solar_flare()
    wine_quality_white()
    letter_recognition()
    
