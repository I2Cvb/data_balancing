#title           :classiciation_imbalanced_study.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre, Mojdeh Rastgoo
#date            :2016/01/19
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
import pandas as pd
import h5py
# Joblib library
### Module to performed parallel processing
from joblib import Parallel, delayed
### Module to performed parallel processing
import multiprocessing
# OS library
import os
from os.path import join, isdir, isfile
# sys library
import sys

# Scikit-learn library
from sklearn.datasets import make_classification
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

from protoclass.classification.classification import Classify

# Initialization to the data paths 
dataPath = sys.argv[1]
path_to_save = sys.argv[2]

#fread = pd.read_csv(dataPath.__add__('feature.csv'))
fread = pd.read_csv(join(dataPath, 'feature.csv'))
FeatureLists = fread.values
FeatureLists = FeatureLists[:,0]

#f= h5py.File(dataPath.__add__('PH2_Train_Test_80_20.mat'), 'r')
f = h5py.File(join(dataPath, 'PH2_Train_Test_80_20.mat'), 'r')
#CVIdx = sio.loadmat(datapath.__add__('TrainTestIndex_117_39_80.mat'))
trainIdx = np.asmatrix(f.get('trainingIdx')) 
trainIdx = trainIdx.T
trainIdx = trainIdx - 1.
testIdx = np.asmatrix(f.get('testingIdx'))
testIdx = testIdx.T
testIdx = testIdx - 1.
Labels= np.asmatrix(f.get('BinaryLabels'))
Labels = Labels.T
ntree = 100; 

config = [{'classifier_str' : 'naive-bayes', 'class_prior_override' : np.array([.2, .8])},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
	    'balancing_criterion' : 'random-over-sampling'},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'smote', 'kind_smote' : 'regular'},
           #{'classifier_str' : 'naive-bayes', 
            #'balancing_criterion' : 'smote', 'kind_smote' : 'borderline1'},
           #{'classifier_str' : 'naive-bayes', 
            #'balancing_criterion' : 'smote', 'kind_smote' : 'borderline2'},
           #{'classifier_str' : 'naive-bayes', 
            #'balancing_criterion' : 'smote', 'kind_smote' : 'svm'},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'random-under-sampling', 'replacement' : True},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'tomek_links'},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'clustering'},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'nearmiss', 'version_nearmiss' : 1, 'size_ngh': 3},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'nearmiss', 'version_nearmiss' : 2, 'size_ngh': 3},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'nearmiss', 'version_nearmiss' : 3, 'size_ngh': 3, 'ver3_samp_ngh' : 3},
           #{'classifier_str' : 'naive-bayes', 
            #'balancing_criterion' : 'cnn', 'size_ngh' : 3, 'n_seeds_S' :1},
           #{'classifier_str' : 'naive-bayes', 
            #'balancing_criterion' : 'one-sided-selection', 'size_ngh' : 1, 'n_seeds_S' :1},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'ncr', 'size_ngh' : 3},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'easy-ensemble', 'n_subsets' :  10},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'balance-cascade', 'n_max_subset' : 100, 
            'balancing_classifier' : 'knn', 'bootstrap' : True},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'smote-enn', 'size_ngh' : 3},
           {'classifier_str' : 'naive-bayes', 'class_prior_override' : None,
            'balancing_criterion' : 'smote-tomek'}]

FeaturesIdx = np.array([[1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1],\
[1,1,0,0,0,0], [1,0,1,0,0,0], [1,0,0,1,0,0], [0,1,1,0,0,0], [0,1,0,1,0,0], [0,0,1,1,0,0],\
[0,0,0,0,1,1], [1,1,1,1,0,0], [1,0,0,0,1,1], [0,1,0,0,1,1], [0,0,1,0,1,1], [0,0,0,1,1,1],\
[1,1,0,0,1,1], [1,0,1,0,1,1], [1,0,0,1,1,1], [0,1,1,0,1,1], [0,1,0,1,1,1], [0,0,1,1,1,1]])
#[0,1,0,0],[0,0,1,0],[0,0,0,1] , [0,0,1,1], [1,1,0,0],[1,0,1,1],[0,1,1,1],[1,1,1,1]])


for I in range (0, FeaturesIdx.shape[0]):
    NonzeroIdx = np.ravel(np.nonzero(FeaturesIdx[I]))
    FVcombined = np.empty(shape = [193, 0])
    

    for PIdx in range (0, NonzeroIdx.shape[0]):
        f= h5py.File(join(dataPath,FeatureLists[NonzeroIdx[PIdx]]), 'r')
        #f = sio.loadmat(join(featurePath, FeatureLists[NonzeroIdx[PIdx]]))
        FV =np.asmatrix(f.get('FV'))
 	FV =FV.T   
        FVcombined = np.append(FVcombined, FV, axis = 1)
        del FV
        
        FV = FVcombined

    rocs = []
    gt_labels = []
    pred_labels = []
    pred_probs = []
    # Apply the classification for each fold

    for CV in range (0, trainIdx.shape[1]):
    	print 'Iteration #{}'.format(CV)
    	# Extract the data
    	### Training
        train_data = FV[np.ravel(trainIdx[:,CV].astype(int)), :]
        train_label = np.ravel(Labels[np.ravel(trainIdx[:,CV].astype(int))])
	### Testing
        test_data = FV[np.ravel(testIdx[:,CV].astype(int)), :]
        test_label = np.ravel(Labels[np.ravel(testIdx[:,CV].astype(int))])
    
    	config_roc = []
    	config_pred_label = []
        config_pred_prob = []
    	config_gt_label = []
    	for c in config:
            print c
            pred_label, pred_prob, roc = Classify(train_data, train_label, test_data, test_label, **c)
            config_roc.append(roc)
            config_pred_label.append(pred_label)
            config_pred_prob.append(pred_prob)
            config_gt_label.append(test_label)

    	rocs.append(config_roc)
    	pred_labels.append(config_pred_label)
        pred_probs.append(config_pred_prob)
    	gt_labels.append(config_gt_label)

    # Convert the data to store to numpy data
    rocs = np.array(rocs)
    pred_labels = np.array(pred_labels)
    pred_probs = np.array(pred_probs)
    gt_labels = np.array(gt_labels)

    # Reshape the array to have the first index corresponding to the
    # configuration, the second index to the iteration of the k-fold
    # and the last index to the data themselve.
    rocs = np.swapaxes(rocs, 0, 1)
    pred_labels = np.swapaxes(pred_labels, 0, 1)
    pred_probs = np.swapaxes(pred_probs, 0, 1)
    gt_labels = np.swapaxes(gt_labels, 0, 1)

    # Save the results somewhere
    if not os.path.exists(path_to_save):
    	os.makedirs(path_to_save)

    from os.path import basename
    saving_filename = 'melanoma_imbalanced_80_20_' + str(ntree) +  '_' + str(I)
    saving_path = join(path_to_save, saving_filename)
    np.savez(saving_path, gt_labels=gt_labels, pred_labels=pred_labels, pred_probs=pred_probs, rocs=rocs)
    tosave={}
    tosave['rocs'] = rocs
    tosave['pred_labels'] = pred_labels
    tosave['pred_probs'] = pred_probs
    tosave['gt_labels'] = gt_labels
    saving_path = join(path_to_save, saving_filename)
    from scipy.io import savemat
    savemat(saving_path, tosave)
