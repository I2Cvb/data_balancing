#title           :test_classification.py
#description     :This will create a header for a python script.
#author          :Guillaume Lemaitre
#date            :2015/05/31
#version         :0.1
#notes           :
#python_version  :2.7.6  
#==============================================================================

# Import the needed libraries
# Numpy library
import numpy as np
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

from protoclass.tool.dicom_manip import OpenDataLabel
from protoclass.classification.classification import Classify

# # Get the path to file
filename_data = '../../data/clean/npz/x1data.npz'
print 'Opening the following file: {}'.format(filename_data)

# # Read the data
data, label = OpenDataLabel(filename_data)
# data, label = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
#                            n_informative=3, n_redundant=1, flip_y=0,
#                            n_features=20, n_clusters_per_class=1,
#                            n_samples=5000, random_state=10)

# normalise the data
scaler_min_max = MinMaxScaler(feature_range=(-1., 1.), copy=False)
scaler_min_max.fit_transform(data)

# Create 10-fold for the cross validation
n_folds = 10
n_samples = data.shape[0]
# With unbalanced data we need to make some stratified k-fold
kf = StratifiedKFold(label, n_folds=n_folds)

# Import Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.lda import LDA
# Import the metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

n_jobs = -1
roc_fold = []
auc_fold = []
sens_fold = []
spec_fold = []
for (k, (training_idx, testing_idx)) in enumerate(kf):

    print 'Iteration #{}'.format(k)
    # Extract the data
    ### Training
    training_data = data[training_idx, :]
    training_label = label[training_idx]
    ### Testing 
    testing_data = data[testing_idx, :]
    testing_label = label[testing_idx]

    # Declare the random forest
    #crf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs)
    #crf = AdaBoostClassifier(n_estimators=100)
    #crf = LinearSVC()
    crf = LDA()

    # Train the classifier
    crf.fit(training_data, training_label)

    # Test the classifier
    pred_labels = crf.predict(testing_data)
    pred_probs = crf.predict_proba(testing_data)
    #pred_probs = crf.decision_function(testing_data)

    # Compute the confusion matrix
    cm = confusion_matrix(testing_label, pred_labels)
    # Compute the sensitivity and specificity
    sens = float(cm[1, 1]) / float(cm[1, 1] + cm[1, 0])
    spec = float(cm[0, 0]) / float(cm[0, 0] + cm[0, 1])
    sens_fold.append(sens)
    spec_fold.append(spec)

    # Compute the roc curve
    roc_exp = roc_curve(testing_label, pred_probs[:, 1])
    auc_exp = roc_auc_score(testing_label, pred_probs[:, 1])
    #roc_exp = roc_curve(testing_label, pred_probs)
    #auc_exp = roc_auc_score(testing_label, pred_probs)
    roc_fold.append(roc_exp)
    auc_fold.append(auc_exp)
