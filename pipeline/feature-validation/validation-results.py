#title           :validation-results.py
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
# Scipy library
import scipy as sp
# Matplotlib library
import matplotlib.pyplot as plt
# Sys library
import sys
# OS library
import os
from os.path import join

from collections import namedtuple
roc_auc = namedtuple('roc_auc', ['fpr', 'tpr', 'thresh', 'auc'])

path_to_results = '../../results/data/result_coil_2000.npz'
file_results = np.load(path_to_results)

# build an roc_auc object for each curve
for config in file_results['rocs']:
    # for each cross validation
    for k in config:
        roc = 



rocs = file_results['rocs']
pred_label = file_results['pred_labels']

