#title           :validation-results-roc.py
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
from scipy import polyfit, poly1d
from scipy.interpolate import interp1d
# Matplotlib library
import matplotlib.pyplot as plt
# Sys library
import sys
# OS library
import os
from os.path import join

# Import our metrics
from protoclass.validation.metric import LabelsToSensitivitySpecificity, LabelsToPrecisionNegativePredictiveValue, LabelsToGeometricMean, LabelsToAccuracy, LabelsToF1score, LabelsToMatthewCorrCoef, LabelsToGeneralizedIndexBalancedAccuracy
from sklearn.metrics import precision_recall_curve


from collections import namedtuple
roc_auc = namedtuple('roc_auc', ['fpr', 'tpr', 'thresh', 'auc'])


############################## COLORS ##############################
# These are the "Tableau 20" colors as RGB.    
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    
  
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.    
for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)

############################## CREATE ARRAY WITH TYPE OF BALANCING ##############################

bal_arr = ['No balancing',
           'ROS',
           'SMOTE-reg',
           'RUS',
           'TL',
           'Clus',
           'NM1',
           'NM2',
           'NM3',
           'NCR',
           'EasyEns',
           'BalanceCas',
           'SMOTE+ENN',
           'SMOTE+TL']
    

############################## LOADING DATA ##############################

# Load the file of interest
path_to_results = '../../results/data/melanoma/random-forest/melanoma_imbalanced_80_20_100_0.npz'
file_results = np.load(path_to_results)

# build an roc_auc object for each curve
kind_interp = 'linear'
roc_by_config = []
roc_fitted_by_config = []
for config in file_results['rocs']:
    # for each cross validation
    roc_by_fold = []
    roc_fitted_by_fold = []
    for k in config:
        roc_by_fold.append(roc_auc(k[0], k[1], k[2], k[3]))
        # Declare the x for the linspace
        roc_fitted_by_fold.append(interp1d(k[0], k[1], kind=kind_interp))

    roc_by_config.append(roc_by_fold)
    roc_fitted_by_config.append(roc_fitted_by_fold)

############################## PLOTTING DATA ##############################

plt.figure()
# Compute a single ROC using the mean and the standard deviation with interpolation in the middle
for cof in range(len(roc_by_config)):
    tpr_arr = []
    for roc in roc_fitted_by_config[cof]:
        # create the fpr in order to compute the tpr
        fpr = np.linspace(0., 1., 1000, endpoint=True)
        tpr_arr.append(roc(fpr))

    # Get the mean auc 
    auc_arr = []
    for roc in roc_by_config[cof]:
        auc_arr.append(roc.auc)
    auc_mean = np.mean(auc_arr, axis=0)
    auc_std = np.std(auc_arr, axis=0)

    # Plot the mean curve
    mean_tpr = np.mean(tpr_arr, axis=0)
    std_tpr = np.std(tpr_arr, axis=0)
    fpr = np.linspace(0., 1., 1000, endpoint=True)
    plt.plot(fpr, mean_tpr, lw=2, color=tableau20[cof],
             label=bal_arr[cof]+' - AUC={:.2f} +- {:.2f}'.format(auc_mean, auc_std))
    #plt.fill_between(fpr, mean_tpr+std_tpr, mean_tpr-std_tpr, facecolor=tableau20[cof], alpha=0.2)

# Plot the legend on the left of the figure
lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, .5), prop={'size':7})

# Set the limit
plt.xlim(0, 1)
plt.ylim(0, 1)

# Put some label
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()
