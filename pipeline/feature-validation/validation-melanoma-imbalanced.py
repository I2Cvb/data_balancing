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
path_to_results = '../../results/data/melanoma/random-forest/melanoma_imbalanced_80_20_100_2.npz'
file_results = np.load(path_to_results)

# ############################## ROC ANALYSIS ##############################

# # build an roc_auc object for each curve
# kind_interp = 'linear'
# roc_by_config = []
# roc_fitted_by_config = []
# for config in file_results['rocs']:
#     # for each cross validation
#     roc_by_fold = []
#     roc_fitted_by_fold = []
#     for k in config:
#         roc_by_fold.append(roc_auc(k[0], k[1], k[2], k[3]))
#         # Declare the x for the linspace
#         roc_fitted_by_fold.append(interp1d(k[0], k[1], kind=kind_interp))

#     roc_by_config.append(roc_by_fold)
#     roc_fitted_by_config.append(roc_fitted_by_fold)

# ############################## PLOTTING DATA ##############################

# plt.figure()
# # Compute a single ROC using the mean and the standard deviation with interpolation in the middle
# for cof in range(len(roc_by_config)):
#     tpr_arr = []
#     for roc in roc_fitted_by_config[cof]:
#         # create the fpr in order to compute the tpr
#         fpr = np.linspace(0., 1., 100, endpoint=True)
#         tpr_arr.append(roc(fpr))

#     # Get the mean auc 
#     auc_arr = []
#     for roc in roc_by_config[cof]:
#         auc_arr.append(roc.auc)
#     auc_mean = np.mean(auc_arr, axis=0)
#     auc_std = np.std(auc_arr, axis=0)

#     # Plot the mean curve
#     mean_tpr = np.mean(tpr_arr, axis=0)
#     std_tpr = np.std(tpr_arr, axis=0)
#     fpr = np.linspace(0., 1., 100, endpoint=True)
#     plt.plot(fpr, mean_tpr, lw=2, color=tableau20[cof],
#              label=bal_arr[cof]+' - AUC={:.2f} +- {:.2f}'.format(auc_mean, auc_std))
#     #plt.fill_between(fpr, mean_tpr+std_tpr, mean_tpr-std_tpr, facecolor=tableau20[cof], alpha=0.2)

# # Plot the legend on the left of the figure
# lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, .5), prop={'size':7})

# # Set the limit
# plt.xlim(0, 1)
# plt.ylim(0, 1)

# # Put some label
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')

# plt.show()


# ############################## PRECISION - RECALL ANALYSIS ##############################

# # build an roc_auc object for each curve
# kind_interp = 'linear'
# pr_by_config = []
# pr_fitted_by_config = []
# for config_predl, config_predp, config_gt in zip(file_results['pred_labels'], 
#                                                  file_results['pred_probs'], 
#                                                  file_results['gt_labels']):
#     # for each cross validation
#     pr_by_fold = []
#     pr_fitted_by_fold = []

#     # get the gt and pred for a single cv
#     for cv_predl, cv_predp, cv_gt in zip(config_predl, 
#                                          config_predp,
#                                          config_gt):
#         # Compute the current recall precision curve
#         pr_curve = precision_recall_curve(cv_gt, cv_predp[:, 1])
#         # Insert the down limit
#         print pr_curve[0].shape
#         pr1 = np.concatenate(([0.], pr_curve[0]))
#         pr2 = np.concatenate(([1.], pr_curve[1]))
#         pr_curve = (pr1, pr2)
#         print pr_curve[0].shape
#         pr_by_fold.append(pr_curve)
#         # Declare the x for the linspace
#         pr_fitted_by_fold.append(interp1d(pr_curve[0], pr_curve[1], kind=kind_interp))

#     pr_by_config.append(pr_by_fold)
#     pr_fitted_by_config.append(pr_fitted_by_fold)

# ############################## PLOTTING DATA ##############################

# plt.figure()
# # Compute a single ROC using the mean and the standard deviation with interpolation in the middle
# for cof in range(len(pr_by_config)):
#     prec_arr = []
#     for pr in pr_fitted_by_config[cof]:
#         # create the fpr in order to compute the tpr
#         rec = np.linspace(0., 1., 40, endpoint=True)
#         prec_arr.append(pr(rec))

#     # # Get the mean auc 
#     # auc_arr = []
#     # for roc in roc_by_config[cof]:
#     #     auc_arr.append(roc.auc)
#     # auc_mean = np.mean(auc_arr, axis=0)
#     # auc_std = np.std(auc_arr, axis=0)

#     # Plot the mean curve
#     mean_prec = np.mean(prec_arr, axis=0)
#     std_prec = np.std(prec_arr, axis=0)
#     rec = np.linspace(0, 1., 40, endpoint=True)
#     plt.plot(rec, mean_prec, lw=2, color=tableau20[cof],
#              label=bal_arr[cof])#+' - AUC={:.2f} +- {:.2f}'.format(auc_mean, auc_std))
#     #plt.fill_between(rec, mean_prec+std_prec, mean_prec-std_prec, facecolor=tableau20[cof], alpha=0.2)

# # Plot the legend on the left of the figure
# lgd = plt.legend(loc='center left', bbox_to_anchor=(1.05, .5), prop={'size':7})

# # Set the limit
# plt.xlim(0, 1)
# plt.ylim(0, 1)

# # Put some label
# plt.xlabel('Recall')
# plt.ylabel('Precision')

# plt.show()

############################## COMPUTE THE DIFFERENT METRICS ##############################

# build an roc_auc object for each curve

### all the mean metrics for each configuration
mean_sens_config = []
mean_spec_config = []

### all the std metrics for each configuration
std_sens_config = []
std_spec_config = []

for config_predl, config_predp, config_gt in zip(file_results['pred_labels'], 
                                                 file_results['pred_probs'], 
                                                 file_results['gt_labels']):
    # for each cross validation
    cv_sens = []
    cv_spec = []
    
    # get the gt and pred for a single cv
    for cv_predl, cv_predp, cv_gt in zip(config_predl, 
                                         config_predp,
                                         config_gt):

        # Compute the sensitivity and specificity for each round
        sens, spec = LabelsToSensitivitySpecificity(cv_gt, cv_predl)
        cv_sens.append(sens)
        cv_spec.append(spec)

    # Compute the mean-std of the different cvs
    mean_sens_config.append(np.mean(cv_sens))
    mean_spec_config.append(np.mean(cv_spec))
    std_sens_config.append(np.std(cv_sens))
    std_spec_config.append(np.std(cv_spec))
