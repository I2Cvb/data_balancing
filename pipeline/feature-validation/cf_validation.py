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
from protoclass.validation.metric import LabelsToSensitivitySpecificity, LabelsToPrecisionNegativePredictiveValue, LabelsToGeometricMean, LabelsToAccuracy, LabelsToF1score, LabelsToMatthewCorrCoef, LabelsToGeneralizedIndexBalancedAccuracy, LabelsToCostValue
from sklearn.metrics import precision_recall_curve


from collections import namedtuple
roc_auc = namedtuple('roc_auc', ['fpr', 'tpr', 'thresh', 'auc'])

def cf_validation (path_to_results):

    # Initialization to the data paths 
    #path_to_results = sys.argv[1]
    #path_to_save = sys.argv[2]
    
    
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
    #path_to_results = '../../results/data/melanoma/naive-bayes/melanoma_imbalanced_80_20_100_2.npz'
    file_results = np.load(path_to_results)
    
    ############################## ROC ANALYSIS ##############################
    
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
    
    
    ############################## PRECISION - RECALL ANALYSIS ##############################
    
    # build an roc_auc object for each curve
    kind_interp = 'linear'
    pr_by_config = []
    pr_fitted_by_config = []
    for config_predl, config_predp, config_gt in zip(file_results['pred_labels'], 
                                                     file_results['pred_probs'], 
                                                     file_results['gt_labels']):
        # for each cross validation
        pr_by_fold = []
        pr_fitted_by_fold = []
    
        # get the gt and pred for a single cv
        for cv_predl, cv_predp, cv_gt in zip(config_predl, 
                                             config_predp,
                                             config_gt):
            # Compute the current recall precision curve
            pr_curve = precision_recall_curve(cv_gt, cv_predp[:, 1])
            # Insert the down limit
            print pr_curve[0].shape
            pr1 = np.concatenate(([0.], pr_curve[0]))
            pr2 = np.concatenate(([1.], pr_curve[1]))
            pr_curve = (pr1, pr2)
            print pr_curve[0].shape
            pr_by_fold.append(pr_curve)
            # Declare the x for the linspace
            pr_fitted_by_fold.append(interp1d(pr_curve[0], pr_curve[1], kind=kind_interp))
    
        pr_by_config.append(pr_by_fold)
        pr_fitted_by_config.append(pr_fitted_by_fold)
    
    
    ############################## COMPUTE THE DIFFERENT METRICS ##############################
    
    # build an roc_auc object for each curve
    
    ### all the mean metrics for each configuration
    mean_sens_config = []
    mean_spec_config = []
    mean_prec_config = []
    mean_npv_config = []
    mean_gmean_config = []
    mean_acc_config = []
    mean_f1sc_config = []
    mean_mcc_config = []
    mean_iba_config = []
    mean_cost_config = []
    ### all the std metrics for each configuration
    std_sens_config = []
    std_spec_config = []
    std_prec_config = []
    std_npv_config = []
    std_gmean_config = []
    std_acc_config = []
    std_f1sc_config = []
    std_mcc_config = []
    std_iba_config = []
    std_cost_config = []
    
    for config_predl, config_predp, config_gt in zip(file_results['pred_labels'], 
                                                     file_results['pred_probs'], 
                                                     file_results['gt_labels']):
        # for each cross validation
        cv_sens = []
        cv_spec = []
        cv_prec = []
        cv_npv = []
        cv_gmean = []
        cv_acc = []
        cv_f1sc = []
        cv_mcc = []
        cv_iba = []
	cv_cost =[]
        
        # get the gt and pred for a single cv
        for cv_predl, cv_predp, cv_gt in zip(config_predl, 
                                             config_predp,
                                             config_gt):
    
            # Compute the different metrics for each round of cv
            sens, spec = LabelsToSensitivitySpecificity(cv_gt, cv_predl) 
            prec, npv = LabelsToPrecisionNegativePredictiveValue(cv_gt, cv_predl)
            gmean = LabelsToGeometricMean(cv_gt, cv_predl)
            acc = LabelsToAccuracy(cv_gt, cv_predl)
            f1sc = LabelsToF1score(cv_gt, cv_predl)
            mcc = LabelsToMatthewCorrCoef(cv_gt, cv_predl)
            iba = LabelsToGeneralizedIndexBalancedAccuracy(cv_gt, cv_predl)
            cval = LabelsToCostValue(cv_gt, cv_predl)

            cv_sens.append(sens)
            cv_spec.append(spec)
            cv_prec.append(prec)
            cv_npv.append(npv)
            cv_gmean.append(gmean)
            cv_acc.append(acc)
            cv_f1sc.append(f1sc)
            cv_mcc.append(mcc)
            cv_iba.append(iba)
	    cv_cost.append(cval)
    
        # Compute the mean-std of the different cvs
        ### mean
        mean_sens_config.append(np.mean(cv_sens))
        mean_spec_config.append(np.mean(cv_spec))
        mean_prec_config.append(np.mean(cv_prec))
        mean_npv_config.append(np.mean(cv_npv))
        mean_gmean_config.append(np.mean(cv_gmean))
        mean_acc_config.append(np.mean(cv_acc))
        mean_f1sc_config.append(np.mean(cv_f1sc))
        mean_mcc_config.append(np.mean(cv_mcc))
        mean_iba_config.append(np.mean(cv_iba))
	mean_cost_config.append(np.mean(cv_cost))

    
        ### std
        std_sens_config.append(np.std(cv_sens))
        std_spec_config.append(np.std(cv_spec))
        std_prec_config.append(np.std(cv_prec))
        std_npv_config.append(np.std(cv_npv))
        std_gmean_config.append(np.std(cv_gmean))
        std_acc_config.append(np.std(cv_acc))
        std_f1sc_config.append(np.std(cv_f1sc))
        std_mcc_config.append(np.std(cv_mcc))
        std_iba_config.append(np.std(cv_iba))
        std_cost_config.append(np.std(cv_cost))
    
    return (mean_sens_config, mean_spec_config, mean_prec_config, mean_npv_config, mean_gmean_config, mean_acc_config, mean_f1sc_config, mean_mcc_config,
            mean_iba_config, mean_cost_config, std_sens_config, std_spec_config, std_prec_config, std_npv_config, std_gmean_config, std_acc_config, std_f1sc_config, std_mcc_config,
            std_iba_config, std_cost_config, pr_by_config, pr_fitted_by_config,roc_by_config,roc_fitted_by_config)
