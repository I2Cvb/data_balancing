# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 15:28:24 2016

@author: mojdeh
"""

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

from cf_validation import cf_validation
from protoclass.validation.utility import MakeTable

#fread = pd.read_csv(join(dataPath, 'featureList.csv'))
savepath = '../../results/data/melanoma/random-forest/'
datapath = '../../data/'
fread = pd.read_table(join(datapath, 'FeatureList.txt'))
FeatureLists = fread.values
FeatureLists = FeatureLists[:,0]

mean_sens = []
mean_spec = []
mean_prec = []
mean_npv = []
mean_gmean = []
mean_f1sc = []
mean_mcc = []
mean_iba = []
mean_cost = []

std_sens = []
std_spec = []
std_prec = []
std_npv = []
std_gmean = []
std_f1sc = []
std_mcc = []
std_iba = []
std_cost = []

### Grouping the measurements for all the features 

for Id in range(0,24): 
    path_to_result = savepath + '/melanoma_imbalanced_80_20_100_' + str(Id) + '.npz'
    
    mean_sens_config, mean_spec_config, mean_prec_config, mean_npv_config, mean_gmean_config, mean_acc_config, mean_f1sc_config, mean_mcc_config, mean_iba_config, mean_cost_config, std_sens_config, std_spec_config, std_prec_config, std_npv_config, std_gmean_config, std_acc_config, std_f1sc_config, std_mcc_config, std_iba_config, std_cost_config, pr_by_config, pr_fitted_by_config,roc_by_config,roc_fitted_by_config = cf_validation(path_to_result)
    
    ### Mean values
    mean_sens.append(mean_sens_config)
    mean_spec.append(mean_spec_config)
    mean_prec.append(mean_prec_config)
    mean_npv.append(mean_npv_config)
    mean_gmean.append(mean_gmean_config)
    mean_f1sc.append(mean_f1sc_config)
    mean_mcc.append(mean_mcc_config)
    mean_iba.append(mean_iba_config)
    mean_cost.append(mean_cost_config)
    
    ### standard deviation values
#    std_sens.append(std_sens_config)
#    std_spec.append(std_spec_config)
#    std_prec.append(std_prec_config)
#    std_npv.append(std_npv_config)
#    std_gmean.append(std_gmean_config)
#    std_f1sc.append(std_f1sc_config)
#    std_mcc.append(std_mcc_config)
#    std_iba.append(std_iba_config)

### Converting the list to array 
#### Mean values 
mean_sens = np.asarray(mean_sens)*100
mean_spec = np.asarray(mean_spec)*100  
mean_sens_spec = np.empty([mean_sens.shape[0], mean_sens.shape[1]*2])
mean_sens_spec[:,np.arange(0,28,2)] = mean_sens
mean_sens_spec[:,np.arange(1,28,2)] = mean_spec
mean_prec = np.asarray(mean_prec)*100
mean_npv = np.array(mean_npv)*100
mean_gmean = np.array(mean_gmean)*100
mean_f1sc = np.array(mean_f1sc)*100
mean_mcc = np.array(mean_mcc)*100
mean_iba = np.array(mean_iba)*100
mean_cost = np.array(mean_cost)*100

#### Standard deviations   
#std_sens = np.asarray(std_sens)*100
#std_spec = np.asarray(std_spec)*100  
#std_prec = np.asarray(std_prec)*100
#std_npv = np.array(std_npv)*100
#std_gmean = np.array(std_gmean)*100
#std_f1sc = np.array(std_f1sc)*100
#std_mcc = np.array(std_mcc)*100
#std_iba = np.array(std_iba)*100  
MakeTable ( mean_sens_spec, FeatureLists, savepath, 'sens_spec_random_forest',ext='.tex')
MakeTable ( mean_prec, FeatureLists, savepath, 'prec_random_forest',ext='.tex')
MakeTable ( mean_npv, FeatureLists,savepath,'npv_random_forest',ext='.tex')
MakeTable ( mean_gmean, FeatureLists,savepath,'gmean_random_forests',ext='.tex')
MakeTable ( mean_f1sc, FeatureLists,savepath, 'f1sc_random_forest',ext='.tex')
MakeTable ( mean_mcc, FeatureLists,savepath, 'mcc_random_forest',ext='.tex')
MakeTable ( mean_iba, FeatureLists,savepath, 'iba_random_forest',ext='.tex')
MakeTable ( mean_cost, FeatureLists,savepath, 'cost_value_random_forest',ext='.tex')




    

#==============================================================================
#     # Save the results somewhere
#     if not os.path.exists(path_to_save):
#     	os.makedirs(path_to_save)
# 
#     from os.path import basename
#     saving_filename = 'melanoma_imbalanced_80_20_' + str(ntree) +  '_' + str(I)
#     saving_path = join(path_to_save, saving_filename)
#     np.savez(saving_path, gt_labels=gt_labels, pred_labels=pred_labels, pred_probs=pred_probs, rocs=rocs)
#     tosave={}
#     tosave['rocs'] = rocs
#     tosave['pred_labels'] = pred_labels
#     tosave['pred_probs'] = pred_probs
#     tosave['gt_labels'] = gt_labels
#     saving_path = join(path_to_save, saving_filename)
#     from scipy.io import savemat
#     savemat(saving_path, tosave)
#==============================================================================
