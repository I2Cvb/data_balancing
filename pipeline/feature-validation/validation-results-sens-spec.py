#title           :validation-results-sens-spec.py
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
from matplotlib.patches import Ellipse
# Sys library
import sys
# OS library
import os
from os.path import join

from protoclass.validation.validation import LabelsToSensitivitySpecificity

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

############################## LOADING DATA ##############################

for n_db in range(1, 31):

    path_to_results = '../../results/data/result_x' + str(n_db) + 'data.npz'
    file_results = np.load(path_to_results)

    # build an roc_auc object for each curve
    sens_by_config = []
    spec_by_config = []
    for conf_gt_label, conf_pred_label in zip(file_results['gt_labels'], file_results['pred_labels']):
        # for each cross validation
        sens_by_fold = []
        spec_by_fold = []
        for k_gt_label, k_pred_label in zip(conf_gt_label, conf_pred_label):
            sens, spec = LabelsToSensitivitySpecificity(k_gt_label, k_pred_label)
            # Append the results
            sens_by_fold.append(sens)
            spec_by_fold.append(spec)

        sens_by_config.append(np.array(sens_by_fold))
        spec_by_config.append(np.array(spec_by_fold))

    ############################## PLOTTING DATA ##############################
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    # Compute a single sensitivity and specificity using the mean and the standard deviation with 
    #interpolation in the middle
    for (cof, (sens_arr, spec_arr)) in enumerate(zip(sens_by_config, spec_by_config)):
        
        # Compute the mean and std of sensitivity and specificity
        mean_sens = np.mean(sens_arr, axis=0)
        mean_spec = np.mean(spec_arr, axis=0)
        std_sens = np.std(sens_arr, axis=0)
        std_spec = np.std(spec_arr, axis=0)

        # Plot the corresponding ellipse
        ### Create the ellipse
        ### We should maybe modify the angle depending of the ratio of spec over sens
        ell = Ellipse(xy=(mean_spec, mean_sens), height=std_sens, width=std_spec, angle=0)

        ### Make the plot
        ax.add_artist(ell)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(.5)
        ell.set_facecolor(tableau20[cof])
        
    # Set the limit
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()
