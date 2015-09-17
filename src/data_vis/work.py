# <nbformat>3</nbformat>


"""

"""

#### Some stuff needed

import numpy as np
import pandas as pd
import xray
import sys
import os


#### Check some data integrity

data_path = '../../results/data/logistic_regression/'
# Check that there's no wired label
gt_labels = set()
pred_labels = set()
for f in os.listdir(data_path):
    data = np.load(os.path.join(data_path, f))
    [gt_labels.update(set(x)) for x in data['gt_labels'].ravel()]
    [pred_labels.update(set(x)) for x in data['pred_labels'].ravel()]

print gt_labels
print pred_labels

#### Create the Xray data (toy example)

cvl = ['_cv{0:d}_'.format(x) for x in range(3)]
ll = ['gt', 'pd']

A = [['A'+cv+l for cv in cvl] for l in ll]
B = [['B'+cv+l for cv in cvl] for l in ll]

aa = xray.DataArray(A, coords=[ll, cvl], dims=['cross_val_id', 'label'])
bb = xray.DataArray(B, coords=[ll, cvl], dims=['cross_val_id', 'label'])

data = xray.Dataset({'A':aa, 'B':bb})
print data

#### Some extra functions to evaluate (there's some tests)
truth = {3:'tp', 1:'fp',
         2:'fn', 4:'tn'}

def evaluate(pred, gt):
    pos_sample = 1.
    neg_sample = -1.
    truth_LUT = dict(zip(truth.values(), truth.keys()))
    confusion_matrix_label = np.zeros(pred.shape, dtype=np.int)
    def get_indx(pred_condition, gt_condition):
        return np.logical_and(pred==pred_condition, gt==gt_condition)

    confusion_matrix_label[get_indx(pos_sample, pos_sample)] = truth_LUT['tp']
    confusion_matrix_label[get_indx(neg_sample, neg_sample)] = truth_LUT['tn']
    confusion_matrix_label[get_indx(pos_sample, neg_sample)] = truth_LUT['fp']
    confusion_matrix_label[get_indx(neg_sample, pos_sample)] = truth_LUT['fn']
    return confusion_matrix_label

# Test
# p = np.array([1, 1, -1, -1])
# g = np.array([1, -1, 1, -1])
# assert((evaluate(p,g)==np.array([3,1,2,4])).all)
# (evaluate(p,g)==np.array([3,1,2,4])).all() # to see something if TRUE, we are fine

####

def confusion_matrix(x):
    from collections import Counter
    from itertools import repeat
    d = Counter(dict(zip(truth.keys(), repeat(0))))
    d.update(x)
    assert d.keys()==truth.keys(), "Error while computing the confusion matrix"
    # return np.array(d.values(), dtype=float)/sum(d.values())
    return d.values()

#### load the data as xray

def compute_confusion_matrix(d):
    return np.array([confusion_matrix(x) for x in d.ravel()]).reshape(d.shape)

def get_data_as_xray(fName):
    # some definitions
    data_path = '../../results/data/logistic_regression/'
    balancing_method = ['No balancing', 'ROS', 'SMOTE-reg', 'SMOTE-bord1',
                        'SMOTE-bord2', 'SMOTE-SVM', 'RUS', 'TL', 'Clus', 'NM1',
                        'NM2', 'NM3', 'CNN', 'OSS', 'NCR', 'EasyEns', 'BalanceCas',
                        'SMOTE+ENN', 'SMOTE+TL']

    def get_evaluation_labels(pred, gt):
        evaluation_label = [evaluate(p,g) for (p,g) in zip(pred.ravel(), gt.ravel())]
        return np.array(evaluation_label).reshape(pred.shape)

    def get_data(f):
        d = np.load(os.path.join(data_path, f))
        return [d['gt_labels'], d['pred_labels'],
                get_evaluation_labels(d['pred_labels'], d['gt_labels'])]

    def make_xray(d):
        a = xray.DataArray(d, coords=[['gt', 'pred', 'eval'], balancing_method, range(10)],
                              dims=['label', 'balancing', 'crossval_id'])
        # b = xray.DataArray(compute_confusion_matrix(d[2]), coords=[truth.values(), balancing_method, range(10)],
        #                       dims=['conf_matrix', 'balancing', 'crossval_id'])
        # return xray.concat([a,b])
        return (a,d)

    return make_xray(get_data(fName))

data_path = '../../results/data/logistic_regression/'
# logistic_results = xray.Dataset({f:get_data_as_xray(f)
#                                  for f in os.listdir(data_path)})

####

# xx = {f:get_data_as_xray(f) for f in os.listdir(data_path)}



####
####
####

