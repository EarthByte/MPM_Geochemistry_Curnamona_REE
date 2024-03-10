#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code utilizes the DevNet network to implement anomaly identification on the training data.
Code modified from Pang, G., Shen, C., & Van Den Hengel, A. (2019, July).
Deep anomaly detection with deviation networks.
In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining (pp. 353-362).

"""


import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def dataLoading(path):
    # loading data
    df = pd.read_csv(path)

    labels = df['Class']

    x_df = df.drop(['Class'], axis=1)

    x = x_df.values
    print("Data shape: (%d, %d)" % x.shape)

    return x, labels


def aucPerformance(mse, labels):
    roc_auc = roc_auc_score(labels, mse)
    ap = average_precision_score(labels, mse)
    print("AUC-ROC: %.4f, AUC-PR: %.4f" % (roc_auc, ap))
    return roc_auc, ap


def writeResults(name, n_samples, dim, n_samples_trn, n_outliers_trn, n_outliers, rauc, ap, std_auc, std_ap,
                 train_time, test_time, path="./georesults/auc_performance_cl0.5.csv"):
    csv_file = open(path, 'a')
    row = name + "," + str(n_samples) + "," + str(dim) + ',' + str(n_samples_trn) + ',' + str(
        n_outliers_trn) + ',' + str(n_outliers) + ','  + str(rauc) + "," + str(std_auc) + "," + str(
        ap) + "," + str(std_ap) + "," + str(train_time) + "," + str(test_time) + "\n"
    csv_file.write(row)
