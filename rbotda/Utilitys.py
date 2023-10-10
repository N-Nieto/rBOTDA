# -*- coding: utf-8 -*-
"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar
"""
import os
import numpy as np
import pandas as pd


#  Auxiliar Functions
def split_data_unbalanced(X, Y, Balance, shuffle=False):
    # The data that is not used in train is used in val.
    if shuffle:
        index = np.random.permutation(Y.shape[0])
        Y = Y[index]
        X = X[index]

    Y_c1 = Y[Y == 1]
    X_c1 = X[Y == 1]

    Y_c2 = Y[Y == 2]
    X_c2 = X[Y == 2]

    # Validation Pints run
    X_trn = np.vstack([X_c1[0:Balance[0], :], X_c2[0:Balance[1], :]])
    Y_trn = np.concatenate([Y_c1[0:Balance[0]], Y_c2[0:Balance[1]]])

    # DEJAR FIJO EL TEST
    X_val = np.vstack([X_c1[Balance[0]::, :], X_c2[Balance[1]::, :]])
    Y_val = np.concatenate([Y_c1[Balance[0]::], Y_c2[Balance[1]::]])

    return X_trn, Y_trn, X_val, Y_val


def filter_df(df, compared_methods):
    df_aux = df.copy()
    filter = df_aux["Method"] == compared_methods[0]
    filter2 = df_aux["Method"] == compared_methods[1]
    filter = filter + filter2
    df_aux = df_aux[filter]

    return df_aux


def acomodate_df(df, compared_methods):
    df_aux = df.copy()
    df_aux["Method"][df_aux["Method"] == compared_methods[1]] = 1
    df_aux["Method"][df_aux["Method"] == compared_methods[0]] = 0

    df_aux["Subject"] = pd.to_numeric(df_aux["Subject"])

    return df_aux


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_mean_acc(df, method, steps):
    df_aux = df.copy()
    df_aux["Balance"] = pd.to_numeric(df_aux["Balance"])
    df_aux["Test Accuracy [%]"] = pd.to_numeric(df_aux["Test Accuracy [%]"])
    filter = df_aux["Method"] == method

    NO_OT_full = df_aux[filter]

    NO_OT_array = 0
    for n in range(steps):
        filter = NO_OT_full["Balance"] == n

        NO_OT = NO_OT_full[filter]
        NO_OT = NO_OT["Test Accuracy [%]"]

        NO_OT_array = np.append(NO_OT_array, np.mean(NO_OT))

    NO_OT_array = np.delete(NO_OT_array, 0)

    return NO_OT_array


def get_std_acc(df, method, steps):
    df_aux = df.copy()
    df_aux["Balance"] = pd.to_numeric(df_aux["Balance"])
    df_aux["Test Accuracy [%]"] = pd.to_numeric(df_aux["Test Accuracy [%]"])
    filter = df_aux["Method"] == method

    NO_OT_full = df_aux[filter]

    NO_OT_array = 0
    for n in range(steps):

        filter = NO_OT_full["Balance"] == n

        NO_OT = NO_OT_full[filter]
        NO_OT = NO_OT["Test Accuracy [%]"]

        NO_OT_array = np.append(NO_OT_array, np.std(NO_OT))

    NO_OT_array = np.delete(NO_OT_array, 0)

    return NO_OT_array
