# -*- coding: utf-8 -*-
"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from Pruebas_Nico_OT.Distance_functions import penalized_coupling

#  Auxiliar Functions
def split_data_umbalanced(X, Y, Balance, shuffle=False):
  # The data that is not used in train is used in val.

  if shuffle:
    index = np.random.permutation(Y.shape[0])
    Y = Y[index]
    X = X[index]

  Y_c1= Y[Y==1]
  X_c1= X[Y==1]

  Y_c2= Y[Y==2]
  X_c2= X[Y==2]

  # Validation Pints run
  X_trn = np.vstack([X_c1[0:Balance[0],:] , X_c2[0:Balance[1],:]] )
  Y_trn = np.concatenate([Y_c1[0:Balance[0]] , Y_c2[0:Balance[1]]])

  # DEJAR FIJO EL TEST
  X_val = np.vstack([X_c1[Balance[0]::,:],X_c2[Balance[1]::,:]])
  Y_val = np.concatenate([Y_c1[Balance[0]::], Y_c2[Balance[1]::]])

  return X_trn , Y_trn , X_val , Y_val


def train_transport(ot_object,Xs, Xt, yt, ys,X_test,Y_test, clf, k=0,wrong_cls=True,balanced_target=True ,balanced_source=True,metrica="acc",verbose=False,penalized="d"):

  # Compute the new coupling with the penalized cost matrix
  G01 = penalized_coupling(Xs, Xt, yt, ys, clf=clf, k=k,wrong_cls=wrong_cls,balanced_target=balanced_target, balanced_source=balanced_source,penalized=penalized)

  # Replace the coupling
  ot_object.coupling_ = G01

  # transport source samples onto target samples
  T_source_lda = ot_object.transform(Xs = X_test)

  if metrica=="acc":

   performance = clf.score(T_source_lda,Y_test)

  elif metrica=="F1":

    y_pred = clf.predict(T_source_lda)

    performance = f1_score(Y_test,y_pred)

  if verbose:
    print("K="+str(k)+":\t\t" + str(performance))

  return performance

def name_array(name,ref_array):
  name_ret= []
  for n in range(ref_array.shape[0]):
    name_ret.append(name)
  return name_ret

def filter_df(df,compared_methods):
  df_aux = df.copy()
  filter = df_aux["Method"] == compared_methods[0] 
  filter2 = df_aux["Method"] == compared_methods[1]
  filter = filter  + filter2
  df_aux = df_aux[filter]

  return df_aux

def acomodate_df(df,compared_methods):
  df_aux = df.copy()
  df_aux["Method"][df_aux["Method"]==compared_methods[1]] = 1
  df_aux["Method"][df_aux["Method"]==compared_methods[0]] = 0

  df_aux["Subject"]= pd.to_numeric(df_aux["Subject"])

  return df_aux


def ensure_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)




def get_mean_acc(df,Method,steps):
  df_aux = df.copy()
  df_aux["Balance"]= pd.to_numeric(df_aux["Balance"])
  df_aux["Test Accuracy [%]"]= pd.to_numeric(df_aux["Test Accuracy [%]"])
  filter = df_aux["Method"] == Method

  NO_OT_full = df_aux[filter]

  NO_OT_array = 0
  for n in range (steps):

    filter = NO_OT_full["Balance"] == n

    NO_OT = NO_OT_full[filter]
    NO_OT = NO_OT["Test Accuracy [%]"]

    NO_OT_array = np.append(NO_OT_array,np.mean(NO_OT))

  NO_OT_array = np.delete(NO_OT_array,0)

  return NO_OT_array


def get_std_acc(df,Method,steps):
  df_aux = df.copy()
  df_aux["Balance"]= pd.to_numeric(df_aux["Balance"])
  df_aux["Test Accuracy [%]"]= pd.to_numeric(df_aux["Test Accuracy [%]"])
  filter = df_aux["Method"] == Method

  NO_OT_full = df_aux[filter]

  NO_OT_array = 0
  for n in range (steps):

    filter = NO_OT_full["Balance"] == n

    NO_OT = NO_OT_full[filter]
    NO_OT = NO_OT["Test Accuracy [%]"]

    NO_OT_array = np.append(NO_OT_array,np.std(NO_OT))

  NO_OT_array = np.delete(NO_OT_array,0)

  return NO_OT_array

