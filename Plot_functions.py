# -*- coding: utf-8 -*-
"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar
"""



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import colors
from Pruebas_Nico_OT.Utilitys import ensure_dir, acomodate_df, filter_df
from statannot.statannot import add_stat_annotation

def plot_violin_pair_comparison(df,compared_methods,subjects,save_dir="",save_fig=False,prefix_name=""):

  df_aux= df.copy()
  df_aux = filter_df(df_aux,compared_methods)

  fig = plt.figure(figsize=[20,10])

  ax = fig.add_subplot(1, 1, 1)

  ax = sns.violinplot(data = df_aux, x= "Subject" , y = pd.to_numeric(df["Test Accuracy [%]"]),  ax = ax, hue = "Method",
                        scale = "width" ,split=True,inner = None)

  ax.grid(b = True, color = 'black', linestyle = ':', linewidth = 1, alpha = 0.4)

  ax.set_xticklabels(subjects)

  df_aux = acomodate_df(df_aux,compared_methods)
  # Comparisons for statistical test
  box_list = [((N_S, 0), (N_S,1)) for N_S in subjects]

  add_stat_annotation(ax, data = df_aux, x = "Subject", y = pd.to_numeric(df["Test Accuracy [%]"]),  hue = "Method",
                      box_pairs = box_list, test = 't-test_paired', text_format = 'star', loc = 'inside', 
                      pvalue_thresholds = [[1,"ns"], [0.01,"*"]],verbose=False)
  
  if save_fig:
    ensure_dir(save_dir)
    file_name = prefix_name+"_"+compared_methods[0]+"_vs_"+compared_methods[1]+"_.pdf"
    plt.savefig(save_dir+file_name,pad_inches=0.5) 

  plt.show()

  return 

def plot_violin_pair_comparison(df,compared_methods,subjects,save_dir="",save_fig=False,prefix_name=""):

  df_aux = df.copy()
  df_aux = filter_df(df_aux,compared_methods)

  fig = plt.figure(figsize=[20,10])

  ax = fig.add_subplot(1, 1, 1)

  ax = sns.violinplot(data = df_aux, x= "Subject" , y = pd.to_numeric(df["Test Accuracy [%]"]),  ax = ax, hue = "Method",
                        scale = "width" ,split=True,inner = None)

  ax.grid(b = True, color = 'black', linestyle = ':', linewidth = 1, alpha = 0.4)

  ax.set_xticklabels(subjects)

  df_aux = acomodate_df(df_aux,compared_methods)
  # Comparisons for statistical test
  box_list = [((N_S, 0), (N_S,1)) for N_S in subjects]

  add_stat_annotation(ax, data = df_aux, x = "Subject", y = pd.to_numeric(df["Test Accuracy [%]"]),  hue = "Method",
                      box_pairs = box_list, test = 't-test_paired', text_format = 'star', loc = 'inside', 
                      pvalue_thresholds = [[1,"ns"], [0.01,"*"]],verbose=False)
  
  if save_fig:
    ensure_dir(save_dir)
    file_name = prefix_name+"_"+compared_methods[0]+"_vs_"+compared_methods[1]+"_.pdf"
    plt.savefig(save_dir+file_name, bbox_inches='tight') 

  plt.show()

  return 


# @title Plot utilities
def plot_lda_db(lda):
  
  nx, ny = 200, 200
  x_min, x_max = plt.xlim()
  y_min, y_max = plt.ylim()
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                        np.linspace(y_min, y_max, ny))

  Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
  Z = Z[:, 1].reshape(xx.shape)
  plt.contour(xx[:,0:2], yy, Z, [0.5], linewidths=2., colors='black')
  return
  
def plot_mean_lda(lda):
  plt.plot(lda.means_[0][0], lda.means_[0][1],
        '*', color='yellow', markersize=15, markeredgecolor='grey',label='Estimated mean')
  plt.plot(lda.means_[1][0], lda.means_[1][1],
        '*', color='yellow', markersize=15, markeredgecolor='grey')
  return

def plot_area_lda(lda):
  nx, ny = 200, 200
  x_min, x_max = plt.xlim()
  y_min, y_max = plt.ylim()
  xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                        np.linspace(y_min, y_max, ny))

  Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
  Z = Z[:, 1].reshape(xx.shape)
  plt.pcolormesh(xx, yy, Z ,alpha=0.5,cmap='BrBG',zorder=0)
  return

