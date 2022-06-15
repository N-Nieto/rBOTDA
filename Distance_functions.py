# -*- coding: utf-8 -*-
"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar
"""
import numpy as np
import ot.plot

## Distance pack
def distance_to_hyperplane(X,clf):
  b = clf.intercept_ 
  W = clf.coef_ 
  # Module
  mod = np.sqrt(np.sum(np.power(W,2)))
  # distance
  d = np.abs(np.dot(X,W.T)+b)/mod
  return d[:,0]


def P_matix(X,clf,k=1):
  # Compute distance
  d = distance_to_hyperplane(X,clf)
  d = np.power(d,k) 

  # Normalization term
  nom = np.prod(np.power(d,1/len(d)))
  # P is inverse to the distance to lda desicion boundry
  P = nom/d
  P_mat = np.diag(P)

  return P_mat

def Q_matix(X,clf,k=1):

  d = distance_to_hyperplane(X,clf)
  d = np.power(d,k) 
  nom = np.prod(np.power(d,1/len(d)))
  # Q is proporcional to the distance to lda desicion boundry
  Q = d/nom
  Q_mat = np.diag(Q)

  return Q_mat

def Proba_matix(X,clf,k=1):

  d = clf.predict_proba(X) 
  d = np.abs( d - np.mean(d)* np.ones(d.shape))
  d = d.sum(axis=1)

  d = np.power(d,k) 
  d = d + 1e-10
  nom = np.prod(np.power(d,1/len(d)))
  # Q is proporcional to the distance to lda desicion boundry
  Q = d/nom
  Proba_mat = np.diag(Q)

  return Proba_mat

def distance_to_mean(X,Y):
  n_classes = np.unique(Y)
  cl = 0
  X_mean = X.copy()
  for classes in n_classes:

    # Keep data for one class
    X_c = X[Y==classes]
    # Compute the mean for each feature
    X_c = np.mean(X_c,axis=0)

    ind_cl = np.where(Y==classes)
    
    X_mean[ind_cl] = X[ind_cl]-X_c
    
  dist = np.sqrt(np.sum(np.power(X_mean,2),axis=1))

  return dist

def balance_weights(y, w,relevance = []):
    # Get classes for labels
    classes = np.unique(y)

    if relevance == []:
      # Uniform relevance for each class
      relevance = 1/len(classes)*np.ones(classes.shape)
    elif sum(relevance)==1:
      raise Exception("Relevance vector needs to sum = 1")
    elif relevance.shape == classes.shape:
      raise Exception("Relevance vector needs to have the same shape as the number of classes")
      
    # Initialize the new weigths
    w_final = np.zeros(w.shape)
    
    for cl in classes:
      rel = relevance[classes==cl]
      # total points 
      w_cl = w.copy()
        
      # keep only the points for one class
      w_cl[y!=cl] = 0
      
      if sum(w_cl) == 0:
        w_cl =  np.ones((len(w_cl),)) / len(w_cl)

      # Normalilize the weights with respect to the relevance of the class and the total mass of the class
      w_cl = w_cl / ( sum(w_cl) * rel )
      
      # Sum the weight of the particular class to the final weight vector
      w_final = w_final + w_cl
    return w_final

def deal_with_wrong_classified_point(Xt,yt,clf, clf_retrain, wrong_cls, ot_method):
  # Delete the wrong classified point from the analysis and
  if clf_retrain:
    clf.fit(Xt,yt)
    Xt, yt = delete_wrong_classified(clf,Xt,yt)
    clf.fit(Xt,yt)
    
  # Start with Source uniform weights
  a = np.ones((len(Xs),)) / len(Xs)

  # Start with Target uniform weights
  b =  np.ones((len(Xt),)) / len(Xt)

  # Do not assign mass to missclassified points
  if wrong_cls:
    Y_pred = clf.predict(Xt)
    b[yt!=Y_pred] = 0
    
    # If all the datapoins for one classe were missclassified
    if sum(b)==np.NaN:
      # Target uniform weights
      b =  np.ones((len(Xt),)) / len(Xt)

    # Replace mass=0 for Grup Lasso method
    if ot_method == "sinkhorn_gl" or ot_method == "s_gl":
        # Not suport mass==0
        b[yt!=Y_pred] = 1e-10

  return Xt, yt, clf, a, b

def compute_penalization(Xt,clf,b, k,penalized):
    # Calculate the distance of each point to lda decision straight
  if penalized == "distance" or penalized == "d":
      Q = Q_matix(Xt, clf, k=k)
      
  elif penalized == "proba" or penalized == "p":
      Q = Proba_matix(Xt, clf,k=k)
  else:
      raise Exception("Penalized not supported")
  
  # change the point weight proportionaly to the computed score
  b = np.dot(b,Q)

  return b

def compute_balance_weights(a,b,ys,yt,balanced_source,balanced_target):
  # Balance target
  # Chek if empy or False
  if not(balanced_target):
    # If empty use uniform relevance for each class
    if balanced_target==[]:
      b_final = balance_weights(yt, b, balanced_target)
    # If False uniform balance
    else:
      b_final = b/sum(b)
  # If not False or Empty check if the first element is int or float
  elif isinstance(balanced_target[0],(int,float)):
      b_final = balance_weights(yt, b, balanced_target)
  else:
    raise Exception("Balance target not supported")

  # Balance source
  # Chek if empy or False
  if not(balanced_source):
    # If empty use uniform relevance for each class
    if balanced_source==[]:
      a_final = balance_weights(ys, a, balanced_source)
    # If False not normalization
    else:
      a_final = a
  # If not False or Empty check if the first element is int or float
  elif isinstance(balanced_source[0],(int,float)):
      a_final = balance_weights(ys, a, balanced_source)
  else:
    raise Exception("Balance source not supported")

  return a_final , b_final 


def compute_coupling(ot_method, a_final, b_final, M, Xs, Xt, reg_e, eta):

  # Compute coupling
  if ot_method == "emd":
    G0 = ot.lp.emd(a_final, b_final, M)

  elif ot_method == "emd2":
    G0 = ot.lp.emd2(a_final, b_final, M, return_matrix =True)
    G0 = G0[1]['G']
    
  elif ot_method == "sinkhorn" or ot_method == "s":
    G0 = ot.sinkhorn(a=a_final, b=b_final, M=M, reg=reg_e)

  elif ot_method == "sinkhorn_gl" or ot_method == "s_gl":
    G0 = ot.da.sinkhorn_l1l2_gl(a_final, ys, b_final, M, reg=reg_e, eta= eta)
  
  elif ot_method == "emd_laplace" or ot_method == "emd_l":
    G0 = ot.da.emd_laplace(a_final,b_final,Xs, Xt, M, eta=eta)
  else:
      raise Exception("OT method not supported")
  return G0

def penalized_coupling(Xs, ys, Xt, yt,  clf, k=1, metric="euclidean", penalized="p", ot_method="emd", wrong_cls=True, balanced_target=[], balanced_source=[], clf_retrain=False, reg_e=1, eta = 0.1):
  """
    Returns
    -------
    G0 :  ((ns x nt) ndarray) – Optimal transportation matrix for the given parameters
    """
  # Data sanity check
  if Xs.shape[0]!=ys.shape[0]:
    raise Exception("Missmach source samples")

  if Xt.shape[0]!=yt.shape[0]:
    raise Exception("Missmach target samples")

  # Deal with wrong classified point in the target domain
  Xt, yt, clf, a, b =  deal_with_wrong_classified_point(Xt, yt, clf, clf_retrain, wrong_cls, ot_method)

  # Change the weights of the target points with respect a penalization
  b = compute_penalization(Xt, clf, b, k, penalized)
 
  # Balance weights
  a , b = compute_balance_weights(a, b, ys, yt, balanced_source, balanced_target)

  # Compute cost matrix
  M = ot.dist(Xs, Xt, metric = metric)

  # Compute coupling with different OT methods
  G0 = compute_coupling(ot_method, a, b, M, Xs, Xt, reg_e, eta)

  if clf_retrain:
    return G0, clf
  
  else:
    return G0

def delete_wrong_classified(clf,X,Y):
  # Make a prediction
  Y_pred = clf.predict(X)
  # Delete the data missclassified
  X = X[Y==Y_pred]
  Y = Y[Y==Y_pred]

  return X, Y


def best_k(ot_emd,Xs, ys, Xt, yt, clf, search=23, metric="euclidean", penalized="p", ot_method="emd", wrong_cls=True, balanced_target=[], balanced_source=[], clf_retrain=False, reg_e=1, eta = 0.1):
  # Initialization
  acc = 0

  for k in range(search):
    # Compute the new coupling with the penalized cost matrix
    G01 = penalized_coupling(Xs, ys, Xt, yt,  clf, k=k, metric=metric,penalized=penalized, ot_method=ot_method, wrong_cls=wrong_cls, balanced_target=balanced_target, balanced_source=balanced_source, clf_retrain=clf_retrain, reg_e=reg_e, eta = eta)

    # Replace the coupling
    ot_emd.coupling_ = G01

    # Transport source samples onto target samples
    T_Xs = ot_emd.transform(Xs=Xs)
    
    # Compute accuracy
    acc = np.append(acc,clf.score(T_Xs,ys))
  
  # Delete initialization
  acc = np.delete(acc,0)
  # Get the max accuracy
  acc_max = np.max(acc)
  # get the k where the acc was max
  acc_max_pos = np.where(acc==acc_max)
  # get the min position
  pos = np.min(acc_max_pos)

  return pos

