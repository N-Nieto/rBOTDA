# -*- coding: utf-8 -*-
"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar
"""

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

def balance_weights(y, w):
    # Get classes for labels
    classes = np.unique(y)

    # Initialize the new weigths
    w_f = np.zeros(w.shape)
    
    for cl in classes:
      
      # total points 
      w_cl = w.copy()
        
      # keep only the points for one class
      w_cl[y!=cl] = 0
      
      if sum(w_cl) == 0:
        w_cl =  np.ones((len(w_cl),)) / len(w_cl)


      # Change the weight of the points for one class
      w_cl = w_cl / ( sum(w_cl) * len(classes) )
      
      # Sum the weight of the particular class to the final weight vector
      w_f = w_f + w_cl
    return w_f


def penalized_coupling(Xs, Xt, yt, ys, clf, k=1, metric="euclidean",penalized="distance", ot_method="emd", wrong_cls=True, balanced_target=True, balanced_source=True, reg_e=1, eta = 0.1):
  """
    Returns
    -------
    G0 :  ((ns x nt) ndarray) – Optimal transportation matrix for the given parameters
    """

  if Xs.shape[0]!=ys.shape[0]:
    print("Missmach source samples")

  if Xt.shape[0]!=yt.shape[0]:
    print("Missmach target samples")


  # Source uniform weights
  a = np.ones((len(Xs),)) / len(Xs)

  # Target uniform weights
  b =  np.ones((len(Xt),)) / len(Xt)

  if wrong_cls:
    # do not assign mass to missclassified points
    Y_pred = clf.predict(Xt)
    b[yt!=Y_pred] = 0
    
    if sum(b)==np.NaN:
      # Target uniform weights
      b =  np.ones((len(Xt),)) / len(Xt)

    if ot_method == "sinkhorn_gl" or ot_method == "s_gl":
        # Not suport mass==0
        b[yt!=Y_pred] = 1e-10
    
  # Calculate the distance of each point to lda decision straight
  if penalized == "distance" or penalized == "d":
      Q = Q_matix(Xt, clf, k=k)
      
  elif penalized == "proba" or penalized == "p":
      Q = Proba_matix(Xt, clf,k=k)
  else:
      Warning("Penalized not supported")
  
  # change the point weight proportionaly to its lda hiperplane
  b = np.dot(b,Q)
 
  # Balance target weights
  if balanced_target:
    b_final = balance_weights(yt, b)
  else:
    # Uniform normalization
    b_final = b/sum(b)

  # Balance source weights
  if balanced_source:
    a_final = balance_weights(ys, a)
  else:
    # Uniform normalization
    a_final = a

  # Compute cost matrix
  M = ot.dist(Xs, Xt, metric = metric)

  ##### Compute coupling
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

  return G0


def delete_wrong_classified (lda,X,Y):
  Y_pred = lda.predict(X)
  X = X[Y==Y_pred]
  Y = Y[Y==Y_pred]
  return X, Y


def best_k(ot_emd,clf,Xs,Xt,yt,ys,penalized,search=23):

  acc = 0
  for k in range(search):

    # Compute the new coupling with the penalized cost matrix
    G01 = penalized_coupling(Xs,Xt,yt,ys,clf=clf,k=k,penalized=penalized)

    # Replace the coupling
    ot_emd.coupling_ = G01

    # transport source samples onto target samples
    T_Xs = ot_emd.transform(Xs=Xs)

    acc = np.append(acc,clf.score(T_Xs,ys))

  acc = np.delete(acc,0)
  acc_max = np.max(acc)
  # get the number of nodes where the acc was max
  acc_max_pos = np.where(acc==acc_max)
  # get the min position
  pos = np.min(acc_max_pos)
  return pos

