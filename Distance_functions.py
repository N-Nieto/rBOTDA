
# %%
# -*- coding: utf-8 -*-
"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar
"""
import numpy as np
import ot.plot


# Distance Functions
def distance_to_hyperplane(X, clf):
    b = clf.intercept_
    W = clf.coef_
    # Module
    mod = np.sqrt(np.sum(np.power(W, 2)))
    # distance
    d = np.abs(np.dot(X, W.T)+b)/mod
    return d[:, 0]


def P_matix(X, clf, k):
    # Compute distance
    d = distance_to_hyperplane(X, clf)
    d = np.power(d, k)

    # Normalization term
    nom = np.prod(np.power(d, 1/len(d)))
    # P is inverse to the distance to lda desicion boundry
    P = nom/d
    P_mat = np.diag(P)

    return P_mat


def Q_matix(X, clf, k):
    d = distance_to_hyperplane(X, clf)
    d = np.power(d, k)
    nom = np.prod(np.power(d, 1/len(d)))
    # Q is proporcional to the distance to lda desicion boundry
    Q = d/nom
    Q_mat = np.diag(Q)

    return Q_mat


def Proba_matix(X, clf, k):
    # Get the Proba for each point
    d = clf.predict_proba(X)
    # Substract the mean for each point and compute the absolute value
    d = np.abs(d - np.mean(d)*np.ones(d.shape))
    d = d.sum(axis=1)
    d = np.power(d, k)
    d = d + 1e-10
    nom = np.prod(np.power(d, 1/len(d)))
    # Q is proporcional to the distance to lda desicion boundry
    Q = d/nom
    Proba_mat = np.diag(Q)

    return Proba_mat


def distance_to_mean(X, Y):
    n_classes = np.unique(Y)
    X_mean = X.copy()

    for classes in n_classes:
        # Keep data for one class
        X_c = X[Y == classes]
        # Compute the mean for each feature
        X_c = np.mean(X_c, axis=0)

        ind_cl = np.where(Y == classes)
        X_mean[ind_cl] = X[ind_cl]-X_c

    dist = np.sqrt(np.sum(np.power(X_mean, 2), axis=1))

    return dist


def balance_weights(y, w, relevance=[]):
    # Get classes for labels
    classes = np.unique(y)

    if relevance == []:
        # Uniform relevance for each class
        relevance = 1/len(classes)*np.ones(classes.shape)
    elif not (sum(relevance) == 1):
        raise Exception("Relevance vector needs to sum = 1")

    # Initialize the new weigths
    w_final = np.zeros(w.shape)
    relevance = np.array(relevance)

    for cl in classes:
        mask = np.array(classes == cl)
        rel = relevance[mask]
        # total points
        w_cl = w.copy()
        # keep only the points for one class
        w_cl[y != cl] = 0

        # In the case that all points in one class have 0 weight
        # (i.e. if all were missclassified)
        if sum(w_cl) == 0:
            w_cl = np.ones((len(w_cl),)) / len(w_cl)

        # Normalilize the weights with respect to the relevance of
        # the class and the total mass of the class
        w_cl = w_cl * rel / (sum(w_cl))

        # Sum the weight of the particular class to the final weight vector
        w_final = w_final + w_cl

    return w_final


def deal_with_wrong_classified_point(Xt, yt, Xs, clf,
                                     clf_retrain, wrong_cls, ot_method):
    # Initialize classifier
    clf.fit(Xt, yt)
    # Delete the wrong classified point from the analysis
    if clf_retrain:
        Xt, yt = delete_wrong_classified(clf, Xt, yt)
        clf.fit(Xt, yt)

    # Start with Source uniform weights
    a = np.ones((Xs.shape[0]),) / (Xs.shape[0])
    # Start with Target uniform weights
    b = np.ones(((Xt.shape[0]),)) / (Xt.shape[0])

    # Do not assign mass to missclassified points
    if wrong_cls:
        Y_pred = clf.predict(Xt)
        b[yt != Y_pred] = 0

        # If all the datapoins for one classe were missclassified
        if sum(b) == np.NaN:
            # Target uniform weights
            b = np.ones(((Xt.shape[0]),)) / (Xt.shape[0])

        # Replace mass=0 for Grup Lasso method
        if ot_method == "sinkhorn_gl" or ot_method == "s_gl":
            # Not suport mass==0
            b[yt != Y_pred] = 1e-10

    return Xt, yt, clf, a, b


def compute_penalization(Xt, clf, b, k, penalized):
    # Calculate the distance of each point to lda decision straight
    if penalized == "distance" or penalized == "d":
        Q = Q_matix(X=Xt, clf=clf, k=k)
    elif penalized == "proba" or penalized == "p":
        Q = Proba_matix(X=Xt, clf=clf, k=k)
    else:
        raise Exception("Penalized not supported")

    # change the point weight proportionaly to the computed score
    b = np.dot(b, Q)

    return b


def compute_balance_weights(a, b, ys, yt,
                            balanced_source, balanced_target, k):
    # Balance target
    # Chek if empy or False
    if not (balanced_target):
        # If empty use uniform relevance for each class
        if balanced_target == []:
            b_final = balance_weights(yt, b, balanced_target)
        # If False uniform balance
        elif k > 0:
            b_final = b/sum(b)
        elif k == 0:
            b_final = b
    # If not False or Empty check if the first element is int or float
    elif isinstance(balanced_target[0], (int, float)):
        b_final = balance_weights(yt, b, balanced_target)
    else:
        raise Exception("Balance target not supported")

    # Balance source
    # Chek if empy or False
    if not (balanced_source):
        # If empty use uniform relevance for each class
        if balanced_source == []:
            a_final = balance_weights(ys, a, balanced_source)
        # If False not normalization
        else:
            a_final = a
    # If not False or Empty check if the first element is int or float
    elif isinstance(balanced_source[0], (int, float)):
        a_final = balance_weights(ys, a, balanced_source)
    else:
        raise Exception("Balance source not supported")

    return a_final, b_final


def compute_coupling(ot_method, a, b, M, Xs, Xt,
                     reg_e, eta, ys=[]):
    # Compute coupling
    if ot_method == "emd":
        G0 = ot.da.emd(a=a, b=b, M=M)

    elif ot_method == "sinkhorn" or ot_method == "s":
        G0 = ot.da.sinkhorn_lpl1_mm(a=a, labels_a=ys, b=b, M=M, reg=reg_e)

    elif ot_method == "sinkhorn_gl" or ot_method == "s_gl":
        G0 = ot.da.sinkhorn_l1l2_gl(a=a, labels_a=ys, b=b, M=M,
                                    reg=reg_e, eta=eta)

    elif ot_method == "emd_laplace" or ot_method == "emd_l":
        G0 = ot.da.emd_laplace(a=a, b=b, Xs=Xs, Xt=Xt, M=M, eta=eta)
    else:
        raise RuntimeError("OT method not supported")
    return G0


def delete_wrong_classified(clf, X, Y):
    # Make a prediction
    Y_pred = clf.predict(X)

    # Check if we do not delete all points for one class
    Y_test = Y[Y == Y_pred]

    if len(np.unique(Y_test)) < 2:
        Warning("Retrain not possible")
    else:
        # Delete the data missclassified
        X = X[Y == Y_pred]
        Y = Y[Y == Y_pred]

    return X, Y


def penalized_coupling(Xs, ys, Xt, yt, clf, k=-10, metric="euclidean",
                       penalized="p", ot_method="emd", wrong_cls=True,
                       balanced_target=[], balanced_source=[],
                       clf_retrain=False, reg_e=1, eta=0.1,
                       cost_norm=None, limit_max=10):
    """
      Returns
      -------
      ot_obj :  Optimal transport instance
      """
    # Data sanity check
    if Xs.shape[0] != ys.shape[0]:
        raise RuntimeError("Missmach source samples")

    if Xt.shape[0] != yt.shape[0]:
        raise RuntimeError("Missmach target samples")

    # Deal with wrong classified point in the target domain
    Xt, yt, clf, a, b = deal_with_wrong_classified_point(Xt, yt, Xs, clf,
                                                         clf_retrain,
                                                         wrong_cls, ot_method)

    # Create OT object
    ot_obj = initialize_ot_obj(ot_method, metric, reg_e, eta)
    # Fit the object to the data
    ot_obj = ot_obj.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)

    # Pass a negative value for search the best k until the passed value
    if k < 0:
        k = best_k(ot_obj, Xs, ys, Xt, yt, clf, -k, metric, penalized,
                   ot_method, wrong_cls, balanced_target, balanced_source,
                   clf_retrain, reg_e, eta)
        # Change the weights of the target points with respect a penalization
        b = compute_penalization(Xt, clf, b, k, penalized)
    else:
        # Change the weights of the target points with respect a penalization
        b = compute_penalization(Xt, clf, b, k, penalized)

    # Balance weights
    a, b = compute_balance_weights(a, b, ys, yt,
                                   balanced_source, balanced_target, k)
    # Compute cost matrix
    M = compute_cost_matrix(Xs=Xs, ys=ys, Xt=Xt, yt=yt, metric=metric,
                            cost_norm=cost_norm, limit_max=limit_max)

    # Compute coupling with different OT methods
    G0 = compute_coupling(ot_method, a, b, M, Xs, Xt, reg_e, eta)

    # Replace the coupling with the penalized one
    ot_obj.coupling_ = G0
    ot_obj.mu_t = b
    ot_obj.mu_s = a
    ot_obj.cost_ = M

    return ot_obj, clf, k


def initialize_ot_obj(ot_method, metric, reg_e, eta):

    # Compute coupling
    if ot_method == "emd":
        ot_obj = ot.da.EMDTransport(metric=metric)

    elif ot_method == "sinkhorn" or ot_method == "s":
        ot_obj = ot.da.SinkhornTransport(metric=metric, reg_e=reg_e)

    elif ot_method == "sinkhorn_gl" or ot_method == "s_gl":
        ot_obj = ot.da.SinkhornL1l2Transport(metric=metric, reg_e=reg_e,
                                             reg_cl=eta)

    elif ot_method == "emd_laplace" or ot_method == "emd_l":
        ot_obj = ot.da.EMDLaplaceTransport(metric=metric)
    else:
        raise RuntimeError("OT method not supported")

    return ot_obj


def best_k(ot_obj, Xs, ys, Xt, yt, clf, search, metric, penalized, ot_method,
           wrong_cls, balanced_target, balanced_source, clf_retrain, reg_e,
           eta):

    # Initialization
    acc = 0
    for k in range(search):
        # Compute the new coupling with the penalized cost matrix
        ot_obj, clf_rt, k = penalized_coupling(Xs, ys, Xt, yt,  clf, k, metric,
                                               penalized, ot_method, wrong_cls,
                                               balanced_target,
                                               balanced_source,
                                               clf_retrain, reg_e, eta)

        # Transport source samples onto target samples
        T_Xs = ot_obj.transform(Xs=Xs)

        # Compute accuracy
        acc = np.append(acc, clf_rt.score(T_Xs, ys))

    # Delete initialization
    acc = np.delete(acc, 0)
    # Get the max accuracy
    acc_max = np.max(acc)
    # get the k where the acc was max
    acc_max_pos = np.where(acc == acc_max)
    # get the min position
    pos = np.min(acc_max_pos)

    return pos


def compute_cost_matrix(Xs, ys, Xt, yt, metric, cost_norm, limit_max):
    # pairwise distance
    M = ot.dist(Xs, Xt, metric=metric)
    M = ot.utils.cost_normalization(M, cost_norm)

    if (ys is not None) and (yt is not None):

        if limit_max != np.infty:
            limit_max = limit_max * np.max(M)

        # assumes labeled source samples occupy the first rows
        # and labeled target samples occupy the first columns
        classes = [c for c in np.unique(ys) if c != -1]
        for c in classes:
            idx_s = np.where((ys != c) & (ys != -1))
            idx_t = np.where(yt == c)

            # all the coefficients corresponding to a source sample
            # and a target sample :
            # with different labels get a infinite
            for j in idx_t[0]:
                M[idx_s[0], j] = limit_max

    return M
