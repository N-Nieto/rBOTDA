# %%
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import mne
import scipy.io
import pandas as pd
from mne.decoding import CSP
import sys
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

mne.set_log_level(verbose='warning')   # to avoid info at terminal
random_state = 23
np.random.seed(random_state)

sys.path.append('/home/nnieto/Nico/OTclf_info/')

from Pruebas_Nico_OT.rbotda.Utilitys import split_data_unbalanced # noqa
from rBOTDA import rBOTDA                                   # noqa

# %%
# Drive direction
root_dir = "/home/nnieto/Nico/OTclf_info/data/Datos procesados/"


def filter_data(data):

    [nt, nc, ns] = np.shape(data)
    data = np.reshape(data, [nt, nc*ns])
    filter_data = mne.filter.filter_data(data, 128, 8, 30)
    filter_data = np.reshape(data, [nt, nc, ns])

    return filter_data


def load_data(subject, session):
    if session == 1:
        ses = 'A'
    elif session == 2:
        ses = 'B'
    else:
        print("Valid Session are [1,2]")
    if subject < 10:
        sub = '0'+str(subject)
    else:
        sub = str(subject)

    fName = 'BCIH2015001_S'+str(sub)+ses+'_128.mat'
    s = scipy.io.loadmat(root_dir+fName)
    data = s['X']
    label = s['y'][:, 0]

    return data, label


# TODO: chequear el CV
# #### Variables
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# OT variables
# We train on TARGET, and learn to transpor from target to source
Balance_test = [85, 85]    # Max 100 trials for each class
Balance_source = [10, 10]  # Max 100 trials for each class
Balance_target = [95, 95]  # Max 100 trials for each class
metric = "euclidean"

# # Cross-Validation
KF = 20
# Classifier
clf = LinearDiscriminantAnalysis()
verbose = False
metrica = "acc"
penalized = "p"
search = -10
ot_method = "s_gl"
csp = CSP(n_components=6, reg='empirical', log=True,
          norm_trace=False, cov_est='epoch')

# Plot Settings
save_folder = "OT+LDA info/Images/Experimento_mismo_N/"
save_dir = root_dir+save_folder
prefix_name = "T_Balance_"+str(Balance_target[0]/Balance_target[1]) + \
               "_S_Balance_"+str(Balance_source[0]/Balance_source[1])
save_fig = False
# %%
result_list = []
timing = []

# #### Initializations
for sub in subjects:
    print("################################")
    print("Processing subject: "+str(sub)+" of " + str(len(subjects)))

    # ### Load data
    # Session 1, used as train (Target domain) - calibration
    X_tr, Y_tr = load_data(subject=sub, session=1)
    # Session 2, used as validation/test (Source domain) - re-calibration
    X_test, Y_test_or = load_data(subject=sub, session=2)

    # ### Data processing
    # Learn csp filters
    X_tr = csp.fit_transform(X_tr, Y_tr)

    X_test_or = csp.transform(X_test)

    # Keep test data out of the loop
    X_test, Y_test, X_val_or, Y_val_or = split_data_unbalanced(X_test_or,
                                                               Y_test_or,
                                                               Balance_test,
                                                               shuffle=False)
    for kv in range(KF):
        # ### Split data
        X_trn, Y_trn, X_val_not_used, Y_val_not_used = split_data_unbalanced(
                                                        X_tr, Y_tr,
                                                        Balance_target,
                                                        shuffle=True)

        X_val, Y_val, X_not_used, Y_not_used = split_data_unbalanced(
                                                        X_val_or,
                                                        Y_val_or,
                                                        Balance_source,
                                                        shuffle=True)

        # Set Target and Source domains
        Xs, ys = X_val, Y_val
        Xt, yt = X_trn, Y_trn

        # ### Clasification
        # Train clasiffier
        clf.fit(Xt, yt)
        y_pred = clf.predict(X_test)
        bacc = balanced_accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test, y_pred_proba)
        result_list.append(["NO_OT", bacc, f1, auc, sub, kv])
        # Traditional Transport
        rbotda = rBOTDA(wrong_cls=True, balanced_source=[], balanced_target=[])
        rbotda.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt, clf=clf)
        Xs_transform = rbotda.transform(Xs=X_test)
        y_pred = clf.predict(Xs_transform)
        bacc = balanced_accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        y_pred_proba = clf.predict_proba(Xs_transform)[:, 1]
        auc = roc_auc_score(Y_test, y_pred_proba)
        result_list.append(["OT_balanced", bacc, f1, auc, sub, kv])

        rbotda = rBOTDA(wrong_cls=False, balanced_source=False,
                        balanced_target=False)
        rbotda.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt, clf=clf)
        Xs_transform = rbotda.transform(Xs=X_test)
        y_pred = clf.predict(Xs_transform)
        bacc = balanced_accuracy_score(Y_test, y_pred)
        f1 = f1_score(Y_test, y_pred)
        y_pred_proba = clf.predict_proba(Xs_transform)[:, 1]
        auc = roc_auc_score(Y_test, y_pred_proba)
        result_list.append(["OT", bacc, f1, auc, sub, kv])


# %%
_, ax = plt.subplots(1, 1, figsize=[20, 8])
results = pd.DataFrame(result_list, columns=["method", "bACC", "F1", "AUC",
                                             "Subjects", "Fold"])

df_filtered = results[results['method'].isin(['OT', 'OT_balanced',
                                              'NO_OT'])]

sns.boxplot(data=df_filtered, x="Subjects", y="AUC", ax=ax, hue="method")
plt.grid(color="black", alpha=0.8, axis="y")
# %%

# %%
