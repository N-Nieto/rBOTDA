# %%
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
random_state = 23
np.random.seed(random_state)

from rbotda.Utilitys import split_data_unbalanced  # noqa
from rBOTDA import rBOTDA                                   # noqa
# %%
# Drive direction
root_dir = "/home/nnieto/Nico/OTclf_info/data/synthetic_data/"

data = sio.loadmat(file_name=root_dir + "Synthetic_data.mat")

X_data = data['X']
X_data = X_data[:, 0:2]

Y_data = data['Y']
# Labels are in -1 +1, change it to 0 +1
Y_data = ((Y_data + 1) / 2)+1
# %%
# #### Variables

# OT variables
# Classifier is trained on TARGET, and learn to transpor from target to source
Balance_test = [500, 500]    # Max 100 trials for each class
Balance_source = [200, 200]  # Max 100 trials for each class
Balance_target = [300, 300]  # Max 100 trials for each class
# Which distance is used in the OT
metric = "euclidean"
# Type of penaliation
penalized = "d"     # d = distance / p = proba
# OT method
ot_method = "emd"
# Penalization strength
k = 2


# # Cross-Validation
KF = 20
# Classifier
clf = LinearDiscriminantAnalysis()

# Plot Settings
save_folder = "OT+LDA info/Images/Experimento_mismo_N/"
save_dir = root_dir+save_folder
prefix_name = "T_Balance_"+str(Balance_target[0]/Balance_target[1])+"_S_Balance_"+str(Balance_source[0]/Balance_source[1]) # noqa
save_fig = False
# %%
# Greate a list to append results
result_list = []

# Simple Cross validation
for kv in range(KF):

    X_tr, X_test, Y_tr, Y_test = train_test_split(X_data, Y_data,
                                                  test_size=0.3,
                                                  shuffle=True,
                                                  random_state=kv)

    Y_tr = np.squeeze(Y_tr)
    Y_test = np.squeeze(Y_test)

    # ### Split data
    X_trn, Y_trn, X_val_not_used, Y_val_not_used = split_data_unbalanced(
                                                    X_tr, Y_tr,
                                                    Balance_target,
                                                    shuffle=True)

    X_val, Y_val, X_not_used, Y_not_used = split_data_unbalanced(
                                                    X_test,
                                                    Y_test,
                                                    Balance_source,
                                                    shuffle=True)

    # Set Target and Source domains
    Xs, ys = X_val, Y_val
    Xt, yt = X_trn, Y_trn

    # ### Clasification
    # Train clasiffier
    clf.fit(Xt, yt)
    # Compute acc
    performance = clf.score(X_test, Y_test)
    # Append the result in the list
    result_list.append(["NO_OT", performance])

    # Traditional Transport, nothhing supervised
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=False,
                    balanced_source=False, balanced_target=False,
                    cost_supervised=False)

    rbotda.fit_tl_supervised(Xs=Xs, Xt=Xt, ys=ys, yt=yt, clf=clf)
    Xs_transform = rbotda.transform(Xs=X_test)
    performance = clf.score(Xs_transform, Y_test)
    result_list.append(["OT_cost_no_sup", performance])

    # Traditional Transport cost supervised
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=False,
                    balanced_source=False, balanced_target=False,
                    cost_supervised=True)

    rbotda.fit_tl_supervised(Xs=Xs, Xt=Xt, ys=ys, yt=yt, clf=clf)
    Xs_transform = rbotda.transform(Xs=X_test)
    performance = clf.score(Xs_transform, Y_test)
    result_list.append(["OT_cost_sup", performance])

    # rBOTDA, no supervised
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=True,
                    balanced_source=False, balanced_target=False,
                    cost_supervised=False)

    rbotda.fit_tl_supervised(Xs=Xs, Xt=Xt, ys=ys, yt=yt, clf=clf)
    Xs_transform = rbotda.transform(Xs=X_test)
    performance = clf.score(Xs_transform, Y_test)
    result_list.append(["rOT_cost_no_sup", performance])

    # rBOTDA, cost supervised
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=True,
                    balanced_source=False, balanced_target=False,
                    cost_supervised=True)

    rbotda.fit_tl_supervised(Xs=Xs, Xt=Xt, ys=ys, yt=yt, clf=clf)
    Xs_transform = rbotda.transform(Xs=X_test)
    performance = clf.score(Xs_transform, Y_test)
    result_list.append(["rOT_cost_sup", performance])

    # rBOTDA, no supervised
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=True,
                    balanced_source=[], balanced_target=[],
                    cost_supervised=False)

    rbotda.fit_tl_supervised(Xs=Xs, Xt=Xt, ys=ys, yt=yt, clf=clf)
    Xs_transform = rbotda.transform(Xs=X_test)
    performance = clf.score(Xs_transform, Y_test)
    result_list.append(["rOT_all_balanced_cost_no_sup", performance])

    # rBOTDA, cost supervised
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=True,
                    balanced_source=[], balanced_target=[],
                    cost_supervised=True)

    rbotda.fit_tl_supervised(Xs=Xs, Xt=Xt, ys=ys, yt=yt, clf=clf)
    Xs_transform = rbotda.transform(Xs=X_test)
    performance = clf.score(Xs_transform, Y_test)
    result_list.append(["rOT_all_balanced_cost_sup", performance])

    # rBOTDA, cost supervised
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=True,
                    balanced_source=False, balanced_target=[],
                    cost_supervised=True)

    rbotda.fit_tl_supervised(Xs=Xs, Xt=Xt, ys=ys, yt=yt, clf=clf)
    Xs_transform = rbotda.transform(Xs=X_test)
    performance = clf.score(Xs_transform, Y_test)
    result_list.append(["rOT_cost_sup_target_balanced", performance])

    # rBOTDA, cost supervised
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=True,
                    balanced_source=[], balanced_target=False,
                    cost_supervised=True)

    rbotda.fit_tl_supervised(Xs=Xs, Xt=Xt, ys=ys, yt=yt, clf=clf)
    Xs_transform = rbotda.transform(Xs=X_test)
    performance = clf.score(Xs_transform, Y_test)
    result_list.append(["rOT2_cost_sup_source_balanced", performance])
# %%
_, ax = plt.subplots(1, 1, figsize=[30, 8])
results = pd.DataFrame(result_list, columns=["method", "ACC"])


sns.boxplot(data=results, x="method", y="ACC", ax=ax)
plt.grid(color="black", alpha=0.8, axis="y")
# %%

# %%
