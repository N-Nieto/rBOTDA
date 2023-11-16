# %%
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from lib.utilitys import split_synthetic_data_unbalanced, load_synthetic_data  # noqa
from lib.rBOTDA import rBOTDA
random_state = 23
np.random.seed(random_state)

# %%
# #### Variables
X, Y = load_synthetic_data(unrepresentative_features=0)
print(X.shape)
# OT variables
# Classifier is trained on SOURCE, and learn to transpor from target to source
Balance_train = [400, 200]  # Max 100 trials for each class
Balance_val = [100, 200]  # Max 100 trials for each class
Balance_test = [100, 100]    # Max 100 trials for each class
# OT method
ot_method = "emd"

# # Cross-Validation
n_KF = 10

# Classifier
clf = LinearDiscriminantAnalysis()

# %%
# Greate a list to append results
result_list = []

# Simple Cross validation
for kv in range(n_KF):

    X_tr, X_test_whole, Y_tr, Y_test_whole = train_test_split(X, Y,
                                                              test_size=0.4,
                                                              shuffle=True,
                                                              random_state=kv)

    # Costume function to unbalance the for each class
    X_train, Y_train, X_val_not_used, Y_val_not_used = split_synthetic_data_unbalanced(     # noqa
                                                    X_tr, Y_tr,
                                                    Balance_train,
                                                    shuffle=True)

    X_val, Y_val, X_test, Y_test = split_synthetic_data_unbalanced(
                                                    X_test_whole,
                                                    Y_test_whole,
                                                    Balance_val,
                                                    shuffle=True)

    # ### Clasification
    # Train clasiffier
    clf.fit(X_train, Y_train)
    # Compute acc
    performance = clf.score(X_test, Y_test)
    # Append the result in the list
    result_list.append(["NO_OT", performance])

    # BOTDA
    rbotda = rBOTDA(k=0, ot_method=ot_method, wrong_cls=False,
                    balanced_train=None, balanced_val=None,
                    cost_supervised=True)

    rbotda.fit(X_train=X_train, X_val=X_val,
               y_train=Y_train, y_val=Y_val, clf=clf)

    X_test_transform = rbotda.transform(X=X_test)
    performance = clf.score(X_test_transform, Y_test)
    result_list.append(["BOTDA", performance])

    # rBOTDA
    rbotda = rBOTDA(k=1, ot_method=ot_method, wrong_cls=True,
                    balanced_train="auto", balanced_val="auto",
                    cost_supervised=True)

    rbotda.fit(X_train=X_train, X_val=X_val,
               y_train=Y_train, y_val=Y_val, clf=clf)

    X_test_transform = rbotda.transform(X=X_test)
    performance = clf.score(X_test_transform, Y_test)
    result_list.append(["rBOTDA", performance])

# %% Plots
_, ax = plt.subplots(1, 1, figsize=[10, 8])
results = pd.DataFrame(result_list, columns=["method", "ACC"])

sns.boxplot(data=results, x="method", y="ACC", ax=ax)
plt.grid(color="black", alpha=0.8, axis="y")

# %%
