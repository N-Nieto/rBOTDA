
# %%
# -*- coding: utf-8 -*-
"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar
"""
from rbotda.ot_helper_functions import initialize_ot_obj
from rbotda.ot_helper_functions import compute_cost_matrix, compute_coupling
from rbotda.balancing_functions import deal_with_wrong_classified_point
from rbotda.balancing_functions import compute_balance_weights, initialize_sample_weights       # noqa
from rbotda.penalization_functions import compute_penalization


def naming_check(ot_method, metric, penalized_type):

    if ot_method not in ["emd", "sinkhorn", "s", "sinkhorn_gl", "s_gl",
                         "emd_laplace", "emd_l"]:
        RuntimeError("Invalid OT method")
    if metric not in ["euclidean"]:
        RuntimeError("Invalid metric")
    if penalized_type not in ["distance", "d", "probability", "proba", "p"]:
        RuntimeError("Penalized type")

    return


def data_check(Xs, Xt, ys=[], yt=[]):
    # Check consistensy if ys is provided
    if (ys is not None):
        # Data sanity check
        if Xs.shape[0] != ys.shape[0]:
            raise RuntimeError("Missmach source samples")
    # Check consistensy if yt is provided
    if (yt is not None):
        if Xt.shape[0] != yt.shape[0]:
            raise RuntimeError("Missmach target samples")
    return


class rBOTDA():
    def __init__(self, ot_method="emd", k=5, metric="euclidean",
                 penalized_type="p", wrong_cls=True,
                 balanced_target=[], balanced_source=[],
                 reg_e=1, eta=0.1,
                 cost_norm=None, limit_max=10,
                 cost_supervised=True,
                 coupling_supervised=True):

        # check naming
        naming_check(ot_method=ot_method, metric=metric,
                     penalized_type=penalized_type)

        # Initialize any hyperparameters for OT
        # Optimal Transport Method. EMD, S, GL
        self.ot_method = ot_method
        # Distance metric to compute transport plan
        self.metric = metric
        # Parameter for regularized version of OT
        self.reg_e = reg_e
        self.eta = eta
        # Cost norm and limit max of OT
        self.cost_norm = cost_norm
        self.limit_max = limit_max
        self.cost_supervised = cost_supervised
        self.coupling_supervised = coupling_supervised
        # Initialize any hyperparameters for penalization
        # Type of penalization (Distance or classifier probability)
        self.penalized_type = penalized_type
        # Penalization intensity. Is and hyperparameter to tune
        self.k = k
        # If removing points wrong classified before applied transport.
        # Poinsts will be assigned with 0 mass.
        self.wrong_cls = wrong_cls

        # Initialize any hyperparameters for balance

        # Balancing the target and source samples.
        # if "auto" is passed, the target mass will be normalized
        # so the sum of all point's mass sum 1/n_classes.
        # A vector with different proportions can be also passed as parameter.
        # The vector muss sum one, and then the points for the corresponding
        # class will sum up until the proportion passed.
        self.balanced_target = balanced_target
        self.balanced_source = balanced_source

        initialize_ot_obj(self)

    def fit_tl(self, Xs, Xt, clf, yt=None):
        # Fit optimal transport method with

        """
        Returns
        -------
        ot_obj :  Optimal transport instance
        """
        # Check consistency
        data_check(Xs=Xs, Xt=Xt, ys=None, yt=yt)

        a, b = initialize_sample_weights(Xt, Xs)
        # If target (train), labels are provided, then enter to the function
        if (yt is not None):
            # Deal with wrong classified point in the target domain
            Xt, yt, a, b = deal_with_wrong_classified_point(self, a, b,
                                                            Xt, yt, clf)

        # Fit the object to the data
        self.ot_obj = self.ot_obj.fit(Xs=Xs, Xt=Xt, yt=yt)

        # Change the weights of the target points with respect a penalization
        b = compute_penalization(self, Xt, clf, b)

        # Balance weights for Source and Target
        a, b = compute_balance_weights(self, a, b, yt, ys=None)

        # Compute cost matrix
        M = compute_cost_matrix(self, Xs=Xs, Xt=Xt, yt=yt, ys=None)

        # Compute coupling with different OT methods
        G0 = compute_coupling(self, a, b, M, Xs, Xt, None)

        # Replace the coupling with the penalized one
        self.ot_obj.coupling_ = G0
        self.ot_obj.mu_t = b
        self.ot_obj.mu_s = a
        self.ot_obj.cost_ = M

        return self

    def fit_tl_supervised(self, Xs, Xt, clf, ys=None, yt=None):
        # Fit optimal transport method with

        """
        Returns
        -------
        ot_obj :  Optimal transport instance
        """
        # Check consistency
        data_check(Xs=Xs, Xt=Xt, ys=ys, yt=yt)

        a, b = initialize_sample_weights(Xt, Xs)
        # If target (train), labels are provided, then enter to the function
        if (yt is not None):
            # Deal with wrong classified point in the target domain
            Xt, yt, a, b = deal_with_wrong_classified_point(self, a, b,
                                                            Xt, yt, clf)

        # Fit the object to the data
        self.ot_obj = self.ot_obj.fit(Xs=Xs, ys=ys, Xt=Xt, yt=yt)

        # Change the weights of the target points with respect a penalization
        b = compute_penalization(self, Xt, clf, b)

        # Balance weights for Source and Target
        a, b = compute_balance_weights(self, a, b, ys, yt)
        # Compute cost matrix
        M = compute_cost_matrix(self, Xs=Xs, ys=ys, Xt=Xt, yt=yt)

        # Compute coupling with different OT methods
        G0 = compute_coupling(self, a, b, M, Xs, Xt, ys)

        # Replace the coupling with the penalized one
        self.ot_obj.coupling_ = G0
        self.ot_obj.mu_t = b
        self.ot_obj.mu_s = a
        self.ot_obj.cost_ = M

        return self

    def transform(self, Xs):
        # Transform data using the fitted
        Xs_transform = self.ot_obj.transform(Xs=Xs)
        return Xs_transform

# %%
