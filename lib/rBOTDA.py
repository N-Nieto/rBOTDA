
# %%
# -*- coding: utf-8 -*-
# @author: Nieto Nicolás
# @email: nnieto@sinc.unl.edu.ar

from .ot_helper_functions import initialize_ot_obj
from .ot_helper_functions import compute_cost_matrix, compute_backward_coupling                 # noqa
from .balancing_functions import deal_with_wrong_classified                               # noqa
from .balancing_functions import balance_samples, initialize_uniform_weights            # noqa
from .penalization_functions import compute_penalization
from .utilitys import naming_check, data_consistency_check
from typing import Optional
from numpy.typing import ArrayLike


class rBOTDA():
    """  # noqa
    Regularized Backward optimal Transport (rBOTDA) class

    The regularization version of BOTDA has three main improvements:

    1) The possibility to use the already trained classifier to improve the transport.                          
       The points that were wrongly classified can be removed using "wrong_cls"

       The points that are closer to the classifier decision boundlry are penalized using "k" and "penalized_type"

    2) Possibility to change the mass for each class in both domains.
        Use the parameters "balanced_train" and "balanced_val"
    """
    def __init__(self,
                 k: int,
                 ot_method: str = "emd",
                 metric: str = "euclidean",
                 penalized_type: str = "p",
                 wrong_cls: bool = True,
                 balanced_train="auto",
                 balanced_val="auto",
                 reg: Optional[float] = 1,
                 eta: Optional[float] = 0.1,
                 max_iter: Optional[int] = 10,
                 cost_norm: Optional[bool] = None,
                 limit_max: Optional[int] = 10,
                 cost_supervised: Optional[bool] = True) -> None:
        """ # noqa
        Initialize the rBOTDA object.
        Args:
            k (int): penalization strength. If k=0 no penalization applied

            ot_method (str, optional): Optimal transport method applied. Defaults to "emd".
                Supported: "emd" / Earth Movers distance
                           "s" / Sinkhorn
                           "s_gl" / Sinkhorn Group Lasso
                           "emd_l" / Laplace Earth Movers distance

            metric (str, optional): Distance metric. Defaults to "euclidean".

            From POT:
                'sqeuclidean' or 'euclidean' on all backends.
                On numpy the function also accepts from
                the scipy.spatial.distance.cdist function :
                'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener',
                'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.


            penalized_type (str, optional): Penalize the samples by a metric. Defaults to "p".
                Supported: "p" - Probability. sklearn classifier must have the
                                              "predict_proba" function
                                              Only possible for binary
                                              classification
                           "d" - Hyperplane distance (only linear classifiers)
                                 Classifier must have "intercept_" and
                                 "coef_" attributes

            wrong_cls (bool, optional): Delete points wrong classified on train. Defaults to True.

            balanced_train (Any, optional): Balance the train set. Defaults to "auto".

                Supported: "auto" - Balance the train domain using train labels
                                    The mass of the point for each class must sum 1/Number Classes.

                            None - No balance the train domain. All points will have the same mass

                            [] - List containing the sum of mass for each class
                                 Lenght of list = number of classes
                                 Sum of elements in list must be 1
                                 For example, if [0.6, 0.4] is provided, the mass of the class 1 will sum 0.6
                                 while the ones for class 2 will sum 0.4.

            balanced_val (Any, optional): Balance the validation set. Defaults to "auto".

                Supported: "auto" - Balance the val domain using val labels
                                    The mass of the point for each class must
                                    sum 1/Number Classes.

                            None - No balance the val domain.
                                   All points will have the same mass

                            [] - List containing the sum of mass for each class
                                 Lenght of list = number of classes
                                 Sum of elements in list must be 1
                                 For example, if [0.6, 0.4] is provided, the mass of the class 1 will sum 0.6
                                 while the ones for class 2 will sum 0.4.

            reg (float, optional): Regularization Parameter. Defaults to 1.
                                     Only used when ot_method = "s" or "s_gl"

            eta (float, optional): Regularization Parameter. Defaults to 1.

                                   Only used when ot_method = "s_gl" or "emd_l"
            max_iter (int, optional): (from POT) the maximum numer of iteration before stopping the optimization procedure
            cost_norm (bool, optional): Normalize the cost matrix
                                        Defaults to None.

            From POT: Type of normalization from 'median', 'max', 'log',
                      'loglog'. Any other value do not normalize

            limit_max (int, optional):  Controls the semi supervised mode.
                                        Transport between labeled source and
                                        target samples of different classes
                                        will exhibit an infinite cost (10 times
                                        the maximum value of the cost matrix)
                                        Defaults to 10.

            cost_supervised (bool, optional): Supervise the cost matrix. Defaults to True.
                                              This allows the user to calculate the cost matriz unsupervised even
                                              if the train and val labels are provided
        """
        # check naming
        naming_check(ot_method=ot_method,
                     penalized_type=penalized_type)

        # Initialize any hyperparameters for OT
        # Optimal Transport Method. EMD, S, GL
        self.ot_method = ot_method
        # Distance metric to compute transport plan
        self.metric = metric
        # Parameter for regularized version of OT
        self.reg = reg
        self.eta = eta
        # Cost norm and limit max of OT
        self.cost_norm = cost_norm
        self.limit_max = limit_max
        self.cost_supervised = cost_supervised
        self.max_iter = max_iter
        # Initialize any hyperparameters for penalization
        # Type of penalization (Distance or classifier probability)
        self.penalized_type = penalized_type
        # Penalization intensity. Is and hyperparameter to tune
        # If k=0 no penalization is applied
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
        self.balanced_train = balanced_train
        self.balanced_val = balanced_val

        self.ot_obj = initialize_ot_obj(self)

    def fit(self,
            X_train: ArrayLike,
            X_val: ArrayLike,
            clf,
            y_train: Optional[ArrayLike] = None,
            y_val: Optional[ArrayLike] = None,
            ):
        """
        Fit optimal transport method with penalization
        Classifier has to be trained on Xs as in BOTDA method.

        Parameters:
            X_train (ArrayLike): _description_
            X_val (ArrayLike): _description_
            clf (sklearn Classifier): _description_
            y_train (ArrayLike], optional): _description_.
            y_val (ArrayLike], optional): _description_.

            Defaults to None.

        Returns
        -------
        ot_obj :  Fitted Optimal transport instance
        """
        # Check consistency
        data_consistency_check(X_train=X_train,
                               X_val=X_val,
                               y_train=y_train,
                               y_val=y_val)

        mass_train, mass_val = initialize_uniform_weights(X_train=X_train,
                                                          X_val=X_val)
        # If source (train), labels are provided, then enter to the function
        # that allows to remove the wrongly classified points in train
        if (self.wrong_cls):
            # Deal with wrong classified point in the target domain
            X_train, y_train, mass_train = deal_with_wrong_classified(X_train,
                                                                      y_train,
                                                                      mass_train,                   # noqa
                                                                      clf)

        # Fit the object to the data in a backward way
        self.ot_obj = self.ot_obj.fit(Xs=X_val, ys=y_val,
                                      Xt=X_train, yt=y_train)

        # Change the weights of the target points with respect a penalization
        mass_train = compute_penalization(self.penalized_type, self.k,
                                          mass_train,
                                          X_train, clf)

        # Change the weights of the target points with respect a penalization
        mass_train = balance_samples(self.balanced_train, mass_train, y_train)
        mass_val = balance_samples(self.balanced_val, mass_val, y_val)

        # Compute cost matrix
        M = compute_cost_matrix(self, X_train=X_train, y_train=y_train,
                                X_val=X_val, y_val=y_val)

        # Compute coupling with different OT methods in a backward way
        G0 = compute_backward_coupling(self, mass_train, mass_val, M,
                                       X_train=X_train,
                                       X_val=X_val, y_val=y_val)

        # Replace the coupling with the penalized one
        self.ot_obj.coupling_ = G0
        self.ot_obj.mu_t = mass_train
        self.ot_obj.mu_s = mass_val
        self.ot_obj.cost_ = M
        return

        # Transform data using the fitted OT Element.
        # Transport samples from target to source

    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Transform data using the fitted OT Element.

        Args:
            X (ArrayLike): Input data to be transformed.

        Returns:
            ArrayLike: Transformed data.
        """
        # Transport samples from target to source using the fitted OT Element
        X_transformed = self.ot_obj.transform(Xs=X)
        return X_transformed
