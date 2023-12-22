# -*- coding: utf-8 -*-
"""
utils for rBOTDA
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar
"""
import os
import numpy as np
import scipy.io as sio
from numpy.typing import ArrayLike
from typing import Optional, Tuple


def ensure_dir(dir_name: str):
    """Ensure Directory

    Args:
        dir_name (string): direction name
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def naming_check(ot_method: str,
                 penalized_type: str):
    """Check the variables naming

    Args:
        ot_method (str): OT Method to implement
        penalized_type (str): Type of penalization
    """

    if ot_method not in ["emd", "sinkhorn", "s", "sinkhorn_gl", "s_gl",
                         "emd_laplace", "emd_l"]:
        RuntimeError("Invalid OT method")

    if penalized_type not in ["distance", "d", "probability", "proba", "p"]:
        RuntimeError("Invalid penalized type")


def data_consistency_check(X_train: ArrayLike,
                           X_val: ArrayLike,
                           y_train: Optional[ArrayLike] = None,
                           y_val: Optional[ArrayLike] = None):
    """ # noqa
    Check data dimensions

    Args:
        X_train (ArrayLike): Trian data [Samples x Features]
        X_val (ArrayLike): Val data [Samples x Features]
        y_train (ArrayLike optional): Train Targets [Samples]. Defaults to None.
        y_val (ArrayLike, optional): Val Targets [Sample]. Defaults to None.
    """

    # Check consistensy if ys is provided
    if (y_train is not None):
        # Data sanity check
        if X_train.shape[0] != y_train.shape[0]:
            raise RuntimeError("Missmach in train samples")
    # Check consistensy if yt is provided
    if (y_val is not None):
        if X_val.shape[0] != y_val.shape[0]:
            raise RuntimeError("Missmach validation samples")


def load_synthetic_data(unrepresentative_features: int = 0
                        ) -> Tuple[ArrayLike, ArrayLike]:
    """ Load data with unrepresentative features
        The informative features are two half moons

    Args:
        unrepresentative_features (int, optional): Defaults to 0.

    Returns:
        X: Data     [Samples x Features]
        y: target   [Samples]
    """
    # Relative path
    root_dir = "../data/synthetic_data/"

    # Load all data
    data = sio.loadmat(file_name=root_dir + "Synthetic_data.mat")

    # Get the X
    X = data['X']

    # Keep only the number of features
    X = X[:, 0:2 + unrepresentative_features]

    # Get the target
    y = data['Y']

    # Labels are in -1 +1, change it to 1 2
    y = ((y + 1) / 2) + 1
    y = np.squeeze(y)
    return X, y


def split_synthetic_data_unbalanced(X: ArrayLike, Y: ArrayLike,
                                    Balance, shuffle: bool = False
                                    ) -> Tuple[ArrayLike, ArrayLike,
                                               ArrayLike, ArrayLike]:
    """ # noqa
    Split synthetic unbalanced data into training and validation sets.

    Args:
        X (ArrayLike): Features.
        Y (ArrayLike): Labels.
        Balance (tuple): Number of samples for each class in the training set.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]: Tuple containing X_train, Y_train, X_val, and Y_val.
    """
    if shuffle:
        # Shuffle the data if requested
        indices = np.random.permutation(Y.shape[0])
        X, Y = X[indices], Y[indices]

    # Create boolean masks for class 1 and class 2
    mask_c1, mask_c2 = (Y == 1), (Y == 2)

    # Training set samples and labels
    X_trn = np.vstack([X[mask_c1][:Balance[0]], X[mask_c2][:Balance[1]]])
    Y_trn = np.concatenate([Y[mask_c1][:Balance[0]], Y[mask_c2][:Balance[1]]])

    # Validation set samples and labels
    X_val = np.vstack([X[mask_c1][Balance[0]:], X[mask_c2][Balance[1]:]])
    Y_val = np.concatenate([Y[mask_c1][Balance[0]:], Y[mask_c2][Balance[1]:]])

    return X_trn, Y_trn, X_val, Y_val
