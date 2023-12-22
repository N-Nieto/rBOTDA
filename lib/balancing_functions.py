import numpy as np
from numpy.typing import ArrayLike
from typing import List, Union, Tuple
from sklearn.base import ClassifierMixin


def balance_weights(y: ArrayLike, weights: ArrayLike,
                    balance: List[Union[int, float]]) -> np.ndarray:
    """
    Balance the weights of samples based on
    the specified balance strategy for different classes.
º
    Parameters:
    - y (ArrayLike): Target labels for class-specific balancing.
    - weights (ArrayLike): Input weights to be balanced.
    - balance (List[Union[int, float]]): List of weights for
    balancing different classes. If "auto," uniform relevance for each class.

    Returns:
    np.ndarray: Balanced weights for the samples.
    """
    # Get unique classes from labels
    classes = np.unique(y)
    if balance == "auto":
        # All classes sum the same
        balance = 1 / len(classes) * np.ones(classes.shape)
    elif not (sum(balance) == 1):
        raise ValueError("Relevance vector needs to sum to 1")

    # Initialize the new weights
    w_final = np.zeros(weights.shape)
    balance = np.array(balance)

    for cl in classes:
        mask = np.array(classes == cl)
        rel = balance[mask]

        # Total points
        w_cl = weights.copy()

        # Keep only the points for one class
        w_cl[y != cl] = 0

        # In the case that all points in one class have 0 weight
        # (i.e., if all were misclassified)
        if sum(w_cl) == 0:
            w_cl = np.ones((len(w_cl),)) / len(w_cl)

        # Normalize the weights with respect to the balance of
        # the class and the total mass of the class
        w_cl = w_cl * rel / (sum(w_cl))

        # Sum the weight of the particular class to the final weight vector
        w_final = w_final + w_cl

    return w_final


def initialize_uniform_weights(X_train: ArrayLike,
                               X_val: ArrayLike
                               ) -> Tuple[ArrayLike, ArrayLike]:
    """
    Initialize uniform weights for train and validation data.

    Parameters:
    - X_train (array-like): Train data.
    - X_val (array-like): Validation data.

    Returns:
    tuple: Tuple containing train and val uniform weights.
    """

    # Start train with uniform weights
    mass_train = np.ones((X_train.shape[0],)) / X_train.shape[0]

    # Start target with uniform weights
    mass_val = np.ones((X_val.shape[0],)) / X_val.shape[0]

    return mass_train, mass_val


def balance_samples(balance: Union[str, List[Union[int, float]]],
                    mass: ArrayLike, y: ArrayLike) -> np.ndarray:
    """
    Balance the weights of samples based on the specified balance strategy.

    Parameters:
    - balance ([str or List) Balance strategy.
                Can be "auto" for uniform relevance for each class
                or a list of weights.
    - madd (ArrayLike): Input weights to be balanced.
    - y (ArrayLike): Target labels for class-specific balancing.

    Returns:
    np.ndarray: Balanced weights for the samples.
    """

    if balance is None:
        balanced_sampes = mass

    # If "auto" use uniform relevance for each classs
    elif (balance == "auto"):
        if y is not None:
            # Check the y was provided for this type of balancing
            balanced_sampes = balance_weights(y, mass, balance)
        else:
            ValueError("label must be provided for balancing")
    # If the first element is int or float
    elif isinstance(balance[0], (int, float)):
        if y is not None:
            balanced_sampes = balance_weights(y, mass, balance)
        else:
            ValueError("label must be provided for balancing")
    else:
        raise Exception("Balance target not supported")

    # In any case, make sure the samples sum is 1.
    balanced_sampes = balanced_sampes/sum(balanced_sampes)
    return balanced_sampes


def deal_with_wrong_classified(X_train: ArrayLike, y_train: ArrayLike,
                               mass_train: ArrayLike,
                               clf: ClassifierMixin
                               ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Handle wrongly classified points in train data.

    Parameters:
    - X_train (npt.ArrayLike): Train data.
    - y_train (npt.ArrayLike): True labels for train data.
    - mass_train (npt.ArrayLike): Weights associated with each data point.
    - clf (ClassifierMixin): The trained classifier.

    Returns:
    Tuple[ArrayLike, ArrayLike, ArrayLike]: Tuple containing filtered
                                            X_train, y_train, and
                                            updated weights (mass_train).
    """
    if y_train is None:
        ValueError("Train labels must be provided to delet wrong classified")
    # Generate prediction over train data
    y_pred = clf.predict(X_train)

    # Check if we do not delete all points for one class
    if len(np.unique(y_train[y_train == y_pred])) < 2:
        Warning("All points for one class wrongly classified, continuing without removing wrong classified")    # noqa
    else:
        # Delet the points from the Xs, a and target
        X_train = X_train[y_train == y_pred]
        mass_train = mass_train[y_train == y_pred]
        y_train = y_train[y_train == y_pred]

        # If all the datapoins were missclassified
        if np.isnan(np.sum(mass_train)):
            # Target uniform weights
            mass_train = np.ones(((X_train.shape[0]),)) / (X_train.shape[0])

    return X_train, y_train, mass_train


def subsample_set(X: np.ndarray, y: np.ndarray,
                  mass: np.ndarray, train_size: List[int]
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Extracts data points with the highest mass for each class.

    Parameters:
    - X (np.ndarray): Feature matrix.
    - y (np.ndarray): Array of class labels.
    - mass (np.ndarray): Array of mass values for each data point.
    - train_size (List[int]): List specifying the number of data points to be extracted for each class.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray]: Extracted data (X, y, mass).
    # noqa 
    """
    # Dimension checks
    unique_classes = np.unique(y)

    for class_label, size in zip(unique_classes, train_size):
        mask = (y == class_label)
        if size > mask.sum():
            raise ValueError(" Not enoght samples in class " + str(class_label) + ". Requested size: " + (str(size) + " Available samples: " + str(mask.sum()))) # noqa

    # Initialization
    X_f = []
    y_f = []
    mass_f = []

    for class_label, size in zip(unique_classes, train_size):
        mask = (y == class_label)
        # Get indices of highest mass values
        sorted_indices = np.argsort(mass[mask])[::-1]
        selected_indices = np.where(mask)[0][sorted_indices][:size]

        X_f.extend(X[selected_indices])
        y_f.extend(y[selected_indices])
        mass_f.extend(mass[selected_indices])

    return np.array(X_f), np.array(y_f), np.array(mass_f)
