import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import ClassifierMixin


def distance_to_hyperplane(X: ArrayLike, clf: object) -> ArrayLike:
    """
    Calculate the distance of data points to
    the hyperplane defined by a classifier.

    Parameters:
    - X (array-like): Input data [Samples x Features].
    - clf (classifier): The trained classifier.

    Returns:
    array-like: Array of distances from data points to the hyperplane.
    """
    # Get the intercept and coefficients from the classifier
    b = clf.intercept_
    W = clf.coef_

    # Calculate the modulus of the coefficients
    mod = np.sqrt(np.sum(np.power(W, 2)))

    # Calculate the distance from data points to the hyperplane
    d = np.abs(np.dot(X, W.T) + b) / mod

    return d[:, 0]


def Dist_matrix(X: ArrayLike, clf: object, k: int) -> np.ndarray:
    """
    Compute the matrix Q based on the distance
    to the hyperplane defined by a classifier.

    Parameters:
    - X (ArrayLike): Input data.
    - clf (Optional[object]): The trained classifier.
    - k (int): Power parameter for distance transformation.

    Returns:
    np.ndarray: Matrix Q based on the distance to the hyperplane.
    """
    # Compute distance
    d = distance_to_hyperplane(X, clf)
    d = np.power(d, k)

    # Normalization term
    nom = np.prod(np.power(d, 1 / len(d)))

    # Penalization is proportional to the distance to
    # the classifier decision boundary
    penalization = d / nom
    penalization = np.diag(penalization)

    return penalization


def Proba_matrix(X: ArrayLike, clf: object, k: int) -> np.ndarray:
    """
    Compute the matrix of probabilities based on
    the output probabilities from a classifier.

    Parameters:
    - X (ArrayLike): Input data.
    - clf (Optional[object]): The trained classifier.
    - k (int): Power parameter for distance transformation.

    Returns:
    np.ndarray: Matrix of probabilities based on the classifier's output.
    """
    # Get the probabilities for each point
    d = clf.predict_proba(X)

    # Subtract the mean for each point and compute the absolute value
    d = np.abs(d - np.mean(d) * np.ones(d.shape))
    d = d.sum(axis=1)
    d = np.power(d, k)
    d = d + 1e-10

    # Normalization term
    nom = np.prod(np.power(d, 1 / len(d)))

    # penalization is proportional to the distance to
    # the classifier decision boundary
    penalization = d / nom
    penalization = np.diag(penalization)

    return penalization


def compute_penalization(penalized_type: str, k: int, mass: ArrayLike,
                         X_train: ArrayLike,
                         clf: ClassifierMixin) -> np.ndarray:
    """
    Compute the penalization with respect to the classifier.
    The penalization could be inversely proportional to the distance
    from the samples to the decision hyperplane (only linear classifiers)
    or with respect to the probability output of the classifier.

    Parameters:
    - penalized_type (str): Type of penalization ("distance" or "probability").
    - k (int): Power parameter for distance transformation.
    - a (ArrayLike): Input samples weights.
    - X_train (ArrayLike): Input training data.
    - clf (Optional[object]): The trained classifier.

    Returns:
    np.ndarray: Updated samples weights based on the computed penalization.
    """
    # Calculate the distance of each sample to the LDA decision straight
    if penalized_type in ["distance", "d"]:
        penalization = Dist_matrix(X=X_train, clf=clf, k=k)
    elif penalized_type in ["probability", "proba", "p"]:
        penalization = Proba_matrix(X=X_train, clf=clf, k=k)
    else:
        raise Exception("Penalization not supported")

    # Change the sample weight proportionally to the computed score
    mass = np.dot(mass, penalization)

    return mass
