import numpy as np


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


def compute_penalization(self, Xt, clf, b):
    """
    Compute the penalization with respect the classifier
    The penalization could be inversly proportional to the distance
    from the samples to the decicion hyperplane (only linear classifiers)
    Or with respect the probability output of the classifier.
    """
    # Calculate the distance of each point to lda decision straight
    if self.penalized_type in ["distance", "d"]:
        Q = Q_matix(X=Xt, clf=clf, k=self.k)
    elif self.penalized_type in ["probability", "proba", "p"]:
        Q = Proba_matix(X=Xt, clf=clf, k=self.k)
    else:
        raise Exception("Penalization not supported")

    # change the point weight proportionaly to the computed score
    b = np.dot(b, Q)

    return b
