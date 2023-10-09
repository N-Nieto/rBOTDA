import numpy as np


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


def initialize_sample_weights(Xt, Xs):
    # Start with Source uniform weights
    a = np.ones((Xs.shape[0]),) / (Xs.shape[0])
    # Start with Target uniform weights
    b = np.ones(((Xt.shape[0]),)) / (Xt.shape[0])
    return a, b


def deal_with_wrong_classified_point(self, a, b, Xt, yt, clf):

    # Do not assign mass to missclassified points
    if self.wrong_cls:
        Y_pred = clf.predict(Xt)
        b[yt != Y_pred] = 0

        # If all the datapoins for one classe were missclassified
        if sum(b) == np.NaN:
            # Target uniform weights
            b = np.ones(((Xt.shape[0]),)) / (Xt.shape[0])

        # Replace mass=0 for Grup Lasso method
        if self.ot_method in ["sinkhorn_gl", "s_gl"]:
            # Not suport mass==0
            b[yt != Y_pred] = 1e-10

    return Xt, yt, a, b


def compute_balance_weights(self, a, b, yt, ys):
    # Balance target
    # Chek if empy or False
    if not (self.balanced_target):
        # If empty use uniform relevance for each class
        if (self.balanced_target == []):
            if (yt is not None):
                b_final = balance_weights(yt, b, self.balanced_target)
            else:
                print("Target not balanced as yt was not provided")
                # If False uniform balance
                if self.k > 0:
                    b_final = b/sum(b)
                elif self.k == 0:
                    if self.wrong_cls:
                        b_final = b/sum(b)
                    else:
                        b_final = b

        # If False uniform balance
        elif self.k > 0:
            b_final = b/sum(b)
        elif self.k == 0:
            if self.wrong_cls:
                b_final = b/sum(b)
            else:
                b_final = b
    # If not False or Empty check if the first element is int or float
    elif isinstance(self.balanced_target[0], (int, float)):
        b_final = balance_weights(yt, b, self.balanced_target)
    else:
        raise Exception("Balance target not supported")
    # Balance source
    # Chek if empy or False
    if not (self.balanced_source):
        # If empty use uniform relevance for each class
        if (self.balanced_source == []):
            if (ys is not None):
                a_final = balance_weights(ys, a, self.balanced_source)
            else:
                print("Source not balanced as ys was not provided")
                # If False uniform balance
                a_final = a
        # If False not normalization
        else:
            a_final = a
    # If not False or Empty check if the first element is int or float
    elif isinstance(self.balanced_source[0], (int, float)):
        a_final = balance_weights(ys, a, self.balanced_source)
    else:
        raise Exception("Balance source not supported")

    return a_final, b_final


def delete_wrong_classified(clf, X, Y):
    # Make a prediction
    Y_pred = clf.predict(X)

    # Check if we do not delete all points for one class
    Y_test = Y[Y == Y_pred]

    if len(np.unique(Y_test)) < 2:
        Warning("All points for one class wrongly classified, continuing without removing wrong classified")    # noqa
    else:
        # Delete the data missclassified
        X = X[Y == Y_pred]
        Y = Y[Y == Y_pred]

    return X, Y
