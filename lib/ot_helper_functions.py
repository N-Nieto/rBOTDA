import ot
import numpy as np
import numpy.typing as npt
from typing import Optional


def initialize_ot_obj(self) -> ot.da.BaseTransport:
    """
    Initialize the optimal transport object based on the chosen OT method.

    Returns:
        ot.da.BaseTransport: Initialized optimal transport object.
    """
    if self.ot_method == "emd":
        ot_obj = ot.da.EMDTransport(metric=self.metric)

    elif self.ot_method in ["sinkhorn", "s"]:
        ot_obj = ot.da.SinkhornTransport(metric=self.metric,
                                         reg_e=self.reg)

    elif self.ot_method in ["sinkhorn_gl", "s_gl"]:
        ot_obj = ot.da.SinkhornL1l2Transport(metric=self.metric,
                                             reg_e=self.reg,
                                             reg_cl=self.eta)

    elif self.ot_method in ["emd_laplace", "emd_l"]:
        ot_obj = ot.da.EMDLaplaceTransport(metric=self.metric)
    else:
        raise RuntimeError("OT method not supported")

    return ot_obj


def compute_cost_matrix(self, X_train: npt.ArrayLike, X_val: npt.ArrayLike,
                        y_train: Optional[npt.ArrayLike] = None, y_val:
                        Optional[npt.ArrayLike] = None) -> npt.ArrayLike:
    """
    Compute the cost matrix for optimal transport between
    source (val) and target (train) domains.

    Args:
        X_train (npt.ArrayLike): Target domain samples.
        X_val (npt.ArrayLike): Source domain samples.
        y_train (Optional[npt.ArrayLike]): Target domain labels.
        y_val (Optional[npt.ArrayLike]): Source domain labels.

    Returns:
        npt.ArrayLike: Cost matrix for optimal transport.
    """
    # Pairwise distance computation
    M = ot.dist(X_val, X_train, metric=self.metric)

    # Normalize the cost matrix
    M = ot.utils.cost_normalization(M, self.cost_norm)

    if (y_train is not None) and (y_val is not None) and (self.cost_supervised):                # noqa
        # Apply cost adjustments for supervised training
        if self.limit_max != np.infty:
            limit_max = self.limit_max * np.max(M)

        # Assumes labeled source samples occupy the first rows
        # and labeled target samples occupy the first columns
        classes = [c for c in np.unique(y_val) if c != -1]
        for c in classes:
            idx_s = np.where((y_val != c) & (y_val != -1))
            idx_t = np.where(y_train == c)

            # Set the coefficients corresponding to a source sample
            # and a target sample with different labels to infinity
            for j in idx_t[0]:
                M[idx_s[0], j] = limit_max

    return M


def compute_backward_coupling(self,
                              mass_train: npt.ArrayLike,
                              mass_val: npt.ArrayLike,
                              M: npt.ArrayLike,
                              X_val: npt.ArrayLike,
                              y_val: npt.ArrayLike,
                              X_train: npt.ArrayLike) -> npt.ArrayLike:
    """
    Compute the backward coupling between source and target samples.
    This methods lears to transport samples from the validation domain (source)
    to the train domain (target)

    Args:
        mass_train (npt.ArrayLike): Source sample weights.
        mass_val (npt.ArrayLike): Target sample weights.
        M (npt.ArrayLike): Cost matrix.
        X_val (npt.ArrayLike): Source samples.
        yt (npt.ArrayLike): Target sample labels.
        X_train (npt.ArrayLike): Target samples.


    Returns:
        npt.ArrayLike: Backward coupling matrix.
    """
    # Check if the OT method is supported
    if self.ot_method == "emd":
        # Earth Mover's Distance (EMD) coupling
        G0 = ot.da.emd(a=mass_val, b=mass_train, M=M)

    elif self.ot_method in ["sinkhorn", "s"]:
        # Sinkhorn coupling
        G0 = ot.da.sinkhorn(a=mass_val, labels_a=y_val, b=mass_train, M=M,
                            reg=self.reg)

    elif self.ot_method in ["sinkhorn_gl", "s_gl"]:
        # Sinkhorn coupling with Group L1L2 regularization
        G0 = ot.da.sinkhorn_l1l2_gl(a=mass_val, labels_a=y_val,
                                    b=mass_train, M=M,
                                    reg=self.reg, eta=self.eta)

    elif self.ot_method in ["emd_laplace", "emd_l"]:
        # EMD coupling with Laplace regularization
        G0 = ot.da.emd_laplace(a=mass_val, b=mass_train,
                               Xs=X_val, Xt=X_train, M=M, eta=self.eta)

    else:
        # Raise an error if the OT method is not supported
        raise RuntimeError("OT method not supported")

    return G0
