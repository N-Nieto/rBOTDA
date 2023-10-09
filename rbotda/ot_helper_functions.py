import numpy as np
import ot


def initialize_ot_obj(self):
    # Compute coupling
    if self.ot_method == "emd":
        self.ot_obj = ot.da.EMDTransport(metric=self.metric)

    elif self.ot_method in ["sinkhorn", "s"]:
        self.ot_obj = ot.da.SinkhornTransport(metric=self.metric,
                                              reg_e=self.reg_e)

    elif self.ot_method in ["sinkhorn_gl", "s_gl"]:
        self.ot_obj = ot.da.SinkhornL1l2Transport(metric=self.metric,
                                                  reg_e=self.reg_e,
                                                  reg_cl=self.eta)

    elif self.ot_method in ["emd_laplace", "emd_l"]:
        self.ot_obj = ot.da.EMDLaplaceTransport(metric=self.metric)
    else:
        raise RuntimeError("OT method not supported")

    return


def compute_cost_matrix(self, Xs, ys, Xt, yt):
    # pairwise distance
    M = ot.dist(Xs, Xt, metric=self.metric)
    M = ot.utils.cost_normalization(M, self.cost_norm)

    if (ys is not None) and (yt is not None):
        if self.limit_max != np.infty:
            limit_max = self.limit_max * np.max(M)

        # assumes labeled source samples occupy the first rows
        # and labeled target samples occupy the first columns
        classes = [c for c in np.unique(ys) if c != -1]
        for c in classes:
            idx_s = np.where((ys != c) & (ys != -1))
            idx_t = np.where(yt == c)

            # all the coefficients corresponding to a source sample
            # and a target sample :
            # with different labels get a infinite
            for j in idx_t[0]:
                M[idx_s[0], j] = limit_max

    return M


def compute_coupling(self, a, b, M, Xs, Xt, ys):
    # Compute coupling
    if self.ot_method == "emd":
        G0 = ot.da.emd(a=a, b=b, M=M)

    elif self.ot_method in ["sinkhorn", "s"]:
        G0 = ot.da.sinkhorn(a=a, labels_a=ys, b=b, M=M, reg=self.reg_e)

    elif self.ot_method in ["sinkhorn_gl", "s_gl"]:
        G0 = ot.da.sinkhorn_l1l2_gl(a=a, labels_a=ys, b=b, M=M,
                                    reg=self.reg_e, eta=self.eta)

    elif self.ot_method in ["emd_laplace", "emd_l"]:
        G0 = ot.da.emd_laplace(a=a, b=b, Xs=Xs, Xt=Xt, M=M, eta=self.eta)
    else:
        raise RuntimeError("OT method not supported")
    return G0
