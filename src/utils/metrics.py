"""
Implement metrics used in Unmixing scenarios
"""
import logging

import numpy as np
import numpy.linalg as LA
import pandas as pd

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class BaseMetric:
    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    def _check_input(X, Xref):
        assert X.shape == Xref.shape
        assert type(X) == type(Xref)
        return X, Xref

    def __call__(self, X, Xref):
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}"


class MeanAbsoluteError(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return 100 * (1 - np.abs((E / normE).T @ (Eref / normEref)))


class SpectralAngleDistance(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        tmp = (E / normE).T @ (Eref / normEref)
        ret = np.minimum(tmp, 1.0)  # NOTE Handle floating errors
        # return np.arccos((E / normE).T @ (Eref / normEref))
        return np.arccos(ret)


class SADDegrees(SpectralAngleDistance):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        tmp = super().__call__(E, Eref)
        return (np.diag(tmp) * (180 / np.pi)).mean()


class MSE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)

        normE = LA.norm(E, axis=0, keepdims=True)
        normEref = LA.norm(Eref, axis=0, keepdims=True)

        return np.sqrt(normE.T**2 + normEref**2 - 2 * (E.T @ Eref))


class aRMSE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, A, Aref):
        A, Aref = self._check_input(A, Aref)
        return 100 * np.sqrt(((A - Aref) ** 2).mean())


class eRMSE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, E, Eref):
        E, Eref = self._check_input(E, Eref)
        # TODO L2 normalize endmembers
        return 100 * np.sqrt(((E - Eref) ** 2).mean())


class SRE(BaseMetric):
    def __init__(self):
        super().__init__()

    def __call__(self, X, Xref):
        X, Xref = self._check_input(X, Xref)
        return 20 * np.log10(LA.norm(Xref, "fro") / LA.norm(Xref - X, "fro"))


def compute_metric(metric, X_gt, X_hat, labels, detail=True, on_endmembers=False):
    """
    Return individual and global metric
    """
    d = {}
    d["Overall"] = round(metric(X_hat, X_gt), 4)
    if detail:
        for ii, label in enumerate(labels):
            if on_endmembers:
                x_gt, x_hat = X_gt[:, ii][:, None], X_hat[:, ii][:, None]
                d[label] = round(metric(x_hat, x_gt), 4)
            else:
                d[label] = round(metric(X_hat[ii], X_gt[ii]), 4)

    log.info(f"{metric} => {d}")
    return d


class RunAggregator:
    def __init__(
        self,
        metric,
        use_endmembers=False,
        detail=True,
    ):
        """
        Aggregate runs by tracking a metric
        """
        self.metric = metric
        self.use_endmembers = use_endmembers
        self.filename = f"{metric}.json"
        self.data = {}
        self.df = None
        self.summary = None
        self.detail = detail

    def add_run(self, run, X, Xhat, labels):

        d = {}
        d["Overall"] = self.metric(X, Xhat)
        if self.detail:
            for ii, label in enumerate(labels):
                if self.use_endmembers:
                    x, xhat = X[:, ii][:, None], Xhat[:, ii][:, None]
                    d[label] = self.metric(x, xhat)
                else:
                    d[label] = self.metric(X[ii], Xhat[ii])

        log.debug(f"Run {run}: {self.metric} => {d}")

        self.data[run] = d

    def aggregate(self, prefix=None):
        self.df = pd.DataFrame(self.data).T
        self.summary = self.df.describe().round(2)
        log.info(f"{self.metric} summary:\n{self.summary}")
        self.save(prefix)

    def save(self, prefix=None):
        prefix = "" if prefix is None else f"{prefix}-"

        df_fname = f"{prefix}runs-{self.filename}"
        summary_fname = f"{prefix}summary-{self.filename}"

        self.df.to_json(df_fname)
        self.summary.to_json(summary_fname)


class SADAggregator(RunAggregator):
    def __init__(self):
        super().__init__(
            SADDegrees(),
            use_endmembers=True,
            detail=True,
        )


class RMSEAggregator(RunAggregator):
    def __init__(self):
        super().__init__(
            aRMSE(),
            use_endmembers=False,
            detail=True,
        )


class ERMSEAggregator(RunAggregator):
    def __init__(self):
        super().__init__(
            eRMSE(),
            use_endmembers=True,
            detail=True,
        )


class SREAggregator(RunAggregator):
    def __init__(self):
        super().__init__(
            SRE(),
            use_endmembers=False,
            detail=False,
        )



import torch
import torch.nn as nn


class GammaKL:
    """
        KL divergence closed form for mGamma distrib from
        "Dirichlet Variational Autoencoder"by
        Weonyoung Joo, Wonsung Lee, Sungrae Park & Il-Chul Moon

        Analytical form:
        Let P and Q be 2 multiGamma distributions of parameters
        $\alpha, \beta$ (resp. $\alpha, \beta$) where $\sum_{i=1}^K \alpha_i = 1$
        and $\beta > 0$, (resp. $\sum_{i=1}^K \hat \alpha_i =1 \text{ and } \beta > 0$).
        Based on the paper the analytical form is derived as follows
        $$
            D_{KL} (Q \| P) =   \sum_{i=1}^K log(\Gamma(\hat \alpha_i)) \\
                              - \sum_{i=1}^K log( \Gamma( \alpha_i ) )
                              + \sum_{i=1}^K ( \hat \alpha_i - \alpha_i ) \psi(\hat \alpha_i)
        $$
        where $\psi(\cdot)$ is the derivative of the gamma function (digamma function).
    """
    def __init__(self,
                 alphas: torch.Tensor,
                 reduction: str = "sum"):
        """
            Provide alphas of target distribution
            (betas is assumed to be the same as target)
        """
        self.alphas    = alphas.to(dtype=torch.float32)
        self.reduction = reduction

    def to(self, device):
        self.alphas = self.alphas.to(device)
        return self

    def __repr__(self):
        return f"GammaKL({self.alphas}, beta assumed to be the same as target)"

    def __call__(self,
                 input: torch.Tensor):
        """
            Args:
                input:  moments of shape (batch_size, n_ems)
            Returns:
                output: shape (batch_size,)
        """
        #alphas shape (1, x) --> shape (batch_size, x)
        batch_size = input.shape[0]
        alphas = self.alphas.expand(batch_size, -1)

        loss  = torch.sum(torch.lgamma(alphas), dim=1)
        loss -= torch.sum(torch.lgamma(input), dim=1)
        loss += torch.sum((input - alphas) * torch.digamma(input), dim=1)

        if   self.reduction == "sum":  loss = loss.sum()
        elif self.reduction == "mean": loss = loss.mean()
        elif self.reduction == "none": pass
        else:
            raise ValueError("Invalid reduction mode")

        return loss

