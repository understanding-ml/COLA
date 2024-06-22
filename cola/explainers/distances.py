import ot
import numpy as np
from typing import Optional
import torch
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

import explainers.auxiliary as aux
from explainers.auxiliary import *


class WassersteinDivergence:
    def __init__(self, reg=1):
        self.nu = None
        self.reg = reg

    def distance(self, y_s: torch.tensor, y_t: torch.tensor, delta):
        # Validate delta
        if delta < 0 or delta > 0.5:
            raise ValueError("Delta should be between 0 and 0.5")

        y_s = y_s.squeeze()
        y_t = y_t.squeeze()

        # Calculate quantiles
        lower_quantile_s = torch.quantile(y_s, delta)
        upper_quantile_s = torch.quantile(y_s, 1 - delta)
        lower_quantile_t = torch.quantile(y_t, delta)
        upper_quantile_t = torch.quantile(y_t, 1 - delta)

        # Indices in the original tensors that correspond to the filtered values
        indices_s = torch.where((y_s >= lower_quantile_s) & (y_s <= upper_quantile_s))[
            0
        ]
        indices_t = torch.where((y_t >= lower_quantile_t) & (y_t <= upper_quantile_t))[
            0
        ]

        # Create a meshgrid to identify the locations in self.nu to be updated
        indices_s_grid, indices_t_grid = torch.meshgrid(
            indices_s, indices_t, indexing="ij"
        )

        # Filter data points
        y_s_filtered = y_s[indices_s]
        y_t_filtered = y_t[indices_t]

        proj_y_s_dist_mass = torch.ones(len(y_s_filtered)) / len(y_s_filtered)
        proj_y_t_dist_mass = torch.ones(len(y_t_filtered)) / len(y_t_filtered)

        trimmed_M_y = ot.dist(
            y_s_filtered.reshape(y_s_filtered.shape[0], 1),
            y_t_filtered.reshape(y_t_filtered.shape[0], 1),
            metric="sqeuclidean",
        ).to("cpu")

        trimmed_nu = ot.emd(proj_y_s_dist_mass, proj_y_t_dist_mass, trimmed_M_y)
        # trimmed_nu = ot.bregman.sinkhorn(
        #     proj_y_s_dist_mass, proj_y_t_dist_mass, M_y, reg=self.reg
        # )
        # trimmed_nu = torch.diag(torch.ones(len(y_s)))
        dist = torch.sum(trimmed_nu * trimmed_M_y) * (1 / (1 - 2 * delta))

        self.nu = torch.zeros(len(y_s), len(y_t))

        # Place the values of trimmed_nu in the correct positions in self.nu
        self.nu[indices_s_grid, indices_t_grid] = trimmed_nu

        return dist, self.nu

    def distance_interval(
        self,
        y_s: torch.tensor,
        y_t: torch.tensor,
        delta: float,
        alpha: Optional[float] = 0.05,
        bootstrap=True,
    ):
        if bootstrap:
            return bootstrap_1d(
                y_s.detach().numpy(), y_t.detach().numpy(), delta=delta, alpha=alpha
            )
        else:
            return exact_1d(
                y_s.detach().numpy(), y_t.detach().numpy(), delta=delta, alpha=alpha
            )


class SlicedWassersteinDivergence:
    def __init__(self, dim: int, n_proj: int, reg=1):
        self.dim = dim
        self.n_proj = n_proj
        # self.thetas = np.random.randn(n_proj, dim)
        # self.thetas /= np.linalg.norm(self.thetas, axis=1)[:, None]

        # sample from the unit sphere
        self.thetas = np.random.multivariate_normal(
            np.repeat(0, dim), np.identity(dim), size=n_proj
        )
        self.thetas = np.apply_along_axis(
            lambda x: x / np.linalg.norm(x), 1, self.thetas
        )

        self.wd = WassersteinDivergence()

        self.reg = reg

        self.mu_list = []

    def distance(self, X_s: torch.tensor, X_t: torch.tensor, delta):
        """
        Compute the sliced Wasserstein distance between X_s and X_t

        Parameters:
        X_s : np.ndarray, shape (n_samples_a, dim)
            samples in the source domain
        X_t : np.ndarray, shape (n_samples_b, dim)
            samples in the target domain
        metric : str, optional
            metric to be used for Wasserstein-1 distance computation

        Returns:
        swd : float
            Sliced Wasserstein Distance between X_s and X_t
        """

        self.mu_list = []
        dist = 0
        for theta in self.thetas:
            # Project data onto the vector theta
            theta = torch.from_numpy(theta).float()
            proj_X_s = X_s.to("cpu") @ theta
            proj_X_t = X_t.to("cpu") @ theta

            dist_wd, mu = self.wd.distance(proj_X_s, proj_X_t, delta)

            self.mu_list.append(mu)

            dist += dist_wd

        return dist / self.n_proj, self.mu_list

    def distance_interval(
        self,
        X_s: torch.tensor,
        X_t: torch.tensor,
        delta: float,
        alpha: Optional[float] = 0.05,
        bootstrap=True,
    ):
        if bootstrap:
            return bootstrap_sw(X_s, X_t, delta=delta, alpha=alpha, swd=self)
        else:
            N = len(self.thetas)
            low = []
            up = []
            for theta in self.thetas:
                # Project data onto the vector theta
                theta = torch.from_numpy(theta).float()
                proj_X_s = X_s.to("cpu") @ theta
                proj_X_t = X_t.to("cpu") @ theta

                l, u = self.wd.distance_interval(
                    proj_X_s, proj_X_t, delta=delta, alpha=alpha / N
                )

                low.append(np.power(l, 2))
                up.append(np.power(u, 2))

            left = np.power(np.mean(low), 1 / 2)
            right = np.power(np.mean(up), 1 / 2)

            return left, right


## The code below refers to
## https://github.com/tmanole/SW-inference/blob/master/ci.py
##
## Manole, Tudor, Sivaraman Balakrishnan, and Larry Wasserman.
## "Minimax confidence intervals for the sliced wasserstein distance."
## Electronic Journal of Statistics 16.1 (2022): 2252-2345.

""" One-dimensional CIs. """


def exact_1d(x, y, delta, alpha, r=2, mode="DKW", nq=1000):
    """Confidence intervals for W_{r,delta}(P, Q) in one dimension.

    Parameters
    ----------
    x : np.ndarray (n,)
        sample from P
    y : np.ndarray (m,)
        sample from Q
    r : int, optional
        order of the Wasserstein distance
    delta : float, optional
        trimming constant, between 0 and 0.5.
    alpha : float, optional
        number between 0 and 1, such that 1-alpha is the level of the confidence interval
    mode : str, optional
        either "DKW" to use a confidence interval based on the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality [1,2]
        or "rel_VC" to use a confidence interval based on the relative Vapnik-Chervonenkis (VC) inequality [3]
    nq : int, optional
        number of quantiles to use in Monte Carlo integral approximations

    Returns
    -------
    l : float
        lower confidence limit

    u : float
        upper confidence limit

    References
    ----------

    .. [1] Dvoretzky, Aryeh, Jack Kiefer, and Jacob Wolfowitz.
           "Asymptotic minimax character of the sample distribution function and
           of the classical multinomial estimator." The Annals of Mathematical Statistics (1956): 642-669.

    .. [2] Massart, Pascal. "The tight constant in the Dvoretzky-Kiefer-Wolfowitz inequality." The annals of Probability (1990): 1269-1283.

    .. [3] Vapnik, V., Chervonenkis, A.: On the uniform convergence of relative frequencies of events to
           their probabilities. Theory of Probability and its Applications 16 (1971) 264–280.

    """
    x = x.squeeze()
    y = y.squeeze()
    us = np.linspace(delta, 1 - delta, nq)

    if mode == "DKW":
        try:
            Lx, Ux = aux._dkw(x, us, alpha)
            Ly, Uy = aux._dkw(y, us, alpha)

        except OverflowError:
            return (0, np.Inf)

    elif mode == "rel_VC":
        try:
            Lx, Ux = aux._rel_vc(x, us, alpha)
            Ly, Uy = aux._rel_vc(y, us, alpha)

        except OverflowError:
            return (0, np.Inf)

    elif mode == "sequential":
        Lx, Ux = aux._quantile_seq(x, us, delta=alpha)[-1, :]
        Ly, Uy = aux._quantile_seq(y, us, delta=alpha)[-1, :]

    else:
        raise Exception("Mode unrecognized.")

    low = np.repeat(0, nq)
    up = np.repeat(0, nq)

    low = np.fmax(Lx - Uy, Ly - Ux)
    low = np.power(np.fmax(low, np.repeat(0, nq)), r)
    up = np.power(np.fmax(Ux - Ly, Uy - Lx), r)

    lower_final = np.power((1 / (1 - 2 * delta)) * np.mean(low), 1 / r)
    upper_final = np.power((1 / (1 - 2 * delta)) * np.mean(up), 1 / r)

    return lower_final, upper_final


def bootstrap_1d(x, y, delta, alpha, r=2, B=200):
    x = torch.tensor(x, dtype=torch.float32).squeeze()
    y = torch.tensor(y, dtype=torch.float32).squeeze()

    n, m = x.shape[0], y.shape[0]

    wd = WassersteinDivergence()
    dist_what, _ = wd.distance(x, y, delta)
    dist_what = dist_what.detach().numpy()

    # Generate all bootstrap indices at once
    x_indices = np.random.choice(n, (B, n))
    y_indices = np.random.choice(m, (B, m))

    W = np.empty(B)
    for i in range(B):
        xx = x[x_indices[i]]
        yy = y[y_indices[i]]

        dist, _ = wd.distance(xx, yy, delta)
        dist = dist.detach().numpy()
        W[i] = dist - dist_what

    q1, q2 = np.quantile(W, [alpha / 2, 1 - alpha / 2])

    Wlower = np.maximum(dist_what - q2, 0)
    Wupper = dist_what - q1

    if Wupper < 0:
        return 0, 0

    return np.power(Wlower, 1 / r), np.power(Wupper, 1 / r)


def bootstrap_sw(x, y, delta, alpha, swd, r=2, B=200):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    n, m, d = x.shape[0], y.shape[0], x.shape[1]

    SW_hat, _ = swd.distance(x, y, delta)
    SW_hat = SW_hat.detach().numpy()

    # Generate all bootstrap indices at once
    x_indices = np.random.choice(n, (B, n))
    y_indices = np.random.choice(m, (B, m))

    boot = np.empty(B)
    for i in range(B):
        xx = x[x_indices[i], :]
        yy = y[y_indices[i], :]

        SW_boot, _ = swd.distance(xx, yy, delta)
        SW_boot = SW_boot.detach().numpy()
        boot[i] = SW_boot - SW_hat

    q1, q2 = np.quantile(boot, [alpha / 2, 1 - alpha / 2])

    SW_lower = np.maximum(SW_hat - q2, 0)
    SW_upper = SW_hat - q1

    return np.power(SW_lower, 1 / r), np.power(SW_upper, 1 / r)
