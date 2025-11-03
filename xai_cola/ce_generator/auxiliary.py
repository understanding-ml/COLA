## The code origins from
## https://github.com/tmanole/SW-inference/blob/master/auxiliary.py
##
## Manole, Tudor, Sivaraman Balakrishnan, and Larry Wasserman.
## "Minimax confidence intervals for the sliced wasserstein distance."
## Electronic Journal of Statistics 16.1 (2022): 2252-2345.

import numpy as np

# from confseq import quantiles
# from confseq import boundaries
import numpy as np
import math


""" Helper functions. """


def _sample_quantile(x, u):
    """Compute the u-th sample quantile based on a sample x."""

    if np.any(u <= 0):
        raise OverflowError("Infinite quantile.")

    if np.any(u >= 1):
        raise OverflowError("Infinite quantile.")

    n = len(x)

    if n == 1:
        return np.repeat(x[0], u.size)

    x = np.sort(x)
    J = np.floor(n * u) - 1

    J[J < 0] = 0
    J[J > n - 1] = n - 1

    return x[[int(j) for j in J]]


def _dkw(x, u, alpha):
    """DKW lower and upper (1-alpha/2)-confidence bands for the u-quantiles of a distribution,
    based on a sample x."""

    n = len(x)
    gam = np.sqrt((1 / (2 * n)) * np.log(4 / alpha))  # 4 instead of 2.
    lower = _sample_quantile(x, u - gam)
    upper = _sample_quantile(x, u + gam)

    return lower, upper


def _rel_vc(x, u, alpha):
    """Relative VC lower and upper (1-alpha/2)-confidence bands for the u-quantiles of a distribution,
    based on a sample x."""

    n = len(x)

    alpha_n = 2 * np.sqrt((np.log(2 * n + 1) + np.log(8 / alpha)) / n)
    beta_n = (np.log(4 * n + 1) + np.log(8 / alpha)) / n

    gamma_n = (2 * (u - beta_n) + alpha_n**2) / (2 * (1 + alpha_n**2)) - (
        np.sqrt(
            (2 * (u - beta_n) + alpha_n**2) ** 2
            - 4 * (1 + alpha_n**2) * (u - beta_n) ** 2
        )
        / (2 * (1 + alpha_n**2))
    )
    eta_n = (2 * (u + beta_n) + alpha_n**2) / (2 * (1 - alpha_n**2)) + (
        np.sqrt(
            (2 * (u + beta_n) + alpha_n**2) ** 2
            - 4 * (1 - alpha_n**2) * (u + beta_n) ** 2
        )
        / (2 * (1 - alpha_n**2))
    )

    lower = _sample_quantile(x, gamma_n)
    upper = _sample_quantile(x, eta_n)

    return lower, upper


def _half_vc(x, u, d, alpha):
    """Lower and upper quantile confidence bounds based on the VC inequality over the collection of half spaces in dimension d."""
    n = len(x)
    gamma = np.sqrt((32 / (n)) * (np.log(8 / alpha) + (d + 1) * np.log(n + 1)))

    print(u)
    print(gamma)

    lower = _sample_quantile(x, u - gamma)
    upper = _sample_quantile(x, u + gamma)

    return lower, upper
