import numpy as np
from statistics import NormalDist
from math import erf
from scipy.stats import norm


def _normdist_calculate_overlap(mu1, sigma1, mu2, sigma2):
    """
    Re-implementation of Distnorm.overlap() in Python 3.9 statistics module
    :param mu1:
    :param sigma1:
    :param mu2:
    :param sigma2:
    :return: distributional overlap
    """

    if (sigma2, mu2) < (sigma1, mu1): # sort to assure commutativity
        sigma1, sigma2 = sigma2, sigma1
        mu1, mu2 = mu2, mu1

    X_var, Y_var = np.square(sigma1), np.square(sigma2)
    if not X_var or not Y_var:
        raise RuntimeError('overlap() not defined when sigma is zero')
    dv = Y_var - X_var
    dm = np.fabs(mu2 - mu1)

    if not dv:
        return 1.0 - erf(dm / (2.0 * sigma1 * np.sqrt(2.0)))
    a = mu1 * Y_var - mu2 * X_var
    b = sigma1 * sigma2 * np.sqrt(dm**2.0 + dv * np.log(Y_var / X_var))
    x1 = (a + b) / dv
    x2 = (a - b) / dv

    return 1.0 - (np.fabs(norm.cdf(x1, mu2, sigma2) - norm.cdf(x1, mu1, sigma1)) +
            np.fabs(norm.cdf(x2, mu2, sigma2) - norm.cdf(x2, mu1, sigma1)))


d1 = NormalDist(2, 5)
d2 = NormalDist(3, 8)

print(d1.overlap(d2))
print(_normdist_calculate_overlap(2, 5, 3, 8))
