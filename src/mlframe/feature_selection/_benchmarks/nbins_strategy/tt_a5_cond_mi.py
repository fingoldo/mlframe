"""tt_a5: Gaussian conditional MI I(f(y); x_j | S) in raw-Gaussian form vs Gaussian-copula (rank-normal) form. Shows copula form is invariant to monotone f; raw form is not.
Usage: python tt_a5_cond_mi.py
"""
import numpy as np
from scipy.stats import norm, rankdata


def cmi_gauss(y, x, S):
    """-0.5 log(1 - partial_corr^2) of y,x given columns S (linear regression residuals)."""
    if S.shape[1]:
        B = np.c_[np.ones(len(y)), S]
        y = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
        x = x - B @ np.linalg.lstsq(B, x, rcond=None)[0]
    r = np.corrcoef(y, x)[0, 1]
    return -0.5 * np.log(1 - r * r)


def cop(v):
    return norm.ppf((rankdata(v) - 0.5) / len(v))


rng = np.random.default_rng(0)
n = 20000
X = rng.standard_normal((n, 4))
for nm, y in (("lognormal", np.exp(0.9 * X[:, 0] + 0.6 * X[:, 1] + 0.4 * rng.standard_normal(n))),
              ("cube", (X[:, 0] + 0.7 * X[:, 1] + 0.3 * rng.standard_normal(n)) ** 3)):
    print(nm, "I(f(y);x1 | x0)  raw-gaussian form vs copula form")
    for fn, f in (("identity", lambda t: t), ("log" if nm == "lognormal" else "cbrt", np.log if nm == "lognormal" else np.cbrt)):
        z = f(y)
        print("  %-9s raw %.4f   copula %.4f   (I(f(y);x1) raw %.4f copula %.4f)" % (
            fn, cmi_gauss(z, X[:, 1], X[:, [0]]), cmi_gauss(cop(z), cop(X[:, 1]), cop(X[:, 0])[:, None]),
            cmi_gauss(z, X[:, 1], X[:, :0]), cmi_gauss(cop(z), cop(X[:, 1]), np.zeros((n, 0)))))
