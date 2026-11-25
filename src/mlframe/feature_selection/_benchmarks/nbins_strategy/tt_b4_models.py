"""Judges for the feature-transform study (research B4): ridge, additive spline GAM, HistGradientBoosting, and a Gram-matrix brute force over
(column, candidate) replacements (the oracle). All metrics are on rows the transform recommender never saw.
"""
from __future__ import annotations

import warnings

import numpy as np

from tt_b4_reference import CAND_DEFAULT, ColParams, _col_params_j, design_column, rg_grid, valid_mask

warnings.simplefilter("ignore")


def r2_rmse(y, pred):
    res = ((y - pred) ** 2).sum()
    return float(1 - res / ((y - y.mean()) ** 2).sum()), float(np.sqrt(res / len(y)))


def _std_fit(A):
    mu = A.mean(axis=0)
    sd = A.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def ridge_fit_predict(Atr, ytr, Ate, lam=1.0):
    mu, sd = _std_fit(Atr)
    Z = (Atr - mu) / sd
    ym = ytr.mean()
    G = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(G, Z.T @ (ytr - ym))
    return ((Ate - mu) / sd) @ w + ym


# ---------------------------------------------------------------------------------------------------------------------
# additive spline GAM: per-column cubic B-spline basis on quantile knots (one-hot for columns with <= 8 distinct values), ridge on the stacked bases
# ---------------------------------------------------------------------------------------------------------------------
def _bspline_basis(x, knots_inner, degree=3):
    from scipy.interpolate import BSpline

    lo, hi = knots_inner[0], knots_inner[-1]
    t = np.r_[[lo] * degree, knots_inner, [hi] * degree]
    xc = np.clip(x, lo, hi)
    B = BSpline.design_matrix(xc, t, degree).toarray()
    return B[:, 1:].astype(np.float32)  # drop one column (the basis sums to one: the intercept)


class AdditiveBasis:
    def __init__(self, Xfit, n_knots=6, degree=3):
        self.specs = []
        for j in range(Xfit.shape[1]):
            u = np.unique(Xfit[:, j])
            if u.size <= 8:
                self.specs.append(("onehot", u[1:]))
            else:
                q = np.unique(np.quantile(Xfit[:, j], np.linspace(0, 1, n_knots)))
                if q.size < 4:
                    self.specs.append(("onehot", u[1:] if u.size <= 30 else q))
                else:
                    self.specs.append(("bspline", q))
        self.degree = degree

    def transform(self, X):
        out = []
        for j, (kind, k) in enumerate(self.specs):
            x = X[:, j]
            if kind == "onehot":
                out.append((x[:, None] == k[None, :]).astype(np.float32))
            else:
                out.append(_bspline_basis(x, k, self.degree))
        return np.hstack(out)


def gam_fit_predict(Atr, ytr, Ate, lam=1.0, n_knots=6, dtype=np.float32):
    basis = AdditiveBasis(Atr, n_knots=n_knots)
    Btr = basis.transform(Atr)
    Bte = basis.transform(Ate)
    mu = Btr.mean(axis=0, dtype=np.float64)
    sd = Btr.std(axis=0, dtype=np.float64)
    sd = np.where(sd > 1e-12, sd, 1.0)
    mu, sd = mu.astype(dtype), sd.astype(dtype)
    Btr -= mu; Btr /= sd
    Bte -= mu; Bte /= sd
    ym = ytr.mean()
    G = (Btr.T @ Btr).astype(np.float64) + lam * np.eye(Btr.shape[1])
    w = np.linalg.solve(G, (Btr.T @ (ytr - ym).astype(dtype)).astype(np.float64))
    return (Bte @ w.astype(dtype)).astype(np.float64) + ym


def hgb_fit_predict(Atr, ytr, Ate, seed=0, max_iter=200):
    from sklearn.ensemble import HistGradientBoostingRegressor

    m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                                      early_stopping=False, random_state=seed)
    m.fit(Atr, ytr)
    return m.predict(Ate)


MODELS = {"ridge": ridge_fit_predict, "gam": gam_fit_predict, "hgb": hgb_fit_predict}


# ---------------------------------------------------------------------------------------------------------------------
# brute-force replacement table (oracle): R2 on an evaluation set of the ridge model with ONE column replaced by one candidate transform
# ---------------------------------------------------------------------------------------------------------------------
def replacement_table(Xfit, yfit, Xeval, yeval, cols=None, cands=CAND_DEFAULT, lam=1.0):
    """Returns (R2 base on eval, table (len(cols), K) of eval R2 with column j replaced by candidate k; nan where the candidate is invalid).
    Candidate constants (median, IQR, rank-gauss grid, validity) come from the fit rows only."""
    n, p = Xfit.shape
    cols = list(range(p)) if cols is None else list(cols)
    params = ColParams.fit(Xfit)
    valid = valid_mask(params, cands)
    mu, sd = _std_fit(Xfit)
    Z = (Xfit - mu) / sd
    Ze = (Xeval - mu) / sd
    ym = yfit.mean()
    yc = yfit - ym
    G0 = Z.T @ Z
    c0 = Z.T @ yc
    I = lam * np.eye(p)
    w0 = np.linalg.solve(G0 + I, c0)
    base = r2_rmse(yeval, Ze @ w0 + ym)[0]
    tab = np.full((len(cols), len(cands)), np.nan)
    for a, j in enumerate(cols):
        for k, c in enumerate(cands):
            if not valid[j, k]:
                continue
            if c == "id":
                tab[a, k] = base
                continue
            grid = rg_grid(Xfit[:, j]) if c == "rankgauss" else None
            pj = _col_params_j(params, j)
            v = design_column(Xfit[:, j], pj, c, grid)
            ve = design_column(Xeval[:, j], pj, c, grid)
            m_, s_ = v.mean(), v.std()
            if s_ < 1e-12:
                continue
            v = (v - m_) / s_
            ve = (ve - m_) / s_
            u = Z.T @ v
            G = G0.copy()
            G[j, :] = u; G[:, j] = u; G[j, j] = float(v @ v)
            cc = c0.copy(); cc[j] = float(v @ yc)
            w = np.linalg.solve(G + I, cc)
            pred = Ze @ w - Ze[:, j] * w[j] + ve * w[j] + ym
            tab[a, k] = r2_rmse(yeval, pred)[0]
    return base, tab
