"""Scaled linear probes with the repository's own solvers, in place of ``make_pipeline(StandardScaler(), LinearRegression() | Ridge() | LogisticRegression())``.

Each class standardises the columns with the train fold's mean and population standard deviation (a constant column is scaled by 1, as ``StandardScaler`` does) and solves the same problem as the
scikit-learn model it replaces: ordinary least squares with an intercept (``lstsq``, so the minimum-norm solution of a rank-deficient design, as ``LinearRegression``), ridge with the intercept
unpenalised (``Ridge(alpha)``), and L2 logistic regression with the inverse strength ``C`` and the intercept unpenalised (``LogisticRegression(C)``): Newton for two classes, a quasi-Newton
minimiser of the same multinomial objective for more. The objectives are convex, so the optimum is the one lbfgs approaches; predictions agree with the pipelines to the tolerance of that solver.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _standardise_fit(X: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Train mean and population standard deviation of the columns of ``X`` (a zero spread is replaced by 1)."""
    # Intentional float64: the normal equations and Newton steps square the conditioning of the columns, so the design is held in double precision
    A = np.asarray(X, dtype=np.float64)
    mean = A.mean(axis=0)
    scale = A.std(axis=0)
    return mean, np.where(scale > 0.0, scale, 1.0)


class ScaledOLS:
    """Standardise, then ordinary least squares with an intercept; ``predict`` and ``score`` (R^2) like the pipeline."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledOLS":
        """Fit on the rows of ``X``; returns ``self``."""
        self.mean_, self.scale_ = _standardise_fit(X)
        A = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        design = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
        coef = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64).ravel(), rcond=None)[0]
        self.coef_, self.intercept_ = coef[:-1], float(coef[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictions for the rows of ``X``."""
        return ((np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_) @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Coefficient of determination of the predictions for ``X`` against ``y`` (``1 - SSE / SST``; 1.0 for a constant ``y`` fitted exactly and 0.0 otherwise, as scikit-learn returns)."""
        t = np.asarray(y, dtype=np.float64).ravel()
        sst = float(((t - t.mean()) ** 2).sum())
        sse = float(((t - self.predict(X)) ** 2).sum())
        if sst == 0.0:
            return 1.0 if sse == 0.0 else 0.0
        return 1.0 - sse / sst


class ScaledRidge(ScaledOLS):
    """Standardise, then ridge regression with penalty ``alpha`` (the constructor argument) and an unpenalised intercept."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledRidge":
        """Fit on the rows of ``X``; returns ``self``."""
        from ._fe_probe_ridge import fit_ridge_closed_form

        self.mean_, self.scale_ = _standardise_fit(X)
        A = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        t = np.asarray(y, dtype=np.float64).ravel()
        solved = fit_ridge_closed_form(A, t, self.alpha)
        if solved is None:  # singular or non-finite normal equations: least squares on the centred design stacked with sqrt(alpha) * I
            p = A.shape[1]
            a_mean = A.mean(axis=0)
            stacked = np.concatenate([A - a_mean, np.sqrt(self.alpha) * np.eye(p)], axis=0)
            w = np.linalg.lstsq(stacked, np.concatenate([t - t.mean(), np.zeros(p)]), rcond=None)[0]
            solved = (w, float(t.mean() - a_mean @ w))
        self.coef_, self.intercept_ = solved
        return self


class ScaledLogistic:
    """Standardise, then L2 logistic regression with inverse strength ``C`` and an unpenalised intercept (labels may be any sortable values); ``max_iter`` scales the iteration budget of the minimiser."""

    def __init__(self, C: float = 1.0, max_iter: int = 200) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledLogistic":
        """Fit on the rows of ``X``; returns ``self``."""
        self.mean_, self.scale_ = _standardise_fit(X)
        A = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        self.classes_, codes = np.unique(np.asarray(y).ravel(), return_inverse=True)
        k = self.classes_.size
        if k == 2:
            from ._fe_accuracy_gate_newton import fit_logistic_ridge_newton

            fit = fit_logistic_ridge_newton(A, codes.astype(np.float64), self.C)
            if fit is not None:
                w, b = fit
                self.coef_, self.intercept_ = np.asarray(w, dtype=np.float64).reshape(1, -1), np.asarray([b], dtype=np.float64)
                return self
        self.coef_, self.intercept_ = self._minimise(A, codes, k)
        return self

    def _minimise(self, A: np.ndarray, codes: np.ndarray, k: int) -> "tuple[np.ndarray, np.ndarray]":
        """Minimise the penalised log loss ``sum_i -log p(y_i | A_i) + ||W||^2 / (2 C)`` (intercepts unpenalised) with L-BFGS: the one-column sigmoid form for two classes, the softmax for more."""
        from scipy.optimize import minimize

        n, p = A.shape
        options = {"maxiter": self.max_iter * 50, "gtol": 1e-8}
        if k == 2:

            def binary(theta: np.ndarray) -> "tuple[float, np.ndarray]":
                """Binary penalised log loss and its gradient; ``theta = [w, b]``."""
                w, b = theta[:p], theta[p]
                z = A @ w + b
                loss = float(np.sum(np.logaddexp(0.0, z) - codes * z) + 0.5 * (w @ w) / self.C)
                r = 1.0 / (1.0 + np.exp(-z)) - codes
                return loss, np.concatenate([A.T @ r + w / self.C, [r.sum()]])

            res = minimize(binary, np.zeros(p + 1), jac=True, method="L-BFGS-B", options=options)
            return res.x[:p].reshape(1, -1), np.asarray([res.x[p]])
        onehot = np.zeros((n, k))
        onehot[np.arange(n), codes] = 1.0

        def softmax(theta: np.ndarray) -> "tuple[float, np.ndarray]":
            """Multinomial penalised log loss and its gradient; ``theta`` is the flattened ``(k, p + 1)`` coefficient table."""
            table = theta.reshape(k, p + 1)
            z = A @ table[:, :p].T + table[:, p]
            z -= z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            loss = float(-(onehot * logp).sum() + 0.5 * (table[:, :p] ** 2).sum() / self.C)
            r = np.exp(logp) - onehot
            g = np.concatenate([r.T @ A + table[:, :p] / self.C, r.sum(axis=0)[:, None]], axis=1)
            return loss, g.ravel()

        res = minimize(softmax, np.zeros(k * (p + 1)), jac=True, method="L-BFGS-B", options=options)
        table = res.x.reshape(k, p + 1)
        return table[:, :p], table[:, p]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Scores of the rows of ``X``: a vector for two classes, a ``(n, k)`` table for more."""
        z = ((np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_) @ self.coef_.T + self.intercept_
        return z.ravel() if self.coef_.shape[0] == 1 else z

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, columns ordered as ``classes_``."""
        z = self.decision_function(X)
        if z.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-z))
            return np.column_stack([1.0 - p1, p1])
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most probable class labels."""
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def cross_val_r2(make_model, X: np.ndarray, y: np.ndarray, splits: "list[tuple[np.ndarray, np.ndarray]]") -> "Optional[float]":
    """Mean R^2 of ``make_model().fit(train).score(validation)`` over ``splits`` (a list of ``(train_idx, validation_idx)``); ``None`` without splits."""
    if not splits:
        return None
    scores = [make_model().fit(X[tr], y[tr]).score(X[va], y[va]) for tr, va in splits]
    return float(np.mean(scores))
