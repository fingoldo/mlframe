"""Scaled linear probes with the repository's own solvers, in place of ``make_pipeline(StandardScaler(), LinearRegression() | Ridge() | LogisticRegression())``.

Each class standardises the columns with the train fold's mean and population standard deviation (a constant column is scaled by 1, as ``StandardScaler`` does) and solves the same problem as the
scikit-learn model it replaces: ordinary least squares with an intercept (``lstsq``, so the minimum-norm solution of a rank-deficient design, as ``LinearRegression``), ridge with the intercept
unpenalised (``Ridge(alpha)``), and L2 logistic regression with the inverse strength ``C`` and the intercept unpenalised (``LogisticRegression(C)``): Newton for two classes, a quasi-Newton
minimiser of the same multinomial objective for more. The objectives are convex, so the optimum is the one lbfgs approaches; predictions agree with the pipelines to the tolerance of that solver.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
from mlframe.utils.array_shape import shape_of
from numba import njit


def _standardise_fit(X: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Train mean and population standard deviation of the columns of ``X`` (the spread of a column that is constant up to rounding is replaced by 1, as ``StandardScaler`` does)."""
    # Intentional float64: the normal equations and Newton steps square the conditioning of the columns, so the design is held in double precision
    A = np.asarray(X, dtype=np.float64)
    n = A.shape[0]
    mean = A.mean(axis=0)
    var = A.var(axis=0)
    # a column indistinguishable from a constant (its variance within the rounding bounds of the two-pass variance, Chan, Golub and LeVeque) keeps a spread of 1, as StandardScaler decides
    # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
    eps = np.finfo(np.float64).eps
    constant = var <= n * eps * var + (n * mean * eps) ** 2
    return mean, np.where(constant, 1.0, np.sqrt(var))


def standardised(X: np.ndarray) -> np.ndarray:
    """``StandardScaler().fit_transform(X)``: the columns minus their mean over their population standard deviation (a constant column keeps a spread of 1, so it comes out as zeros)."""
    mean, scale = _standardise_fit(X)
    # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
    return np.asarray((np.asarray(X, dtype=np.float64) - mean) / scale)


class ScaledOLS:
    """Standardise, then ordinary least squares with an intercept; ``predict`` and ``score`` (R^2) like the pipeline."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledOLS":
        """Fit on the rows of ``X``; returns ``self``."""
        self.mean_, self.scale_ = _standardise_fit(X)
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
        A = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        design = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
        coef = np.linalg.lstsq(design, np.asarray(y, dtype=np.float64).ravel(), rcond=None)[0]
        self.coef_, self.intercept_ = coef[:-1], float(coef[-1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictions for the rows of ``X``."""
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
        return np.asarray(((np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_) @ self.coef_ + self.intercept_)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Coefficient of determination of the predictions for ``X`` against ``y`` (``1 - SSE / SST``; 1.0 for a constant ``y`` fitted exactly and 0.0 otherwise, as scikit-learn returns)."""
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
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
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
        A = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
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


@njit(cache=True, nogil=True)
def _logistic_loss_and_residual(z: np.ndarray, y: np.ndarray) -> tuple:
    """``(sum_i log(1 + e^z_i) - y_i z_i, sigmoid(z) - y)`` in one pass over the scores (the numpy form made four temporaries of the row count and two exponentials per row)."""
    n = z.shape[0]
    r = np.empty(n)
    total = 0.0
    for i in range(n):
        zi = z[i]
        e = np.exp(-abs(zi))
        total += max(zi, 0.0) + np.log1p(e) - y[i] * zi
        mu = 1.0 / (1.0 + e) if zi >= 0.0 else e / (1.0 + e)
        r[i] = mu - y[i]
    return total, r


_LBFGS_MIN_WORK = 1e8
"""``n p^2`` from which the binary fit goes to L-BFGS: the Newton iteration forms a (p, p) Hessian from all rows several times (measured 649 ms against 389 ms at 100000 x 50, 263 against 100 at 30000 x 100, 2040 against 670 at 100000 x 200), while below it the Newton step wins by a wide margin."""


class ScaledLogistic:
    """Standardise, then L2 logistic regression with inverse strength ``C`` and an unpenalised intercept (labels may be any sortable values); ``max_iter`` scales the iteration budget of the minimiser."""

    def __init__(self, C: float = 1.0, max_iter: int = 200) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScaledLogistic":
        """Fit on the rows of ``X``; returns ``self``."""
        self.mean_, self.scale_ = _standardise_fit(X)
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
        A = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        self.classes_, codes = np.unique(np.asarray(y).ravel(), return_inverse=True)
        k = self.classes_.size
        if k == 2 and float(A.shape[0]) * float(A.shape[1]) ** 2 < _LBFGS_MIN_WORK:
            from ._fe_accuracy_gate_newton import fit_logistic_ridge_newton

            # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
            fit = fit_logistic_ridge_newton(A, codes.astype(np.float64), self.C)
            if fit is not None:
                w, b = fit
                # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
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
                total, r = _logistic_loss_and_residual(z, codes)
                loss = total + 0.5 * float(w @ w) / self.C
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
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
        z = ((np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_) @ self.coef_.T + self.intercept_
        return np.asarray(z.ravel() if self.coef_.shape[0] == 1 else z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, columns ordered as ``classes_``."""
        z = self.decision_function(X)
        if z.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-z))
            return np.column_stack([1.0 - p1, p1])
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return np.asarray(e / e.sum(axis=1, keepdims=True))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Most probable class labels."""
        return np.asarray(self.classes_[np.argmax(self.predict_proba(X), axis=1)])


class PlainRidge:
    """Ridge regression on the columns as they are (no standardisation) with penalty ``alpha`` and an unpenalised intercept: ``Ridge(alpha)`` of scikit-learn, with ``coef_``, ``intercept_`` and ``predict``."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PlainRidge":
        """Fit on the rows of ``X``; returns ``self``."""
        from ._fe_probe_ridge import fit_ridge_closed_form

        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
        A = np.asarray(X, dtype=np.float64)
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predictions for the rows of ``X``."""
        # Intentional float64: closed-form solves and class probabilities of the probes: normal equations square the conditioning of the columns
        return np.asarray(np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_)


class RffSampler:
    """Random Fourier features of the RBF kernel, the draws and the map of scikit-learn's ``RBFSampler(gamma, n_components, random_state)``.

    ``random_weights_`` (``(n_features, n_components)``) and ``random_offset_`` come from ``RandomState(random_state)`` in the estimator's order (normal weights scaled by ``sqrt(2 gamma)``, then the uniform
    offsets); the map is ``sqrt(2 / n_components) cos(X W + b)`` in the dtype of ``X`` for float32 and double."""

    def __init__(self, gamma: float = 1.0, n_components: int = 100, random_state: int = 0) -> None:
        self.gamma = float(gamma)
        self.n_components = int(n_components)
        self.random_state = int(random_state)

    def fit(self, X: np.ndarray, y: object = None) -> "RffSampler":
        """Draw the weights and offsets for the number of columns of ``X``; returns ``self``."""
        rs = np.random.RandomState(self.random_state)
        n_features = shape_of(X)[1]
        self.random_weights_ = (2.0 * self.gamma) ** 0.5 * rs.normal(size=(n_features, self.n_components))
        self.random_offset_ = rs.uniform(0, 2 * np.pi, size=self.n_components)
        if np.asarray(X).dtype == np.float32:
            self.random_weights_ = self.random_weights_.astype(np.float32)
            self.random_offset_ = self.random_offset_.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The feature map of the rows of ``X``."""
        proj = np.asarray(X) @ self.random_weights_
        proj += self.random_offset_
        np.cos(proj, proj)
        proj *= (2.0 / self.n_components) ** 0.5
        return np.asarray(proj)

    def fit_transform(self, X: np.ndarray, y: object = None) -> np.ndarray:
        """``fit`` then ``transform`` on the same rows."""
        return self.fit(X).transform(X)


def cross_val_r2(make_model: Callable[[], Any], X: np.ndarray, y: np.ndarray, splits: "list[tuple[np.ndarray, np.ndarray]]") -> "Optional[float]":
    """Mean R^2 of ``make_model().fit(train).score(validation)`` over ``splits`` (a list of ``(train_idx, validation_idx)``); ``None`` without splits."""
    if not splits:
        return None
    scores = [make_model().fit(X[tr], y[tr]).score(X[va], y[va]) for tr, va in splits]
    return float(np.mean(scores))
