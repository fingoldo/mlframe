"""Benchmark robust regression on 1M rows for ``linear_residual`` fit alternatives.

User constraint: pick the fastest robust variant and ship ONLY that one (plus
plain OLS) as a registered transform. Slow methods (typically Theil-Sen at
O(n^2)) must NOT enter the production registry.

Tested variants:
- ``ols_lstsq``: ``np.linalg.lstsq`` (current ``_linear_residual_fit`` baseline).
- ``ols_normal``: closed-form normal equations (``np.linalg.solve`` on (X^T X)).
- ``trimmed_ls``: OLS, drop rows where |residual| > 3 * MAD, refit OLS. Two-pass.
- ``huber_irls``: sklearn ``HuberRegressor`` (IRLS, ``epsilon=1.35``, ``max_iter=50``).
- ``ransac``: sklearn ``RANSACRegressor`` (``n_trials=50``).
- ``theil_sen``: sklearn ``TheilSenRegressor`` (random subsets, ``max_subpopulation=1e4``).
- ``lad_quantreg``: ``statsmodels.api.QuantReg`` at q=0.5 (LAD / L1).

Reports wall time on 1M rows, 2 features (univariate base + intercept).
Outputs decision: winner = fastest method whose alpha estimate is within
robust-tolerance (5%) of the OLS-on-clean-data ground truth.
"""
from __future__ import annotations

import time
from typing import Callable, Dict

import numpy as np


def _make_data(
    n: int = 1_000_000,
    outlier_frac: float = 0.05,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Returns ``(X_clean, X_dirty, y, alpha_true, beta_true)`` where dirty has ``outlier_frac`` rows replaced with Cauchy-like noise."""
    rng = np.random.default_rng(seed)
    base = rng.normal(11500.0, 600.0, size=n)
    alpha_true = 0.85
    beta_true = 50.0
    eps = rng.normal(0.0, 5.0, size=n)
    y_clean = alpha_true * base + beta_true + eps
    y = y_clean.copy()
    n_out = int(n * outlier_frac)
    idx = rng.choice(n, size=n_out, replace=False)
    # Heavy outliers scaled to dominate OLS.
    y[idx] += rng.standard_cauchy(n_out) * 200.0
    X_dirty = np.column_stack([base, np.ones(n)])
    X_clean = X_dirty
    return X_clean, X_dirty, y, alpha_true, beta_true


# ---------- regressors ----------


def _ols_lstsq(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _ols_normal(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    XtX = X.T @ X
    Xty = X.T @ y
    coef = np.linalg.solve(XtX, Xty)
    return float(coef[0]), float(coef[1])


def _trimmed_ls(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    mad = float(np.median(np.abs(resid - np.median(resid))))
    keep = np.abs(resid) <= 3.0 * (mad * 1.4826 + 1e-12)
    if keep.sum() < max(10, int(0.5 * len(y))):
        return float(coef[0]), float(coef[1])
    coef2, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    return float(coef2[0]), float(coef2[1])


def _huber_irls(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import HuberRegressor
    # X has intercept column; HuberRegressor adds its own.
    model = HuberRegressor(epsilon=1.35, max_iter=50, alpha=0.0).fit(X[:, :1], y)
    return float(model.coef_[0]), float(model.intercept_)


def _ransac(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import LinearRegression, RANSACRegressor
    model = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=0.1,
        max_trials=50,
        random_state=0,
    ).fit(X[:, :1], y)
    return float(model.estimator_.coef_[0]), float(model.estimator_.intercept_)


def _theil_sen(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.linear_model import TheilSenRegressor
    model = TheilSenRegressor(
        max_subpopulation=10_000,
        n_subsamples=None,
        max_iter=50,
        random_state=0,
        n_jobs=-1,
    ).fit(X[:, :1], y)
    return float(model.coef_[0]), float(model.intercept_)


def _lad_quantreg(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    import statsmodels.api as sm
    model = sm.QuantReg(y, X).fit(q=0.5, max_iter=200)
    return float(model.params[0]), float(model.params[1])


REGRESSORS: Dict[str, Callable[[np.ndarray, np.ndarray], tuple[float, float]]] = {
    "ols_lstsq":   _ols_lstsq,
    "ols_normal":  _ols_normal,
    "trimmed_ls":  _trimmed_ls,
    "huber_irls":  _huber_irls,
    "ransac":      _ransac,
    "theil_sen":   _theil_sen,
    "lad_quantreg": _lad_quantreg,
}


# ---------- bench harness ----------


def time_one(fn, X, y, *, repeats: int = 3, timeout_s: float = 120.0) -> tuple[float, float, float]:
    """Returns ``(best_secs, alpha, beta)``. Bails out and returns ``(inf, nan, nan)`` after timeout."""
    times: list[float] = []
    alpha = float("nan")
    beta = float("nan")
    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            alpha, beta = fn(X, y)
        except Exception as exc:
            return float("inf"), float("nan"), float("nan")
        dt = time.perf_counter() - t0
        times.append(dt)
        if dt > timeout_s:
            return dt, alpha, beta
    return min(times), alpha, beta


def main() -> None:
    n = 1_000_000
    print(f"Synthetic: n={n:,}, 5% outliers, alpha_true=0.85, beta_true=50.0")
    print()
    X_clean, X_dirty, y, alpha_true, beta_true = _make_data(n=n)

    rows: list[tuple[str, float, float, float, float, float]] = []
    for name, fn in REGRESSORS.items():
        # Theil-Sen is O(n^2) -> skip on 1M unless explicitly allowed.
        if name == "theil_sen":
            print(f"  {name:13s}  SKIPPED on n={n:,} (O(n^2) memory). Time it on n=10k instead.")
            t10k = _quick_theil_10k()
            rows.append((name, t10k, float("nan"), float("nan"), float("nan"), 1.0))
            continue
        secs, alpha, beta = time_one(fn, X_dirty, y, repeats=2, timeout_s=120.0)
        alpha_err = abs(alpha - alpha_true) / max(abs(alpha_true), 1e-12)
        beta_err = abs(beta - beta_true) / max(abs(beta_true), 1e-12)
        rows.append((name, secs, alpha, beta, alpha_err, beta_err))
        print(f"  {name:13s}  {secs:7.2f}s   alpha={alpha:.4f} (err {alpha_err*100:5.2f}%)   beta={beta:9.2f} (err {beta_err*100:5.2f}%)")

    print()
    print("Decision rules:")
    print("  - SKIP variants with secs > 5x fastest OLS (production budget).")
    print("  - Among survivors, RANK robust ones by alpha_err -- pick the lowest.")
    print()
    ols_secs = min(s for n_, s, *_ in rows if n_.startswith("ols")) if any(n_.startswith("ols") for n_, *_ in rows) else 1.0
    budget = 5.0 * ols_secs
    print(f"Fastest OLS = {ols_secs:.2f}s; budget = {budget:.2f}s")
    survivors = [(n_, s, a_err) for n_, s, a, b, a_err, b_err in rows if s <= budget and not np.isnan(a_err)]
    print(f"Survivors within budget: {[(n_, f'{s:.2f}s', f'{a_err*100:.2f}%') for n_, s, a_err in survivors]}")
    if survivors:
        # Best alpha among non-OLS (we already know OLS is biased by outliers).
        robust_survivors = [s for s in survivors if not s[0].startswith("ols")]
        if robust_survivors:
            winner = min(robust_survivors, key=lambda r: r[2])
            print(f"\nWINNER (lowest alpha_err): {winner[0]} ({winner[1]:.2f}s, {winner[2]*100:.2f}% alpha error)")


def _quick_theil_10k() -> float:
    """Time Theil-Sen on a 10k subsample to give an order-of-magnitude estimate."""
    X10, _, y10, *_ = _make_data(n=10_000)
    secs, _, _ = time_one(_theil_sen, X10, y10, repeats=1, timeout_s=60.0)
    return secs


if __name__ == "__main__":
    main()
