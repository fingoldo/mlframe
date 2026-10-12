"""Does the composite pre-filter ``mi_gain = MI(T, X_no_base) - MI(y, X_no_base)`` predict the out-of-sample gain of a composite target?

``training/composite`` screens ``T = y - base`` (or ``y / base``) by ``mi_gain``, the mean over the other columns of the MARGINAL binned MI with the target. The production default
``eps_mi_gain = -10`` switches that gate off because a correct composite was seen with a very negative gain. This script measures, on processes where the truth is known, the gain
the gate computes next to what the composite really does to the held-out error (RMSE on the y scale) of a ridge and a gradient-boosting model, with three baselines:

* ``raw``: the model on X (the base included when it is a column of X);
* ``+base``: the model on X plus the base as one more column (what a feature-side step gives a model for free);
* ``comp``: ``base + model(T)`` (offset 1), and ``comp_ols``: ``c * base + model(T_c)`` with c fitted by OLS.

Usage: ``python -m mlframe.feature_selection._benchmarks.target_lens.exp_composite_mi_gain [seeds] [n]``.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")

N_BINS = 16
MODELS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "hgb": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0),
}


def _ar1(n, rng):
    e = rng.standard_normal(n + 6)
    s = np.zeros(n + 6)
    for t in range(1, n + 6):
        s[t] = 0.9 * s[t - 1] + e[t]
    s = s[6:]
    lag = lambda k: np.concatenate([np.zeros(k), s[:-k]])  # noqa: E731
    X = np.column_stack([lag(1), lag(2), lag(3), lag(4), rng.standard_normal(n), rng.standard_normal(n)])
    return X, 0, s, "y = AR(1); base = lag1, X = lags 2-4 + 2 noise columns"


def _additive_independent(n, rng):
    base, x1, x2, x3, nz = (rng.standard_normal(n) for _ in range(5))
    base = 5.0 * base
    y = base + x1 * x2 + np.sin(2 * x3) + 0.3 * rng.standard_normal(n)
    return np.column_stack([base, x1, x2, x3, nz]), 0, y, "y = base + x1*x2 + sin(2 x3), base independent of X (variance 25)"


def _multiplicative(n, rng):
    base = np.exp(rng.standard_normal(n))
    x1, x2, nz = (rng.standard_normal(n) for _ in range(3))
    y = base * np.exp(0.4 * x1 + 0.3 * x2) * np.exp(0.1 * rng.standard_normal(n))
    return np.column_stack([base, x1, x2, nz]), 0, y, "y = base * exp(0.4 x1 + 0.3 x2) (lognormal base)"


def _useless_base(n, rng):
    base = 5.0 * rng.standard_normal(n)
    x1, x2, nz = (rng.standard_normal(n) for _ in range(3))
    y = x1 * x2 + np.sin(2 * x1) + 0.3 * rng.standard_normal(n)
    return np.column_stack([base, x1, x2, nz]), 0, y, "control: y = x1*x2 + sin(2 x1), base is pure noise"


def _correlated_base(n, rng):
    x1, x2, x3, nz = (rng.standard_normal(n) for _ in range(4))
    base = 3.0 * x1 + rng.standard_normal(n)  # base shares x1 with X
    y = base + x2 * x3 + 0.3 * rng.standard_normal(n)
    return np.column_stack([base, x1, x2, x3, nz]), 0, y, "y = base + x2*x3, base = 3 x1 + noise (base correlated with a column of X)"


def _feature_transform_base(n, rng):
    x1, x2, x3, nz = (rng.standard_normal(n) for _ in range(4))
    base = 8.0 * np.sin(3 * x1)  # a transform of the column x1, NOT a column
    y = base + x2 * x3 + 0.3 * rng.standard_normal(n)
    return np.column_stack([x1, x2, x3, nz]), base, y, "y = 8 sin(3 x1) + x2*x3; base = 8 sin(3 x1), a transform of a column"


PROCESSES = {
    "ar1": _ar1,
    "additive_independent": _additive_independent,
    "multiplicative": _multiplicative,
    "useless_base": _useless_base,
    "correlated_base": _correlated_base,
    "feature_transform_base": _feature_transform_base,
}


def _mi_gain(X_no_base: np.ndarray, y: np.ndarray, T: np.ndarray) -> float:
    """The production definition: mean over the columns of the marginal binned MI with T minus the same with y."""
    from mlframe.training.composite.discovery.screening import _mi_to_target

    kw = dict(n_neighbors=3, random_state=0, estimator="bin", nbins=N_BINS, aggregation="mean")
    return float(_mi_to_target(X_no_base, T, **kw) - _mi_to_target(X_no_base, y, **kw))


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _fit_predict(make, Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return make().fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


def one(process: str, n: int, seed: int) -> dict:
    """One draw: ``mi_gain`` and the held-out RMSE of each arm for each model."""
    rng = np.random.default_rng(seed)
    X, base, y, _ = PROCESSES[process](n, rng)
    base_col = isinstance(base, (int, np.integer))
    b = X[:, base] if base_col else np.asarray(base)
    X_no_base = np.delete(X, base, axis=1) if base_col else X
    half = n // 2
    tr, te = np.arange(half), np.arange(half, n)
    multiplicative = process == "multiplicative"
    T = np.log(y / b) if multiplicative else y - b
    from mlframe.training.composite.discovery.screening import _mi_pair_bin

    # the same gain with every column that carries the base removed too: its binned MI with the base exceeds the maximum over 20 permutations of the base (a calibrated null, no fixed bar)
    perm_rng = np.random.default_rng(seed + 1)
    keep = []
    for j in range(X_no_base.shape[1]):
        mi_j = _mi_pair_bin(X_no_base[:, j], b, nbins=N_BINS)
        null = max(_mi_pair_bin(X_no_base[:, j], perm_rng.permutation(b), nbins=N_BINS) for _ in range(20))
        if mi_j <= null:
            keep.append(j)
    out = {"mi_gain": _mi_gain(X_no_base, y, T), "mi_gain_excl": _mi_gain(X_no_base[:, keep], y, T) if keep else float("nan"), "n_kept": len(keep)}
    c = float(np.dot(b[tr] - b[tr].mean(), y[tr] - y[tr].mean()) / np.dot(b[tr] - b[tr].mean(), b[tr] - b[tr].mean()))
    for name, make in MODELS.items():
        Xb = X if base_col else np.column_stack([X, b])
        raw = _fit_predict(make, X[tr], y[tr], X[te])
        plus = _fit_predict(make, Xb[tr], y[tr], Xb[te])
        if multiplicative:
            comp = b[te] * np.exp(_fit_predict(make, X_no_base[tr], T[tr], X_no_base[te]))
            comp_ols = comp
        else:
            comp = b[te] + _fit_predict(make, X_no_base[tr], T[tr], X_no_base[te])
            Tc = y - c * b
            comp_ols = c * b[te] + _fit_predict(make, X_no_base[tr], Tc[tr], X_no_base[te])
        out[name] = {"raw": _rmse(raw, y[te]), "+base": _rmse(plus, y[te]), "comp": _rmse(comp, y[te]), "comp_ols": _rmse(comp_ols, y[te])}
    return out


def main() -> None:
    """Run every process and print the table."""
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    print(f"n={n}, {seeds} seeds, held-out RMSE on the y scale (median over seeds); mi_gain as in training/composite (mean marginal binned MI, base excluded)")
    for process in PROCESSES:
        runs = [one(process, n, 100 + s) for s in range(seeds)]
        gain = float(np.median([r["mi_gain"] for r in runs]))
        gain_excl = float(np.nanmedian([r["mi_gain_excl"] for r in runs]))
        print(f"\n{process}: {PROCESSES[process](50, np.random.default_rng(0))[3]}\n  mi_gain = {gain:+.4f}   base-tracking columns removed: {gain_excl:+.4f} ({runs[0]['n_kept']} columns kept)")
        for name in MODELS:
            med = {k: float(np.median([r[name][k] for r in runs])) for k in ("raw", "+base", "comp", "comp_ols")}
            best_plain = min(med["raw"], med["+base"])
            print(f"  {name:5} raw {med['raw']:.3f}  +base {med['+base']:.3f}  comp {med['comp']:.3f}  comp_ols {med['comp_ols']:.3f}   comp vs best plain: {med['comp'] / best_plain - 1:+.1%}")


if __name__ == "__main__":
    main()
