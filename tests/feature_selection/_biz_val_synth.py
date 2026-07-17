"""Reusable synthetic data generators + helpers for biz_val tests.

Per CLAUDE.md "Every new ML trick gets a biz_val synthetic test":
to keep ~50+ biz_val tests fast + maintainable, the synthetic
generators they all share live HERE. Each generator is deterministic
(fixed seed), small (n=500-2000), and produces a target where a
specific structural feature is present.

Generators come with docstrings + doctests so the test file's intent
stays readable.

Usage::

    from tests.feature_selection._biz_val_synth import (
        make_signal_plus_noise, make_correlated_redundant, make_3way_xor,
        as_df, support_indices,
    )

    def test_biz_val_my_thing():
        df, y, signal = make_signal_plus_noise(n=1500, p_signal=3, p_noise=10)
        ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Core generators
# ---------------------------------------------------------------------------


def make_signal_plus_noise(n: int = 2000, p_signal: int = 3, p_noise: int = 10, seed: int = 42, linear_only: bool = True):
    """Linear binary target with ``p_signal`` true features + ``p_noise``
    pure-noise. ``y = sign(sum(X_signal) + 0.3*noise)``.

    Returns ``(X, y, signal_indices)`` where signal_indices = [0..p_signal-1].

    >>> X, y, sig = make_signal_plus_noise(n=200, p_signal=2, p_noise=3)
    >>> X.shape
    (200, 5)
    >>> sorted(sig)
    [0, 1]
    >>> set(np.unique(y).tolist()).issubset({0, 1})
    True
    """
    rng = np.random.default_rng(seed)
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    if linear_only:
        score = X_sig.sum(axis=1) + 0.3 * rng.normal(size=n)
    else:
        score = X_sig[:, 0] ** 2 - X_sig[:, 1] ** 2 + 0.3 * rng.normal(size=n)
    y = (score > 0).astype(np.int64)
    return X, y, list(range(p_signal))


def make_correlated_redundant(n: int = 2000, n_corr: int = 4, p_noise: int = 5, corr: float = 0.95, seed: int = 42):
    """``n_corr`` features that share a base + 1 unique informative + ``p_noise``.
    Target depends on the unique informative AND one cluster member.

    Returns ``(X, y, unique_idx)`` where unique_idx is the index of the
    one truly orthogonal informative feature.

    >>> X, y, uniq = make_correlated_redundant(n=400, n_corr=3, p_noise=2)
    >>> X.shape
    (400, 6)
    >>> uniq
    3
    """
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    noise_scale = float(np.sqrt(1 - corr**2) / max(corr, 1e-9))
    X_corr = np.column_stack([base + noise_scale * rng.normal(size=n) for _ in range(n_corr)])
    unique = rng.normal(size=(n, 1))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_corr, unique, X_noise])
    y = (X_corr[:, 0] + unique[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    return X, y, n_corr


def make_3way_xor(n: int = 2000, p: int = 10, seed: int = 42):
    """3-way XOR: ``y = sign(x_0 * x_1 * x_2)``. All individual and pair
    MIs ~0; only 3-way joint reveals signal.

    Returns ``(X, y, signal_indices=[0, 1, 2])``.

    >>> X, y, sig = make_3way_xor(n=400, p=5)
    >>> X.shape
    (400, 5)
    >>> sig
    [0, 1, 2]
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (np.sign(X[:, 0] * X[:, 1] * X[:, 2]) > 0).astype(np.int64)
    return X, y, [0, 1, 2]


def make_polynomial_target(n: int = 2000, seed: int = 42, degree: int = 2):
    """``y = sign(0.7*x_a^d - 0.5*x_b^d + 0.3*x_a*x_b)`` for degree d.

    Useful for testing FE / polynomial-pair search. Signal is in
    (x_a, x_b) pair only; remaining columns are noise.

    >>> X, y, sig = make_polynomial_target(n=300, degree=2)
    >>> X.shape
    (300, 8)
    >>> sig
    [0, 1]
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    score = 0.7 * X[:, 0] ** degree - 0.5 * X[:, 1] ** degree + 0.3 * X[:, 0] * X[:, 1]
    y = (score > np.median(score)).astype(np.int64)
    return X, y, [0, 1]


def make_imbalanced(n: int = 2000, imbalance: float = 0.05, p_signal: int = 3, p_noise: int = 8, seed: int = 42):
    """Class-imbalanced binary target. ``imbalance`` is the fraction
    of class-1 (default 5%).

    >>> X, y, sig = make_imbalanced(n=400, imbalance=0.1)
    >>> X.shape
    (400, 11)
    >>> bool(abs(y.mean() - 0.1) < 0.03)
    True
    """
    rng = np.random.default_rng(seed)
    X_sig = rng.normal(size=(n, p_signal))
    X_noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([X_sig, X_noise])
    score = X_sig.sum(axis=1) + 0.3 * rng.normal(size=n)
    threshold = float(np.quantile(score, 1.0 - imbalance))
    y = (score > threshold).astype(np.int64)
    return X, y, list(range(p_signal))


def make_heavy_tail_skewed(n: int = 2000, p_noise: int = 5, seed: int = 42):
    """Heavy-tail lognormal inputs with log-multiplicative target:
    ``y = sign(log(base) + log(other) > median)``. Plug-in MI on
    raw inputs UNDER-estimates the relationship; log-aware estimators
    (or trees) recover full signal.

    >>> X, y, sig = make_heavy_tail_skewed(n=300)
    >>> X.shape
    (300, 7)
    >>> sig
    [0, 1]
    >>> bool(X[:, 0].min() > 0)  # base must be positive (lognormal)
    True
    """
    rng = np.random.default_rng(seed)
    base = np.exp(rng.normal(size=n))
    other = np.exp(rng.normal(size=n))
    noise = rng.normal(size=(n, p_noise))
    X = np.column_stack([base, other, noise])
    score = np.log(base) + np.log(other)
    y = (score > np.median(score)).astype(np.int64)
    return X, y, [0, 1]


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def make_latent_reflections(
    n: int = 4000,
    loadings=(1.0, 1.0, 1.0, 1.0),
    noise_sd=(0.7, 0.7, 0.7, 0.7),
    n_noise: int = 3,
    indep_weight: float = 0.4,
    seed: int = 42,
    shared_noise: float = 0.0,
    distinct_sd: float = 0.0,
):
    """Hidden factor ``z`` reflected in several observed columns ``A_i = loadings_i*z + noise_i`` plus an
    independent signal and pure-noise columns. ``y = sign(z + indep_weight*indep)``.

    Knobs span the biz_val scenario matrix:
    - homoscedastic equal loadings -> S1 (mean is the MLE / BLUE);
    - heterogeneous ``loadings`` -> S2 (PCA wins);
    - heterogeneous ``noise_sd`` -> S3 (inverse-variance / factor-score win);
    - ``distinct_sd>0`` adds a per-reflection distinct signal delta_i that also drives y -> S4
      (aggregation destroys delta -> the unidimensionality gate / supervised gate must protect);
    - ``shared_noise>0`` makes the reflection noise CORRELATED -> S5 (averaging can't denoise -> reject).

    Returns ``(X, y, info)`` with ``info = {"reflections": [0..k-1], "indep": k, "z": z_true}``.
    """
    rng = np.random.default_rng(seed)
    k = len(loadings)
    z = rng.normal(size=n)
    eps_shared = rng.normal(size=n)
    refl_cols = []
    for i in range(k):
        eps = noise_sd[i] * ((1.0 - shared_noise) * rng.normal(size=n) + shared_noise * eps_shared)
        col = loadings[i] * z + eps
        if distinct_sd > 0:
            col = col + distinct_sd * rng.normal(size=n)  # delta_i carried into the column
        refl_cols.append(col)
    indep = rng.normal(size=n)
    score = z + indep_weight * indep
    if distinct_sd > 0:
        # delta_i also drives y so averaging it away costs predictive info (the dangerous S4 case).
        score = score + distinct_sd * sum((c - loadings[i] * z) for i, c in enumerate(refl_cols))
    y = (score > np.median(score)).astype(np.int64)
    X = np.column_stack(refl_cols + [indep] + [rng.normal(size=n) for _ in range(n_noise)])
    info = {"reflections": list(range(k)), "indep": k, "z": z}
    return X, y, info


def make_two_latent_groups(n: int = 6000, k1: int = 4, k2: int = 4, noise: float = 0.85, n_noise: int = 3, seed: int = 42):
    """TWO independent hidden factors z1, z2, each reflected in its own group of noisy columns, plus a
    pure-noise group. ``y = sign(z1 + z2 + small noise)`` (both factors drive the target).

    Within-group columns correlate (~``1/(1+noise^2)``); cross-group correlation is ~0; noise columns
    are independent. A correct clusterer must find exactly TWO clusters (one per latent factor) and
    NOT merge them or pull in noise. Returns ``(X, y, info)`` with
    ``info = {"groupA": [..], "groupB": [..], "noise": [..], "z1": .., "z2": ..}``.
    """
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    groupA = [z1 + noise * rng.normal(size=n) for _ in range(k1)]
    groupB = [z2 + noise * rng.normal(size=n) for _ in range(k2)]
    noise_cols = [rng.normal(size=n) for _ in range(n_noise)]
    score = z1 + z2 + 0.3 * rng.normal(size=n)
    y = (score > np.median(score)).astype(np.int64)
    X = np.column_stack(groupA + groupB + noise_cols)
    info = {
        "groupA": list(range(k1)),
        "groupB": list(range(k1, k1 + k2)),
        "noise": list(range(k1 + k2, k1 + k2 + n_noise)),
        "z1": z1,
        "z2": z2,
    }
    return X, y, info


def as_df(X: np.ndarray, y: np.ndarray, prefix: str = "x"):
    """Wrap a numpy ``(X, y)`` pair as ``(pd.DataFrame, pd.Series)``.

    >>> X = np.zeros((3, 2)); y = np.array([0, 1, 1])
    >>> df, ser = as_df(X, y, prefix="feat")
    >>> list(df.columns)
    ['feat0', 'feat1']
    >>> ser.name
    'y'
    """
    cols = [f"{prefix}{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="y")


def support_indices(sel):
    """Return support_ as integer indices regardless of whether the
    selector exposes a boolean mask, integer array, or list. Works for
    both ``sklearn.RFE``-style and ``mlframe.MRMR``-style selectors.

    >>> import numpy as np
    >>> class Sel: pass
    >>> s = Sel(); s.support_ = np.array([True, False, True])
    >>> support_indices(s)
    [0, 2]
    >>> s.support_ = np.array([0, 2])
    >>> support_indices(s)
    [0, 2]
    """
    s = sel.support_ if hasattr(sel, "support_") else sel
    arr = np.asarray(s)
    if arr.dtype == bool:
        return [int(i) for i in np.flatnonzero(arr)]
    return [int(i) for i in arr]


def signal_overlap(sel, signal: list, top_k: Optional[int] = None) -> int:
    """Count how many signal-feature indices appear in ``sel.support_``.
    If ``top_k`` given, restrict to the first ``top_k`` of the support.

    >>> import numpy as np
    >>> class Sel: pass
    >>> s = Sel(); s.support_ = np.array([0, 1, 5, 7])
    >>> signal_overlap(s, [0, 1, 2])
    2
    >>> signal_overlap(s, [0, 1, 2], top_k=1)
    1
    """
    idx = support_indices(sel)
    if top_k is not None:
        idx = idx[:top_k]
    return len(set(idx) & set(int(i) for i in signal))


import re as _re

_XREF = _re.compile(r"x(\d+)")


def signal_recovery_count(sel, signal: list, top_k: Optional[int] = None, prefix: str = "x") -> int:
    """Count distinct signal columns RECOVERED by a fitted selector,
    crediting engineered features that reference a signal column.

    Full-mode MRMR (``use_simple_mode=False``, the default) returns a
    COMPACT, de-duplicated set: it routinely replaces redundant raw
    signal columns with a single engineered combination of them, e.g.
    on ``y=sign(x0+x1+x2)`` it keeps ``{x1, add(x0,x2), ...}`` rather
    than all three raw ``x0,x1,x2``. ``signal_overlap`` (which only
    looks at the raw integer ``support_`` indices) therefore
    UNDER-counts recovery under the new default.

    This helper reads ``get_feature_names_out()`` and credits a signal
    column ``i`` as recovered if ``x{i}`` appears literally in ANY
    selected feature name (raw or engineered such as ``add(x0,x2)``).
    A column engineered into a survivor is genuinely "used", so this is
    a faithful recovery metric for the de-duplicated regime, not a
    weakening. ``top_k`` (when given) restricts to the first ``top_k``
    selected features, mirroring ``signal_overlap``'s top-k semantics.

    Falls back to the raw-index overlap when the selector lacks
    ``get_feature_names_out``.

    >>> import numpy as np
    >>> class Sel:
    ...     def get_feature_names_out(self):
    ...         return ['x1', 'add(x0,x2)', 'x7']
    >>> signal_recovery_count(Sel(), [0, 1, 2])
    3
    >>> signal_recovery_count(Sel(), [0, 1, 2], top_k=1)
    1
    """
    names = None
    if hasattr(sel, "get_feature_names_out"):
        try:
            names = [str(nm) for nm in sel.get_feature_names_out()]
        except Exception:
            names = None
    if names is None:
        return signal_overlap(sel, signal, top_k=top_k)
    if top_k is not None:
        names = names[:top_k]
    sig = set(int(i) for i in signal)
    pref = prefix
    recovered: set = set()
    for nm in names:
        # Only count column refs that use this generator's prefix (e.g. "x3").
        if pref != "x":
            refs = set(int(m.group(1)) for m in _re.finditer(pref + r"(\d+)", nm))
        else:
            refs = set(int(m) for m in _XREF.findall(nm))
        recovered |= refs & sig
    return len(recovered)


def downstream_auc(sel, df, ys, cv: int = 5) -> float:
    """5-fold ``LogisticRegression`` ``roc_auc`` on ``sel.transform(df)``.

    The honest, model-facing measure of whether a selected feature set
    (raw + engineered) carries the signal. Used to re-baseline
    membership assertions that broke when full-mode de-duplication
    swapped raw signal columns for engineered combinations: a selection
    is "as good as" the all-signal baseline iff this AUC is within a
    small band of the baseline AUC, regardless of which exact columns
    survived. Returns ``nan`` if the selection is empty.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    Xt = sel.transform(df)
    if getattr(Xt, "shape", (0, 0))[1] == 0:
        return float("nan")
    return float(
        cross_val_score(
            LogisticRegression(max_iter=400),
            Xt,
            ys,
            cv=cv,
            scoring="roc_auc",
        ).mean()
    )


def baseline_signal_auc(df, ys, signal: list, prefix: str = "x", cv: int = 5) -> float:
    """5-fold ``LogisticRegression`` ``roc_auc`` using ONLY the raw
    signal columns -- the "all-signal" reference a de-duplicated
    selection is compared against."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    cols = [f"{prefix}{i}" for i in signal]
    return float(
        cross_val_score(
            LogisticRegression(max_iter=400),
            df[cols],
            ys,
            cv=cv,
            scoring="roc_auc",
        ).mean()
    )


# ---------------------------------------------------------------------------
# Canonical MRMR-layer dataset builders + train/eval helpers
#
# These are the most-duplicated module-level ``_build_*`` / ``_train_*`` /
# ``_logreg_auc`` / ``_quantile_bin_local`` / ``_mi_one`` helpers copied across
# the ``test_biz_value_mrmr_layer*.py`` suite. Hosted here so a dataset bug is
# fixed in ONE place; per-file deltas (n, n_noise) are explicit kwargs.
# ---------------------------------------------------------------------------


def _build_linear(seed: int, n: int = 1500):
    """Plain linear-additive signal for the default-disabled contract."""
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise_a": rng.standard_normal(n),
            "noise_b": rng.standard_normal(n),
            "noise_c": rng.standard_normal(n),
        }
    )
    y = ((x1 + 0.7 * x2) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _build_quadratic_classif(seed: int, n: int = 1500, n_noise: int = 5):
    """y = sign(x1^2 - 1). Clean He_2 signal -- a stable winner appended
    regardless of estimator, used for enable / pickle contracts."""
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    cols: dict = {"x1": x1}
    for k in range(n_noise):
        cols[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    y = ((x1**2 + 0.1 * rng.standard_normal(n)) > 1.0).astype(int)
    return X, pd.Series(y, name="y")


def _build_redundant_multi(seed: int, n: int = 2000):
    """``x1`` carries a primary quadratic signal; ``x_dup_a/b/c`` are near-copies
    of ``x1`` (jointly redundant); ``x2`` carries an independent secondary
    quadratic signal. TC-uplift against ``[x1]`` collapses the duplicates and
    surfaces ``x2__He2`` -- the column with genuine new joint information."""
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x_dup_a = x1 + 0.05 * rng.standard_normal(n)
    x_dup_b = x1 + 0.05 * rng.standard_normal(n)
    x_dup_c = x1 + 0.05 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x_dup_a": x_dup_a,
            "x_dup_b": x_dup_b,
            "x_dup_c": x_dup_c,
            "x2": x2,
            "noise_0": rng.standard_normal(n),
        }
    )
    signal = x1**2 + 0.6 * (x2**2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


def _build_xor_redundant(seed: int, n: int = 2000):
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x_dup_a = x1 + 0.05 * rng.standard_normal(n)
    x_dup_b = x1 + 0.05 * rng.standard_normal(n)
    x_dup_c = x1 + 0.05 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x_dup_a": x_dup_a,
            "x_dup_b": x_dup_b,
            "x_dup_c": x_dup_c,
            "x2": x2,
            "noise_0": rng.standard_normal(n),
        }
    )
    signal = x1**2 + 0.6 * (x2**2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


def _train_holdout_split(X: pd.DataFrame, y: pd.Series, *, train_frac: float = 0.6, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(train_frac * len(X))
    tr, ho = idx[:cut], idx[cut:]
    return (
        X.iloc[tr].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True),
        X.iloc[ho].reset_index(drop=True),
        y.iloc[ho].reset_index(drop=True),
    )


def _logreg_auc(X_tr: pd.DataFrame, y_tr: pd.Series, X_ho: pd.DataFrame, y_ho: pd.Series) -> float:
    """LogReg AUC on numeric-only columns of (X_tr -> X_ho). Object cols are
    dropped -- the baseline is "what LogReg can do without a TE step"."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    num_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    if not num_cols:
        return 0.5
    Xn_tr = X_tr[num_cols].to_numpy(dtype=np.float64)
    Xn_ho = X_ho[num_cols].to_numpy(dtype=np.float64)
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(Xn_tr, y_tr.to_numpy())
    proba = clf.predict_proba(Xn_ho)[:, 1]
    return float(roc_auc_score(y_ho.to_numpy(), proba))


def _quantile_bin_local(arr: np.ndarray, nbins: int = 10) -> np.ndarray:
    """Local helper for the diversity test (independent of the prod path)."""
    a = np.asarray(arr, dtype=np.float64)
    finite_mask = np.isfinite(a)
    out = np.zeros(a.size, dtype=np.int64)
    if not finite_mask.any():
        return out
    finite = a[finite_mask]
    qs = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.unique(np.quantile(finite, qs))
    if edges.size <= 2:
        if edges.size == 2:
            out[finite_mask] = (a[finite_mask] >= edges[1]).astype(np.int64)
        return out
    inner = edges[1:-1]
    out[finite_mask] = np.searchsorted(inner, finite, side="right").astype(np.int64)
    return out


def _mi_one(col: np.ndarray, y: np.ndarray, nbins: int = 10) -> float:
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        _mi_classif_batch,
    )

    arr = np.asarray(col, dtype=np.float64).reshape(-1, 1)
    return float(_mi_classif_batch(arr, np.asarray(y).astype(np.int64), nbins=nbins)[0])
