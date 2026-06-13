"""Accuracy-truth gate for engineered-feature candidates (2026-06-04).

The MI-based FE acceptance gates (``_confirm_predictor`` conditional-MI, the
``score_features_by_mi_uplift`` binned-MI uplift) are fooled by the bias
inflation of plug-in MI: a Fourier / chirp / Hermite transform of a strong raw
signal gets an *inflated* MI estimate and outranks (and evicts) the raw column
even when it adds no real predictive value. This gates candidates on the
**multivariate downstream uplift** -- does adding the engineered feature SET to
its raw source measurably improve a held-out linear probe? -- the accuracy
ground truth, not a single-feature geometry heuristic.

Applied to the engineered columns AFTER ``_run_fe_step`` returns them (i.e. the
final screening candidates), NOT inside the generators -- the generators are
reused for internal scorer routing, so filtering there corrupts the routing that
finds the genuine signal. At the post-FE-step level the routing has already run;
this only filters which engineered columns COMPETE in the screen, so a genuine
win (He2 on quadratic, the routed adaptive Fourier on an oscillation) is kept
and a hijacker (Fourier/chirp/He of a monotone / MNAR / leak column) is dropped
so it cannot evict the raw signal from support_.

Verified to separate all four cases (Hermite/Fourier x win/hijack) where a
single MI or R^2 metric cannot.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Minimum held-out score uplift (engineered over its raw source) to keep a candidate. Set just above the variance-reduced CV noise floor: with 10-fold averaging the per-column noise band is ~+-0.015, so this separates pure-noise / redundant columns (~0) from real signal while a genuine win clears it by an order of magnitude (~0.4).
_FE_UPLIFT_MIN: float = 0.015


def measure_feature_uplift(
    X_base: np.ndarray,
    X_eng: np.ndarray,
    y: np.ndarray,
    *,
    classification: bool,
    n_splits: int = 10,
    seed: int = 0,
) -> float | None:
    """Held-out uplift of a linear probe fitted on ``[X_base | X_eng]`` versus
    ``X_base`` alone. Positive => the engineered set adds generalisable signal.
    Returns the mean held-out score delta (AUC binary / accuracy multiclass /
    R^2 regression), or ``None`` when the uplift CANNOT be measured (degenerate
    input, too few rows/classes, or a probe exception). ``None`` is the fail-OPEN
    sentinel: callers must KEEP the engineered column when they cannot assess it
    (a probe that errors must not silently evict a candidate -- the documented
    contract of ``keep_engineered_over_source``). A genuine measured zero uplift
    still returns ``0.0`` (and is correctly dropped)."""
    X_base = np.asarray(X_base, dtype=np.float64)
    X_eng = np.asarray(X_eng, dtype=np.float64)
    y = np.asarray(y).ravel()
    if X_base.ndim == 1:
        X_base = X_base[:, None]
    if X_eng.ndim == 1:
        X_eng = X_eng[:, None]
    n = X_base.shape[0]
    if n != X_eng.shape[0] or n != y.size or n < 40 or X_eng.shape[1] == 0:
        return None
    yf = y.astype(np.float64, copy=False) if y.dtype.kind in ("f", "i", "u", "b") else np.zeros(n)
    finite = np.isfinite(X_base).all(1) & np.isfinite(X_eng).all(1) & np.isfinite(yf)
    if int(finite.sum()) < 40:
        return None
    X_base, X_eng, y = X_base[finite], X_eng[finite], y[finite]

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X_aug = np.concatenate([X_base, X_eng], axis=1)
    if classification:
        classes, y_enc = np.unique(y, return_inverse=True)
        if classes.size < 2:
            return None
        binary = classes.size == 2
        n_splits = min(n_splits, int(np.min(np.bincount(y_enc))))
        if n_splits < 2:
            return None
        split_iter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X_base, y_enc)
    else:
        y_enc = y.astype(np.float64)
        binary = False
        split_iter = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X_base)

    deltas = []
    try:
        for tr, va in split_iter:
            if classification and np.unique(y_enc[tr]).size < 2:
                continue

            def _score(Xfull):
                sc = StandardScaler()
                Xt = sc.fit_transform(Xfull[tr])
                Xv = sc.transform(Xfull[va])
                if classification:
                    m = LogisticRegression(max_iter=200, C=1.0)
                    m.fit(Xt, y_enc[tr])
                    if binary:
                        return roc_auc_score(y_enc[va], m.predict_proba(Xv)[:, 1])
                    return accuracy_score(y_enc[va], m.predict(Xv))
                m = Ridge(alpha=1.0)
                m.fit(Xt, y_enc[tr])
                return r2_score(y_enc[va], m.predict(Xv))

            deltas.append(_score(X_aug) - _score(X_base))
    except Exception as exc:  # pragma: no cover - the probe must never break a fit
        logger.debug("measure_feature_uplift: probe failed (%s); fail-open (None)", exc)
        return None
    if not deltas:
        return None
    return float(np.mean(deltas))


def infer_classification(y: np.ndarray) -> bool:
    """Cheap classification-vs-regression guess from y's dtype + cardinality."""
    y = np.asarray(y).ravel()
    if y.dtype.kind in ("b", "O", "U", "S"):
        return True
    finite = y[np.isfinite(y)] if y.dtype.kind == "f" else y
    if finite.size == 0:
        return True
    if y.dtype.kind in ("i", "u"):
        return np.unique(finite).size <= max(20, int(0.05 * finite.size))
    return np.unique(finite).size <= max(20, int(0.02 * finite.size))


def class_mi_fe_applicable(y: np.ndarray) -> bool:
    """True iff the MI-floor FE operators (pairwise-modular, integer-lattice, row-argmax, conditional-gate) can score a candidate against y.

    These detectors gate every candidate on plug-in class-MI (``_mi_classif_batch``), which treats y as raw discrete class labels. On a
    CONTINUOUS target the int64 cast collapses to ~n distinct classes -> the MI is inflated garbage (a pure-noise column scores >2 nats),
    the per-call cost balloons, and at high spread the kernel can even segfault; the conditional-gate's tau-grid x conditional-divergence
    sweep additionally HANGS (the late-caught regression bug). On a 2D y (multilabel / multi-target regression) the kernel cannot consume
    the label matrix and silently returns 0, so the whole sweep runs on a dead signal. Both are class-MI misuse -> skip the operator cleanly
    (emit nothing) rather than scan garbage. Applicable only when y is 1D AND classification-shaped (discrete, low-cardinality)."""
    arr = np.asarray(y)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        return False
    return infer_classification(arr)


def keep_engineered_over_source(
    src_vals: np.ndarray,
    eng_mat: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float = _FE_UPLIFT_MIN,
    max_probe_n: int = 5000,
    seed: int = 0,
) -> bool:
    """True if the engineered column SET ``eng_mat`` adds held-out downstream
    uplift over its raw source column(s) ``src_vals`` -- i.e. worth keeping
    rather than evicting the raw signal. Subsamples to ``max_probe_n`` rows.
    Fail-open (True) on degenerate input so the gate never silently drops
    everything."""
    src_vals = np.asarray(src_vals, dtype=np.float64)
    eng_mat = np.asarray(eng_mat, dtype=np.float64)
    if src_vals.ndim == 1:
        src_vals = src_vals[:, None]
    if eng_mat.ndim == 1:
        eng_mat = eng_mat[:, None]
    y = np.asarray(y).ravel()
    n = src_vals.shape[0]
    if n != eng_mat.shape[0] or n != y.size or eng_mat.shape[1] == 0:
        return True
    # Missingness-aware fail-closed: if the raw source carries non-trivial missingness, the signal often LIVES in the NaN pattern (MNAR). The held-out probe drops NaN rows and cannot assess that, so a Fourier/Hermite/chirp of such a column must NOT be allowed to out-rank the raw column or its is_missing__/missingness_* FE -- drop it. (Those missingness FE columns are exempt: their source name is not a raw column, so the caller never routes them here.)
    if float(np.mean(~np.isfinite(np.asarray(src_vals, dtype=np.float64)))) > 0.02:
        return False
    if n > max_probe_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, max_probe_n, replace=False)
        src_vals, eng_mat, y = src_vals[idx], eng_mat[idx], y[idx]
    uplift = measure_feature_uplift(src_vals, eng_mat, y, classification=infer_classification(y))
    # Fail-open: a probe that COULD NOT measure uplift (None) must KEEP the column
    # (the docstring contract) -- never silently evict on an unmeasurable probe.
    return True if uplift is None else uplift >= threshold
