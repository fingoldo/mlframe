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

# Content-keyed cache of the per-fold X_base-only CV scores (the ``_score(X_base)`` baseline).
# Several engineered SIBLINGS derived from the SAME raw source (x__He2, x__He3, x__T2, x__L2,
# ...) call ``measure_feature_uplift`` with an IDENTICAL X_base/y/seed -- the baseline is
# deterministic (same KFold/StratifiedKFold split given the same seed) and was being refit
# from scratch for every sibling. Keyed on the exact inputs that determine the split + the
# baseline fit (X_base bytes, y bytes, classification, n_splits, seed); bounded FIFO like
# ``_INFER_CLS_MEMO`` above.
_BASELINE_CV_MEMO: dict = {}
_BASELINE_CV_MEMO_MAXSIZE = 64


def _baseline_cv_key(X_base: np.ndarray, y: np.ndarray, *, classification: bool, n_splits: int, seed: int):
    """Content-hash cache key for the ``X_base``-only CV baseline, or ``None`` when the
    inputs cannot be hashed cheaply (falls back to always recomputing -- never crashes)."""
    try:
        return (
            X_base.shape, hash(X_base.tobytes()),
            y.shape, hash(y.tobytes()),
            bool(classification), int(n_splits), int(seed),
        )
    except Exception:
        return None


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
    from sklearn.model_selection import KFold, StratifiedKFold
    from mlframe.metrics.core import fast_roc_auc, fast_r2_score
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
    # Siblings derived from the SAME raw source share an IDENTICAL X_base/y_enc/seed, so the
    # X_base-only baseline CV score is deterministic across sibling calls -- cache it keyed on
    # the exact content that determines the split + fit (see ``_baseline_cv_key``).
    _bkey = _baseline_cv_key(X_base, y_enc, classification=classification, n_splits=n_splits, seed=seed)
    _bcache = _BASELINE_CV_MEMO.get(_bkey) if _bkey is not None else None
    _base_scores: list[float] = []
    try:
        _fold_i = 0
        for tr, va in split_iter:
            if classification and np.unique(y_enc[tr]).size < 2:
                continue

            def _score(Xfull, tr=tr, va=va):
                """Fit a standardized logistic/linear probe on this fold's train rows of ``Xfull`` and return its held-out score (ROC-AUC for binary, otherwise the classifier/regressor's native scorer), closing over the current fold's ``tr``/``va`` indices via default args."""
                sc = StandardScaler()
                Xt = sc.fit_transform(Xfull[tr])
                Xv = sc.transform(Xfull[va])
                if classification:
                    m = LogisticRegression(max_iter=200, C=1.0)
                    m.fit(Xt, y_enc[tr])
                    if binary:
                        return fast_roc_auc(y_enc[va], m.predict_proba(Xv)[:, 1])
                    # Multiclass accuracy (fraction correct); accuracy_ratio is CAP-AR (2*AUC-1),
                    # a different quantity, so compute the exact hit-rate directly.
                    return float(np.mean(y_enc[va] == m.predict(Xv)))
                m = Ridge(alpha=1.0)
                m.fit(Xt, y_enc[tr])
                return fast_r2_score(y_enc[va], m.predict(Xv))

            aug_score = _score(X_aug)
            if _bcache is not None and _fold_i < len(_bcache):
                base_score = _bcache[_fold_i]
            else:
                base_score = _score(X_base)
                _base_scores.append(base_score)
            _fold_i += 1
            deltas.append(aug_score - base_score)
    except Exception as exc:  # pragma: no cover - the probe must never break a fit
        logger.debug("measure_feature_uplift: probe failed (%s); fail-open (None)", exc)
        return None
    if not deltas:
        return None
    if _bkey is not None and _bcache is None and _base_scores:
        if len(_BASELINE_CV_MEMO) > _BASELINE_CV_MEMO_MAXSIZE:
            _BASELINE_CV_MEMO.pop(next(iter(_BASELINE_CV_MEMO)))
        _BASELINE_CV_MEMO[_bkey] = _base_scores
    return float(np.mean(deltas))


_INFER_CLS_MEMO: dict = {}


def infer_classification(y: np.ndarray) -> bool:
    """Cheap classification-vs-regression guess from y's dtype + cardinality.

    Content-memoised: y is a fit-constant asked about by several independent gates per fit, and the
    cardinality probe is a full-n ``np.unique`` sort each time (measured ~0.7s/fit at 1M rows across the
    STRICT gates). The content-signature hash is ~20x cheaper than the sort; bounded FIFO."""
    y = np.asarray(y).ravel()
    if y.dtype.kind in ("b", "O", "U", "S"):
        return True
    _key = None
    try:
        _key = (y.shape, str(y.dtype), hash(y.tobytes()))
        _hit = _INFER_CLS_MEMO.get(_key)
        if _hit is not None:
            return bool(_hit)
    except Exception:
        _key = None
    finite = y[np.isfinite(y)] if y.dtype.kind == "f" else y
    if finite.size == 0:
        return True
    if y.dtype.kind in ("i", "u"):
        _res = np.unique(finite).size <= max(20, int(0.05 * finite.size))
    else:
        _res = np.unique(finite).size <= max(20, int(0.02 * finite.size))
    if _key is not None:
        if len(_INFER_CLS_MEMO) > 8:
            _INFER_CLS_MEMO.pop(next(iter(_INFER_CLS_MEMO)))
        _INFER_CLS_MEMO[_key] = bool(_res)
    return bool(_res)


def class_mi_fe_applicable(y: np.ndarray) -> bool:
    """True iff the MI-floor FE operators (pairwise-modular, integer-lattice, row-argmax, conditional-gate) can score a candidate against y.

    These detectors gate every candidate on plug-in class-MI (``_mi_classif_batch``). A 1D CLASSIFICATION y feeds the kernel directly as raw
    discrete labels. A CONTINUOUS 1D y is now ALSO eligible: the caller quantile-bins it once via ``bin_y_for_class_mi`` into a proper discrete
    variable before scoring, so the kernel sees a meaningful low-cardinality target (the prior int64-cast collapsed a continuous y to ~n bogus
    classes -> inflated-garbage MI, ballooned cost, even a segfault / the conditional-gate tau-sweep HANG). Only a 2D y (multilabel / multi-target
    regression) stays skipped: quantile-binning a label MATRIX is out of scope, and the kernel cannot consume it (silently returns 0 -> dead
    signal). Applicable iff y is 1D (classification OR continuous); the operator handles the binning internally for the continuous case."""
    arr = np.asarray(y)
    if arr.ndim > 1 and arr.shape[-1] > 1:
        return False
    return arr.ndim <= 1


def bin_y_for_class_mi(y: np.ndarray, nbins: int = 10) -> np.ndarray:
    """Return int64 class labels for the MI-floor FE operators' ``_mi_classif_batch`` relevance path, given a 1D y.

    CLASSIFICATION / already-discrete y (``infer_classification`` true) passes through as ``np.asarray(y).astype(np.int64)`` -- BIT-IDENTICAL to
    the prior per-operator cast, so the discrete path does not move. A CONTINUOUS 1D y is quantile-binned into ``nbins`` bins (``pd.qcut``,
    ``duplicates='drop'``) so the kernel sees a meaningful discrete target instead of ~n collapsed int64 classes; this is mlframe's standard
    continuous-y relevance binning (mirrors the MRMR core ``pd.qcut(..., q=nbins, labels=False, duplicates='drop')`` in ``_fit_impl_core``).
    Binned ONCE per fit by the caller and reused across the whole candidate scan (the operators take the returned labels unchanged). On a qcut
    failure (heavy ties / NaN) it falls back to the int64 cast so the fit still runs (signal may degrade, never crashes). ``nbins`` should be the
    MRMR instance's ``quantization_nbins`` so the operator binning matches the core's relevance binning."""
    import pandas as pd

    arr = np.asarray(y).ravel()
    if infer_classification(arr):
        if arr.dtype.kind in ("O", "U", "S"):
            # Non-numeric class labels (string/object dtype, e.g. 'A'..'E') cannot be cast to int64
            # directly -- numpy tries to parse each label as a decimal integer literal and raises
            # ``ValueError: invalid literal for int() with base 10``. Factorize to dense 0..k-1 integer
            # codes instead. Numeric/bool classification labels keep the direct cast unchanged.
            return np.asarray(np.unique(arr, return_inverse=True)[1]).astype(np.int64)
        return arr.astype(np.int64)
    try:
        binned = pd.qcut(arr, q=int(nbins), labels=False, duplicates="drop")
        return np.asarray(binned).astype(np.int64)
    except Exception as exc:  # pragma: no cover - degenerate continuous y (heavy ties / NaN)
        logger.debug("bin_y_for_class_mi: qcut failed (%s); falling back to int64 cast", exc)
        return arr.astype(np.int64)


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
