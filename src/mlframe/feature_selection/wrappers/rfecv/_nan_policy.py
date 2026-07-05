"""NaN-in-X handling for ``RFECV.fit``, mirroring MRMR's native-NaN contract.

Ordinary tabular data carries NaN, and the sklearn linear cores that drive RFECV by default
(``LogisticRegression`` / ``Ridge``) raise ``ValueError: Input X contains NaN`` from ``validate_data``.
MRMR ingests NaN natively (separate-bin MI + the Layer-37 ``is_missing__{col}`` emitter); RFECV used to
hard-crash. ``nan_in_X_policy`` closes that gap with a graceful default:

* ``"impute"`` (default): median-impute NaN per column at fit entry so the linear core no longer crashes.
  When the core estimator NATIVELY tolerates NaN (HistGradientBoosting / CatBoost / LightGBM / XGBoost),
  imputation is skipped and NaN passes through untouched. Optionally emits ``is_missing__{col}`` indicator
  columns (``nan_indicator_cols=``) so an MNAR signal carried by the missingness pattern stays capturable,
  matching MRMR's Layer-37 path (which is itself opt-in, so the indicators here default OFF for parity).
* ``"raise"``: preserve the strict legacy crash (benchmarks / replay).

RAM-safety: the median-impute mutates a LOCAL working frame that RFECV already owns (the sanitised copy
produced by ``_sanitize_X_inputs``), never the caller's frame. On a NaN-free frame the impute path is a
no-op (the per-column NaN scan short-circuits), so non-NaN selection is byte-identical to before.
"""
from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")

# Estimators that natively handle NaN but may not advertise the sklearn ``allow_nan`` input tag reliably
# (third-party gradient boosters). Matched by class name so we don't import the optional packages here.
_NATIVE_NAN_ESTIMATOR_NAMES = frozenset({
    "HistGradientBoostingClassifier",
    "HistGradientBoostingRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "LGBMClassifier",
    "LGBMRegressor",
    "XGBClassifier",
    "XGBRegressor",
})

INDICATOR_PREFIX = "is_missing__"


def _estimator_tolerates_nan(estimator) -> bool:
    """True iff ``estimator`` can be fit on X containing NaN without raising.

    Detection order: the sklearn ``__sklearn_tags__().input_tags.allow_nan`` flag (canonical for sklearn
    estimators), then a known-name allowlist for third-party boosters that handle NaN but may not expose
    the tag. A Pipeline delegates to its final step. On any introspection failure, return False so the
    safe (impute) path runs.
    """
    if estimator is None:
        return False
    try:
        from sklearn.pipeline import Pipeline

        if isinstance(estimator, Pipeline) and estimator.steps:
            estimator = estimator.steps[-1][1]
    except Exception:
        pass
    try:
        tags = estimator.__sklearn_tags__()
        allow = getattr(getattr(tags, "input_tags", None), "allow_nan", None)
        if allow is True:
            return True
    except Exception:
        pass
    return type(estimator).__name__ in _NATIVE_NAN_ESTIMATOR_NAMES


def _any_core_tolerates_nan(self) -> bool:
    """True iff EVERY configured core estimator tolerates NaN (so we can safely skip imputation).

    Conservative on the multi-estimator path: if any one core would crash on NaN, we impute for all of
    them (a single shared working frame feeds every fold fit).
    """
    estimators = list(self.estimators) if getattr(self, "estimators", None) else ([self.estimator] if getattr(self, "estimator", None) is not None else [])
    if not estimators:
        return False
    return all(_estimator_tolerates_nan(e) for e in estimators)


def apply_nan_in_X_policy(self, X: Union[pd.DataFrame, np.ndarray]):
    """Resolve ``nan_in_X_policy`` against the (sanitised) working frame ``X``; return the possibly-modified X.

    Only acts when X actually contains NaN -- a NaN-free frame returns unchanged (byte-identical selection).
    ndarray X is handled too (median-impute in place on the local copy; indicators are pandas-only and skipped
    with a warning for raw ndarray input, since they need column names).
    """
    policy = getattr(self, "nan_in_X_policy", "impute")
    if policy not in ("impute", "raise"):
        raise ValueError(f"nan_in_X_policy must be 'impute' or 'raise'; got {policy!r}.")

    verbose = getattr(self, "verbose", 0)

    if isinstance(X, pd.DataFrame):
        nan_cols = [c for c in X.columns if X[c].isna().any()]
        has_nan = bool(nan_cols)
    else:
        try:
            _arr = np.asarray(X)
            if _arr.dtype.kind in "fc":
                has_nan = bool(np.isnan(_arr).any())
            elif _arr.dtype.kind == "O":
                # Object-dtype ndarrays can hold embedded ``float('nan')``; np.isnan rejects object dtype,
                # so coerce to float (non-numeric -> NaN, harmless here) and scan. Integer kinds can't hold NaN.
                has_nan = bool(np.isnan(_arr.astype(float, copy=False)).any())
            else:
                has_nan = False
        except (TypeError, ValueError):
            has_nan = False
        nan_cols = []

    if not has_nan:
        return X

    if policy == "raise":
        # Preserve the strict legacy crash so benchmarks / replay can pin it. Raise the same
        # message shape sklearn's validate_data would produce so existing ``except ValueError`` chains match.
        raise ValueError(
            "Input X contains NaN. RFECV(nan_in_X_policy='raise') reproduces the strict legacy "
            "behaviour; pass nan_in_X_policy='impute' (the default) for graceful median-imputation."
        )

    # policy == "impute"
    if _any_core_tolerates_nan(self):
        if verbose:
            logger.info("RFECV: X has NaN but the core estimator natively tolerates it; passing NaN through " "(no imputation).")
        return X

    indicator_cols = tuple(getattr(self, "nan_indicator_cols", ()) or ())

    if isinstance(X, pd.DataFrame):
        # Capture the PRE-impute missingness masks for the requested indicator columns BEFORE fillna,
        # so an ``is_missing__{col}`` indicator reflects the original missingness pattern (MNAR signal).
        indicator_masks: dict = {}
        for c in indicator_cols:
            if c not in X.columns:
                if verbose:
                    logger.warning("RFECV nan_indicator_cols: %r not in X; skipping indicator.", c)
                continue
            indicator_masks[c] = X[c].isna().to_numpy().astype(np.int8)

        # Work on a shallow copy so the caller's frame is never mutated; column reassignment below
        # only rebinds the imputed columns (no full-frame value copy of the untouched columns).
        X = X.copy(deep=False)
        for c in nan_cols:
            col = X[c]
            arr = col.to_numpy(dtype=float, na_value=np.nan)
            med = float(np.nanmedian(arr)) if np.isfinite(arr).any() else 0.0
            X[c] = col.fillna(med)

        for c, mask in indicator_masks.items():
            name = f"{INDICATOR_PREFIX}{c}"
            if name not in X.columns:
                X[name] = mask

        if verbose:
            logger.info(
                "RFECV nan_in_X_policy='impute': median-imputed %d column(s) with NaN " "(core estimator does not tolerate NaN).",
                len(nan_cols),
            )
        return X

    # ndarray path
    arr = np.array(X, dtype=float, copy=True)
    if indicator_cols and verbose:
        logger.warning("RFECV nan_in_X_policy: nan_indicator_cols requires a named DataFrame; " "ignoring indicators for ndarray X.")
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = np.isnan(col)
        if mask.any():
            finite = col[~mask]
            med = float(np.median(finite)) if finite.size else 0.0
            arr[mask, j] = med
    if verbose:
        logger.info("RFECV nan_in_X_policy='impute': median-imputed ndarray X NaN cells.")
    return arr
