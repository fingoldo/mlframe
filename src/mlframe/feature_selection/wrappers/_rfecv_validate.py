"""Input-validation + sanitisation chain for ``RFECV.fit``.

Carved out of ``_rfecv_fit``'s pre-while setup. Runs four operations
in order on ``X`` and emits the post-fix shapes:

1. WARN on ``p >= 5000`` features with no ``max_nfeatures`` cap.
2. WARN on ``max_runtime_mins < 1 second`` (units confusion).
3. WARN on ``cv >= n_samples`` (effective LeaveOneOut on small data).
4. Drop zero-variance / all-null columns BEFORE recording
   ``feature_names_in_`` so a constant column can't land in
   ``support_`` and trip transform-time column-set drift.
5. Drop exact-duplicate columns (numeric + categorical) so RFECV's
   voting doesn't split a duplicated feature's importance across copies.
6. Honour ``must_exclude``: drop named columns at fit entry so they
   never enter the optimiser's universe.
7. Optional leakage scan: Pearson |corr(X, y)| >= leakage_corr_threshold
   trips one of {warn, raise, exclude} per ``leakage_action``.

Re-imported at the parent's module bottom so historical
``from ._rfecv_fit import _sanitize_X_inputs`` keeps resolving
transparently.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("mlframe.feature_selection.wrappers._rfecv")


def _sanitize_X_inputs(self, X, y):
    """Apply validation warnings + the four destructive sanitise passes.

    Returns the (possibly modified) ``X``. ``y`` is not modified;
    the leakage scan is read-only. Comments preserved verbatim from
    the prior in-line shape so the bug-class context stays attached
    to each step.
    """
    # p >> n with no max_nfeatures cap - MBH search space is O(p) and each iter is a CV fit, so runtime is unbounded.
    if X.shape[1] >= 5000 and self.max_nfeatures is None and getattr(self, "verbose", 0):
        logger.warning(
            "RFECV: p=%d features with no max_nfeatures cap. The "
            "MBH search space is O(p) and each iter is a CV fit; runtime "
            "is unbounded. Set max_nfeatures=300 (or use "
            "stability_selection=True / importance_getter='knockoff' "
            "for sub-second selection on this scale).",
            X.shape[1],
        )

    # max_runtime_mins below 1 second is almost certainly a units-confusion mistake.
    if self.max_runtime_mins is not None:
        if self.max_runtime_mins < 0:
            raise ValueError(f"max_runtime_mins must be >= 0; got {self.max_runtime_mins}")
        if 0 < self.max_runtime_mins < (1.0 / 60.0) and getattr(self, "verbose", 0):
            logger.warning(
                "RFECV: max_runtime_mins=%.4f is < 1 second. Did you mean "
                "to pass seconds instead of minutes? RFECV will likely "
                "exit after iter 1 with an under-fit selection.",
                self.max_runtime_mins,
            )

    # cv >= n_samples (LeaveOneOut on small data) gives 1-sample test folds where most metrics are undefined.
    if isinstance(self.cv, int) and self.cv >= X.shape[0] and getattr(self, "verbose", 0):
        logger.warning(
            "RFECV: cv=%d >= n_samples=%d (effectively LeaveOneOut). "
            "Per-fold test set has 1 sample; ROC AUC and many other "
            "metrics are undefined and will produce NaN scores. Use "
            "cv <= n_samples / 5 for stable scoring.",
            self.cv, X.shape[0],
        )

    # Drop zero-variance / all-null columns BEFORE recording ``feature_names_in_``. Otherwise a constant column can land in support_ and
    # downstream pipeline steps that silently drop it (e.g. SimpleImputer with keep_empty_features=False) cause transform-time column-set
    # drift. Vectorised across ALL dtypes via DataFrame.nunique() so constant categorical / string / bool columns are also caught.
    if isinstance(X, pd.DataFrame) and X.shape[1] > 0:
        try:
            nunique = X.nunique(dropna=True)
            degenerate = nunique[nunique <= 1].index.tolist()
        except TypeError:
            # Fallback for exotic dtypes that nunique() can't hash.
            degenerate = []
            for col in X.columns:
                series = X[col]
                if series.isna().all():
                    degenerate.append(col)
                else:
                    try:
                        if series.nunique(dropna=True) <= 1:
                            degenerate.append(col)
                    except TypeError:
                        continue
        if degenerate:
            if getattr(self, "verbose", 0):
                logger.info(
                    "RFECV: dropping %d zero-variance / all-null column(s) "
                    "before fit so they cannot leak into ``support_`` or "
                    "trip a transform-time column-set drift later: %s",
                    len(degenerate), degenerate,
                )
            X = X.drop(columns=degenerate)

    # Drop exact-duplicate columns (numeric AND categorical). Without this, RFECV's voting splits the importance of a duplicated feature across all copies,
    # biasing selection toward isolated noise features whose FI isn't diluted. ``pandas.util.hash_array`` is dtype-agnostic and treats NaN as a single sentinel,
    # so a real ``-1.234e308`` value no longer collides with NaN the way the prior ``np.nan_to_num(...).tobytes()`` path did; categorical columns are factorised
    # to integer codes before hashing so two semantically-identical category sequences with different ordered category dictionaries still dedup.
    if isinstance(X, pd.DataFrame) and X.shape[1] > 1:
        try:
            from pandas.util import hash_array as _hash_array

            _hashes: dict[bytes, str] = {}
            _to_drop = []
            for _col in X.columns:
                _ser = X[_col]
                _dtype = _ser.dtype
                if pd.api.types.is_numeric_dtype(_dtype) or pd.api.types.is_bool_dtype(_dtype):
                    _arr = _ser.to_numpy()
                elif isinstance(_dtype, pd.CategoricalDtype) or _dtype == object or pd.api.types.is_string_dtype(_dtype):
                    # ``factorize`` is the canonical pandas pathway for collapsing label semantics into a code sequence dedup can compare; ``use_na_sentinel=True`` keeps NaN distinct from any real label without forcing a fillna copy.
                    _codes, _ = pd.factorize(_ser, use_na_sentinel=True)
                    _arr = _codes
                else:
                    continue
                _key = bytes(_hash_array(_arr))
                if _key in _hashes:
                    _to_drop.append(_col)
                else:
                    _hashes[_key] = _col
            if _to_drop:
                if getattr(self, "verbose", 0):
                    logger.info(
                        "RFECV: dropping %d duplicate column(s) (exact-equal "
                        "to another column already kept) before fit: %s. "
                        "Pass them via ``feature_groups`` if you want "
                        "all-or-nothing group decisions.",
                        len(_to_drop), _to_drop,
                    )
                X = X.drop(columns=_to_drop)
        except (TypeError, ValueError):
            # Non-hashable dtype - skip dedup.
            pass

    # must_exclude: drop named columns at fit entry so they never enter the optimiser's universe.
    if self.must_exclude and isinstance(X, pd.DataFrame):
        _drop = [c for c in self.must_exclude if c in X.columns]
        if _drop:
            if getattr(self, "verbose", 0):
                logger.info(
                    "RFECV: must_exclude drops %d column(s) at fit entry: %s",
                    len(_drop), _drop,
                )
            X = X.drop(columns=_drop)
        _missing = [c for c in self.must_exclude if c not in (list(X.columns) + _drop)]
        # Missing names are silently ignored: the column is already absent so the exclusion goal is satisfied.

    # Target-leakage early warning: Pearson correlation between each numeric feature and y.
    # Common leak shapes: ID columns that encode the target, post-hoc enrichments, target-encoded categoricals computed on the full set.
    _suspicious: list = []
    if self.leakage_corr_threshold is not None and isinstance(X, pd.DataFrame) and X.shape[0] >= 30:
        try:
            _y_arr = np.asarray(y, dtype=float).ravel()
            if _y_arr.size == X.shape[0] and not np.all(np.isnan(_y_arr)):
                _y_std = float(np.nanstd(_y_arr))
                if _y_std > 1e-12:
                    # Use is_numeric_dtype so pandas nullable extension dtypes (Int8/.../Float64) and boolean dtypes also enter the
                    # leakage scan; select_dtypes(include="number") silently skips them.
                    from pandas.api.types import is_numeric_dtype, is_bool_dtype
                    _numeric_cols = [c for c in X.columns
                                     if is_numeric_dtype(X[c]) or is_bool_dtype(X[c])]
                    for _c in _numeric_cols:
                        _x_arr = np.asarray(X[_c].values, dtype=float)
                        _mask = np.isfinite(_x_arr) & np.isfinite(_y_arr)
                        if _mask.sum() < 10:
                            continue
                        _x_std = float(np.nanstd(_x_arr[_mask]))
                        if _x_std < 1e-12:
                            continue
                        _corr = float(np.corrcoef(_x_arr[_mask], _y_arr[_mask])[0, 1])
                        if abs(_corr) >= float(self.leakage_corr_threshold):
                            _suspicious.append((_c, round(_corr, 4)))
                    # The outer try/except catches only TypeError/ValueError raised by the corr computation, NOT our intentional 'raise'
                    # action - so we collect findings here and raise OUTSIDE the try.
        except (TypeError, ValueError):
            pass
        if _suspicious:
            _msg = (
                f"RFECV: {len(_suspicious)} feature(s) have "
                f"|Pearson(X, y)| >= {self.leakage_corr_threshold}, "
                f"likely target leakage. Inspect: {_suspicious[:20]}. "
                f"To suppress, set leakage_corr_threshold=None or "
                f"list these in must_exclude."
            )
            _action = getattr(self, "leakage_action", "warn")
            if _action == "raise":
                raise ValueError(_msg + " (leakage_action='raise')")
            elif _action == "exclude":
                _leaky_cols = [c for c, _ in _suspicious if c in X.columns]
                if _leaky_cols:
                    logger.warning(
                        _msg + " (leakage_action='exclude' - dropping these columns)"
                    )
                    X = X.drop(columns=_leaky_cols)
            else:
                logger.warning(_msg)

    return X
