"""Input validation + signature computation for ``RFECV.fit``.

Carved out of ``_rfecv_fit.py`` to keep the parent below the 1k-line monolith threshold. The function defined here is invoked from the top of ``fit`` as ``X, y, signature, _polars_time_series_hint, _skip = _init_fit_state(...)``; when ``_skip`` is True the caller short-circuits early (signature matches stored ``self.signature``).

Behavioural contract preserved bit-for-bit. Every side-effect on ``self`` (``self._fit_sample_weight_``, ``self._selected_cols_cache``) is performed here exactly once at the same call ordering as the original inline block.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl

from sklearn.base import is_classifier

from ._validate import _sanitize_X_inputs

logger = logging.getLogger("mlframe.feature_selection.wrappers.rfecv")

# Row-chunk byte budget for streamed hashing (~64 MB). Bounds the peak EXTRA RAM of the X-content
# fingerprint to one chunk's bytes instead of a whole-frame ``.tobytes()`` copy, which on a 100+ GB
# frame would double peak RAM and OOM the host.
_HASH_CHUNK_BYTES = 64 * 1024 * 1024


def _stream_hash_array(h, arr: np.ndarray) -> None:
    """Feed ``arr`` into blake2b ``h`` in C-contiguous row-major order, one row-chunk at a time.

    Bit-identical to ``h.update(np.ascontiguousarray(arr).tobytes())`` but never materialises the whole
    buffer as one extra bytes copy: ``arr_c[i:j].tobytes()`` concatenated over contiguous row slices equals
    ``arr_c.tobytes()`` for a C-contiguous array. Peak extra RAM is ~one chunk, not the whole frame.
    """
    arr_c = np.ascontiguousarray(arr)
    if arr_c.ndim == 0 or arr_c.shape[0] == 0:
        h.update(arr_c.tobytes())
        return
    row_bytes = arr_c.dtype.itemsize * int(np.prod(arr_c.shape[1:], dtype=np.int64))
    rows_per_chunk = max(1, _HASH_CHUNK_BYTES // row_bytes) if row_bytes else arr_c.shape[0]
    for i in range(0, arr_c.shape[0], rows_per_chunk):
        h.update(arr_c[i : i + rows_per_chunk].tobytes())


def _current_params_signature(self) -> object:
    """Hashable fingerprint of the selector's CURRENT params (``get_params(deep=True)``).

    Folded into the skip-retraining signature so ANY selector- or wrapped-estimator-parameter
    change invalidates the in-object identity skip (params are re-read at every fit call, so
    ``set_params`` and direct attribute assignment are both captured). ``deep=True`` expands
    nested ``get_params``-bearing objects (``estimator__C`` etc.) so in-place mutation of the
    wrapped estimator's hyperparams also invalidates. On any failure returns a per-call unique
    token (identity equality) => never matches a stored signature => conservative full refit.

    IMPORTANT: callers that STORE the signature after a successful fit must refresh this slot
    with post-fit values (see ``_rfecv_finalize`` / ``_rfecv_stability_select``): ``fit`` may
    resolve params in place (e.g. ``scoring=None -> make_scorer(...)`` at ``_rfecv_fit``;
    ``force_parallel`` thread pinning mutates the wrapped estimator), so the stored signature
    must reflect the state the NEXT fit's ``get_params`` will see, else identical refits would
    never skip.
    """
    try:
        # Lazy import: ``filters`` (its ``_mrmr_fingerprints``) top-level imports ``wrappers`` (RFECV), so a
        # module-level import here would create a hard import cycle; at call time it is cache-free. The public
        # ``filters`` surface lazily re-exports the symbol via ``__getattr__`` (no extra cycle at package init).
        from mlframe.feature_selection.filters import _hashable_params_signature

        return _hashable_params_signature(self.get_params(deep=True))
    except Exception:
        return object()


def _init_fit_state(
    self,
    X: Union[pd.DataFrame, np.ndarray, "pl.DataFrame"],
    y: Union[pd.DataFrame, pd.Series, np.ndarray, "pl.Series"],
    groups: Union[pd.Series, np.ndarray, None],
    sample_weight: Union[np.ndarray, pd.Series, None],
) -> tuple[Any, Any, tuple, bool, bool]:
    """Validate inputs, convert polars -> pandas, compute the (X, y, selector-params) signature, and decide whether to short-circuit.

    Returns
    -------
    X : validated / pandas-flavoured frame
    y : validated y (polars-Series converted to pandas)
    signature : tuple used by skip_retraining_on_same_shape
    polars_time_series_hint : bool (carried through to CV auto-detect)
    skip : True when signature matches stored self.signature and the caller should ``return self`` immediately
    """
    # sample_weight, when provided, is sliced per CV fold and threaded into both the cloned estimator's
    # ``fit(..., sample_weight=fold_train_w)`` call (if the estimator advertises support) and into the
    # sklearn scorer's ``__call__(..., sample_weight=fold_test_w)`` (if the scorer accepts the kwarg).
    # Default None preserves the legacy code path byte-for-byte (regression sentry); the gating flag
    # ``FeatureSelectionConfig.use_sample_weights_in_fs`` decides whether the caller forwards weights here.
    self._fit_sample_weight_ = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if self._fit_sample_weight_ is not None:
        _n_rows_for_sw = X.shape[0] if hasattr(X, "shape") else len(X)
        if self._fit_sample_weight_.ndim != 1:
            raise ValueError(f"RFECV.fit sample_weight must be 1-D, got shape {self._fit_sample_weight_.shape}")
        if self._fit_sample_weight_.shape[0] != _n_rows_for_sw:
            raise ValueError(f"RFECV.fit sample_weight length {self._fit_sample_weight_.shape[0]} != n_rows {_n_rows_for_sw}")
        # Also enforce the y-length invariant. When the caller derives y independently from X (multi-column y coerced row-by-row, polars+pandas mix at the suite boundary), the X-row check above misses the mismatch and sklearn raises an opaque IndexError deep in the per-fold slice.
        if y is not None:
            try:
                _n_y = int(y.shape[0]) if hasattr(y, "shape") else len(y)
                if self._fit_sample_weight_.shape[0] != _n_y:
                    raise ValueError(f"RFECV.fit sample_weight length {self._fit_sample_weight_.shape[0]} != len(y) {_n_y}")
            except TypeError:
                pass
        if not np.all(np.isfinite(self._fit_sample_weight_)) or (self._fit_sample_weight_ < 0).any():
            raise ValueError("RFECV.fit sample_weight must be finite and non-negative")

    # Polars -> pandas at entry. RFECV uses pandas / numpy idioms throughout (KFold.split, current_features.index(...), passthrough_cols).
    # Inner estimators (notably CatBoost) crash on polars Enum columns, so convert once here and let every downstream caller see pandas.
    # Before lossy conversion, stash a "polars-detected monotonic datetime axis" hint so the CV auto-detect block below can route
    # to TimeSeriesSplit. After to_pandas() the original polars schema is gone and the .index becomes a plain RangeIndex.
    _polars_time_series_hint = False
    if isinstance(X, pl.DataFrame):
        try:
            _dt_cols = [n for n, d in X.schema.items() if d in (pl.Datetime, pl.Date) or str(d).startswith(("Datetime", "Date"))]
            if len(_dt_cols) == 1:
                _col = X.get_column(_dt_cols[0])
                if _col.is_sorted(descending=False) and _col.null_count() == 0:
                    _polars_time_series_hint = True
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            logger.debug("suppressed in _fit_init.py:126: %s", e)
            pass
        # ``self_destruct=True`` releases the polars buffers in-place; safe only when RFECV owns the frame. The suite-side _apply_pre_pipeline_transforms passes cloned per-target subsets so the
        # safe path is the historical default, but ad-hoc / notebook callers who pass their own polars frame would silently lose data. Gate on the ``_mlframe_owned_frame_`` marker (set by suite
        # internals on subsets it owns) OR on the explicit instance flag ``_rfecv_owns_polars_frame_`` so a caller can opt in deliberately. Default behaviour for unmarked frames: pay the extra
        # buffer copy rather than corrupt caller data.
        _frame_is_internally_owned = bool(getattr(X, "_mlframe_owned_frame_", False) or getattr(self, "_rfecv_owns_polars_frame_", False))
        try:
            X = X.to_pandas(use_pyarrow_extension_array=True, split_blocks=True, self_destruct=_frame_is_internally_owned)
        except TypeError:
            X = X.to_pandas()
    if isinstance(y, pl.Series):
        y = y.to_pandas()

    # Reject pathological y / X early instead of letting sklearn raise opaque errors deep in the splitter or estimator.
    try:
        y_arr = np.asarray(y)
    except Exception as exc:
        raise ValueError(f"y must be array-like; got {type(y).__name__}: {exc}") from exc
    if y_arr.size == 0:
        raise ValueError("y is empty; nothing to fit.")
    # sklearn parity (2026-05-28): accept bare Python list as y. The downstream signature
    # tuple uses y.shape; lists don't have .shape, so promote to ndarray now.
    if not hasattr(y, "shape"):
        y = y_arr
    # L6 (Wave 5, 2026-05-28): explicit multi-output guard. Currently RFECV is single-y; users with K targets are best served by
    # looping selection per-target and OR-aggregating support_. Document this via a clear error rather than silently flattening y.
    if y_arr.ndim >= 2 and y_arr.shape[-1] > 1:
        raise NotImplementedError(
            "RFECV.fit: multi-output y (shape=" + str(y_arr.shape) + ") is not yet "
            "supported. Loop selection per target column and aggregate "
            "support_ via union (OR), e.g. ``selected = set().union(*[RFECV(...).fit(X, y[:,k]).get_feature_names_out() for k in range(K)])``."
        )
    # NaN / Inf in y are silent miscompute traps in sklearn folds.
    if y_arr.dtype.kind in "fc":
        n_nan_y = int(np.isnan(y_arr).sum())
        n_inf_y = int(np.isinf(y_arr).sum())
        if n_nan_y or n_inf_y:
            raise ValueError(
                f"y contains {n_nan_y} NaN and {n_inf_y} +/-inf values. "
                f"sklearn CV splitters silently mishandle these. Drop or "
                f"impute these rows before passing y to RFECV."
            )
    # Single-class y for classification is a fold-collapse trap.
    if is_classifier(self.estimator if self.estimator is not None else (self.estimators[0] if self.estimators else None)):
        unique_y = np.unique(y_arr)
        if len(unique_y) < 2:
            raise ValueError(
                f"y has only {len(unique_y)} unique class(es) "
                f"({unique_y.tolist()}). Classification CV requires at "
                f"least 2 classes. Check that y is not constant or that "
                f"upstream filtering didn't drop the minority class."
            )
        # Minority-class size must support the requested CV.
        class_counts = np.bincount(y_arr.astype(int)) if y_arr.dtype.kind in "iu" else None
        if class_counts is not None and len(class_counts) > 0:
            min_class = int(class_counts[class_counts > 0].min())
            cv_n = self.cv if isinstance(self.cv, int) else getattr(self.cv, "n_splits", 3)
            if min_class < cv_n:
                raise ValueError(
                    f"Minority class has {min_class} samples but cv={cv_n}. "
                    f"StratifiedKFold requires at least n_splits samples per "
                    f"class. Reduce cv or oversample the minority class."
                )
            # E7 (Wave 4, 2026-05-28): warn when minority just barely meets
            # n_splits; ROC AUC / log_loss likely to NaN on the all-train-no-test
            # minority fold split. Hard floor 2*cv_n recommended.
            if min_class < 2 * cv_n and getattr(self, "verbose", 0):
                logger.warning(
                    "RFECV: minority class has %d samples vs cv=%d. With 1-2 "
                    "minority per fold, the test fold may have 0 minority -> "
                    "ROC AUC / log_loss NaN on that fold. Increase n or use "
                    "RepeatedStratifiedKFold to mitigate.",
                    min_class, cv_n,
                )

    # must_include + must_exclude intersection is a confusing config error.
    if self.must_include and self.must_exclude:
        mi_set = set(self.must_include)
        me_set = set(self.must_exclude)
        overlap = mi_set & me_set
        if overlap:
            raise ValueError(f"must_include and must_exclude both contain {sorted(overlap)}. " f"Resolve the conflict in your config.")

    # X-side input checks. Run after y validation so common operator mistakes surface clearly at fit entry.
    if isinstance(X, pd.DataFrame):
        # Duplicate column names make ``X[label]`` return a DataFrame (not a Series), whose ``.dtype`` access raises downstream, and break the feature-name -> support_ mapping every selector relies on. Surface a clear error at fit entry.
        if X.columns.has_duplicates:
            dup_names = X.columns[X.columns.duplicated()].unique().tolist()
            raise ValueError(
                f"RFECV.fit: duplicate column names not supported: {dup_names[:10]}. "
                f"De-duplicate (e.g. ``X.loc[:, ~X.columns.duplicated()]`` or rename) before fitting."
            )
        # Estimator behaviour on Inf is undefined - LR crashes, CB silently treats as huge. Stream column-by-column instead of materialising the full numeric block: ``X.select_dtypes(...).to_numpy()`` doubles peak RAM on a 100+ GB frame even before checking finiteness.
        _inf_cols: list = []
        for _c in X.columns:
            try:
                _col = X[_c]
                if _col.dtype.kind not in "biufc":
                    continue
                if np.isinf(_col.to_numpy()).any():
                    _inf_cols.append(_c)
                    if len(_inf_cols) >= 10:
                        break
            except (TypeError, ValueError):
                continue
        if _inf_cols:
            raise ValueError(
                f"X contains +/-Inf values in column(s) {_inf_cols[:10]}. "
                f"Estimator behaviour on Inf is undefined. Drop or "
                f"clip these values before fit()."
            )

        # Tree-based estimators handle NaN; linear models don't.
        if getattr(self, "verbose", 0):
            # Existence-only check, lazy short-circuits at the first NaN cell instead of materialising the full bool-frame ``X.isna().to_numpy().sum()`` (OOMs on 100+ GB frames just to compute the warn-once count).
            if bool(X.isna().any().any()):
                logger.warning(
                    "RFECV: X has NaN cells. Tree-based estimators "
                    "(RF/CB/XGB/HGBM) handle NaN; linear models (LR, "
                    "Ridge, Lasso) do NOT and will crash on .fit(). "
                    "Pre-impute via SimpleImputer / KNNImputer if using "
                    "linear estimators."
                )

        _obj_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        if _obj_cols:
            _user_cats = set(self.cat_features or [])
            _unhandled = [c for c in _obj_cols if c not in _user_cats]
            if _unhandled and getattr(self, "verbose", 0):
                logger.warning(
                    "RFECV: %d object/string/category column(s) %s have "
                    "NOT been listed in cat_features=. CB/XGB will crash "
                    "on string columns; LR will fail on .fit(). Either "
                    "encode them upstream or pass via cat_features.",
                    len(_unhandled), _unhandled[:10],
                )

    # n_samples < 2 * cv breaks every k-fold split.
    cv_n = self.cv if isinstance(self.cv, int) else getattr(self.cv, "n_splits", 3)
    # GroupKFold / StratifiedGroupKFold split on the GROUP axis, not rows. ``X.shape[0] < 2 * cv_n`` slips past for a 30k-row frame with 3 groups + ``cv=GroupKFold(5)`` and then sklearn raises an opaque "Cannot have number of splits > number of groups" deep in the splitter. Check n_unique_groups against the floor explicitly when groups are provided.
    if groups is not None:
        _n_groups_for_floor: int | None = None
        try:
            _n_groups_for_floor = int(len(np.unique(np.asarray(groups))))
        except (TypeError, ValueError):
            _n_groups_for_floor = None
        if _n_groups_for_floor is not None and _n_groups_for_floor < 2 * cv_n:
            raise ValueError(f"n_groups={_n_groups_for_floor} < 2 * cv ({cv_n}); group-aware splitter would fail. Reduce cv or merge groups.")
    if X.shape[0] < 2 * cv_n:
        raise ValueError(f"n_samples={X.shape[0]} < 2 * cv ({cv_n}); each fold would " f"have <2 train samples. Reduce cv or get more data.")

    X = _sanitize_X_inputs(self, X, y)

    # NaN-in-X policy (mirrors MRMR's native-NaN contract): graceful median-impute by default so a linear
    # core no longer crashes on ordinary missing data; 'raise' preserves the strict legacy crash. Runs AFTER
    # sanitise (zero-variance / dup drops already applied) and BEFORE the signature so re-fits are stable and
    # any emitted ``is_missing__{col}`` indicators flow through original_features / voting / support_ normally.
    from ._nan_policy import apply_nan_in_X_policy
    X = apply_nan_in_X_policy(self, X)

    # Inputs/outputs signature. Shape alone isn't enough - two datasets with identical (n, p) but different column identities must
    # trigger a retrain, otherwise self.support_ silently applies stale column selections. y-content is folded in via a blake2b
    # 16-byte digest because two semantically-different targets of the same length and shape (e.g. column-A binary vs column-B
    # binary picked off the same frame) used to replay the prior fit's support_; without the y-hash a per-target FS loop reusing
    # one RFECV instance silently selected features for whichever target arrived first.
    if isinstance(X, pd.DataFrame):
        columns_key = tuple(map(str, X.columns.tolist()))
    else:
        columns_key = ("__ndarray__", int(X.shape[1]))
    try:
        _y_arr = np.ascontiguousarray(y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y))
        _y_hash = hashlib.blake2b(_y_arr.tobytes(), digest_size=16).hexdigest()
    except (TypeError, ValueError):
        # Object-dtype y or otherwise non-bytes-castable; fall back to a stringified per-element hash that still discriminates content.
        _y_hash = hashlib.blake2b(
            ",".join(map(str, np.asarray(y).ravel().tolist())).encode("utf-8"),
            digest_size=16,
        ).hexdigest()
    # Full-content X hash to disambiguate the skip-retrain signature. Strided / sampled X fingerprints collided on heavily-reshuffled X whose sampled rows incidentally matched (e.g. stratified rebalance preserving boundaries); a full blake2b over X.tobytes() rules this out for ~50ms on a 1M x 100 frame -- well under a single RFECV iter cost. Object-dtype frames fall back to str-cast hashing because tobytes is non-deterministic for object arrays. Symmetric with the X-content hash now folded into the MRMR _FIT_CACHE key.
    try:
        _n = int(X.shape[0])
        if _n > 0:
            if isinstance(X, pd.DataFrame):
                _numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
                _nonnum_cols = [c for c in X.columns if c not in set(_numeric_cols)]
                _h = hashlib.blake2b(digest_size=12)
                if _numeric_cols:
                    _stream_hash_array(_h, X[_numeric_cols].to_numpy())
                for _c in _nonnum_cols:
                    _h.update(str(_c).encode("utf-8"))
                    _h.update(b"\x00")
                    _h.update(",".join(map(str, X[_c].astype(str).tolist())).encode("utf-8"))
                    _h.update(b"\x01")
                _x_hash = _h.hexdigest()
            else:
                _x_arr = np.asarray(X)
                if _x_arr.dtype == object:
                    _x_hash = hashlib.blake2b(
                        ",".join(map(str, _x_arr.ravel().tolist())).encode("utf-8"),
                        digest_size=12,
                    ).hexdigest()
                else:
                    # blake2b reads the contiguous array buffer directly (no
                    # .tobytes() copy); bit-identical to hashing tobytes() bytes.
                    _x_hash = hashlib.blake2b(
                        np.ascontiguousarray(_x_arr),
                        digest_size=12,
                    ).hexdigest()
        else:
            _x_hash = "empty"
    except (TypeError, ValueError):
        _x_hash = hashlib.blake2b(
            repr(np.asarray(X).ravel()[:100].tolist()).encode("utf-8"),
            digest_size=12,
        ).hexdigest()
    # 2026-06-10 fix: fold the selector's OWN parameter signature (incl. the wrapped estimator's params via
    # ``deep=True`` expansion) into the skip signature. Pre-fix the signature was
    # ``(X.shape, y.shape, columns_key, y_hash, x_hash)`` -- SELECTOR/ESTIMATOR PARAMS were absent: refitting
    # the same RFECV instance with changed settings (``set_params`` or direct attribute assignment, e.g. a new
    # ``max_nfeatures`` / ``scoring`` / mutated ``estimator`` hyperparams) on identical data silently replayed
    # the prior fit's ``support_`` computed under the OLD params. Same bug class as MRMR's in-object skip
    # (fixed same day) and the earlier y-content hole fixed above. See ``_current_params_signature`` for the
    # set_params/attribute-assignment capture semantics and the post-fit refresh contract at the store sites.
    signature = (X.shape, y.shape, columns_key, _y_hash, _x_hash, _current_params_signature(self))
    # Invalidate stale support_/cache at fit entry so a partial-fit failure cannot leave a previous-fit's selection silently in place.
    # The cache is rebuilt below only on a successful path.
    self._selected_cols_cache = None
    skip = False
    if self.skip_retraining_on_same_shape:
        if signature == self.signature:
            if self.verbose:
                logger.info("Skipping retraining on the same inputs signature %s", signature)
            skip = True
    return X, y, signature, _polars_time_series_hint, skip
