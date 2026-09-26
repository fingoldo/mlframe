"""Input-frame preparation stages of ``_fit_impl``."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
import pandas as pd
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import _NULLABLE_DENSIFY_EAGER_MAX_BYTES
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
# --- end imports ---


def _stash_fe_targets(self, _y_np, X):
    """Stash the targets the FE stages fit against (rank-transformed y for auto-escalation, raw continuous y for the prewarp ALS)."""
    try:
        _y_esc_arr = _y_np
        if _y_esc_arr.ndim == 1 and _y_esc_arr.dtype.kind in "fiub" and len(_y_esc_arr) == len(X):
            _y_esc_rank = np.argsort(np.argsort(_y_esc_arr, kind="stable"), kind="stable").astype(np.float64)
            self._fe_escalation_y_rank_ = _y_esc_rank / max(len(_y_esc_rank) - 1, 1)
        else:
            self._fe_escalation_y_rank_ = None
    except Exception as exc:
        logger.debug("mrmr: FE-escalation y-rank computation failed; rank unavailable this fit: %r", exc, exc_info=True)
        self._fe_escalation_y_rank_ = None

    # PREWARP ALS RECONSTRUCTION TARGET: stash the RAW CONTINUOUS y so
    # the pair-search rank-1 ALS warp reconstructs against the faithful continuous
    # target rather than the coarse equal-frequency screening codes the target-rebin
    # guard (above) produces. The guard correctly coarsens ``classes_y`` for the MI
    # screen/gates, but a least-squares f(a)*g(b) reconstruction loses fidelity on a
    # non-monotone product when fit to 10-bin codes (measured |corr| 0.97 -> 0.88).
    # Unlike the escalation rank-y this is the raw VALUES (the supervised MDLP-quality
    # signal the ALS needs; rank-y only recovered 0.88 -> 0.88 in benchmarking). Same
    # leak-safety: a fit-time supervised target whose emitted recipe stays a
    # closed-form function of x. Deleted at fit end (transient, keeps the pickle slim).
    # Non-numeric / multi-output y -> None (ALS falls back to ``classes_y`` codes).
    try:
        _y_pw_arr = _y_np
        if _y_pw_arr.ndim == 1 and _y_pw_arr.dtype.kind in "fiub" and len(_y_pw_arr) == len(X):
            self._fe_prewarp_y_continuous_ = np.ascontiguousarray(_y_pw_arr, dtype=np.float64)
        else:
            self._fe_prewarp_y_continuous_ = None
    except Exception as exc:
        logger.debug("mrmr: prewarp continuous-y stash failed; ALS reconstruction target unavailable: %r", exc, exc_info=True)
        self._fe_prewarp_y_continuous_ = None


def _finalise_feature_names_in(self, X, _all_cols, _engineered_names_set, verbose):
    """Set ``feature_names_in_`` to the caller's input columns, never the engineered ones an FE stage appended."""
    if isinstance(X, pd.DataFrame) and X.columns.has_duplicates:
        # Layer 64 defense: keep only the FIRST occurrence
        # of each duplicate-label column position in X. The engineered
        # rosters and the recipe ledger are NOT pruned here - the
        # recipe is what the transform path uses to re-emit the column,
        # so dropping the name from the roster would break
        # ``transform`` (it tries to look up the support_ name in the
        # input X, doesn't find the recipe replay output, and raises
        # "MRMR.transform: N/K selected columns missing from input X").
        # The duplicate is purely a fit-time X-frame artefact (one FE
        # stage re-emitted a column another stage already appended);
        # the recipe replay produces a single canonical column at
        # transform time.
        _seen_cols: set[str] = set()
        _keep_positions: list[int] = []
        _shadowed_eng_names: set[str] = set()
        _n_dropped = 0
        for _i, _c in enumerate(_all_cols):
            if _c in _seen_cols:
                if _c in _engineered_names_set:
                    _shadowed_eng_names.add(_c)
                _n_dropped += 1
                continue
            _seen_cols.add(_c)
            _keep_positions.append(_i)
        X = X.iloc[:, _keep_positions].copy()
        _all_cols = X.columns.tolist()
        if verbose:
            logger.warning(
                "MRMR.fit: pruned %d duplicate column label(s) before "
                "target injection; engineered names shadowed (kept "
                "first occurrence + recipe ledger entry intact): %s",
                _n_dropped,
                sorted(_shadowed_eng_names),
            )
    # When embedding/text passthrough narrowed X above, ``_all_cols`` lacks the passthrough columns; ``feature_names_in_`` must still reflect the FULL user-facing
    # input (passthrough columns included, in their original positions) so the sklearn ``n_features_in_`` contract matches transform's input width. The passthrough
    # indices are re-appended to ``support_`` at fit-end so transform re-emits them.
    _names_source = getattr(self, "_passthrough_full_columns_", None) if self._passthrough_features_ else None
    if _names_source is not None:
        _fni = [c for c in _names_source if c not in _engineered_names_set]
    else:
        _fni = [c for c in _all_cols if c not in _engineered_names_set]
    return X, _fni


def _prepare_input_frame(self, X, verbose):
    """Bring X to a pandas frame MI discretisation can read: numpy -> DataFrame, embedding/free-text passthrough, nullable dtypes densified."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        self._feature_names_in_synthesized_ = True
    else:
        self._feature_names_in_synthesized_ = False

    # EMBEDDING / FREE-TEXT PASSTHROUGH. MI discretisation needs scalar (hashable, orderable) cells; embedding-vector columns (object cells = list/ndarray) and
    # long free-text columns violate that and would crash the discretiser or mis-bin into a useless ~N-level categorical. Detect them here and EXCLUDE them from
    # the working frame so the screen / FE / MI never see them, but PASS THEM THROUGH to the transform output unchanged - the learnable-embedding MLP / recurrent
    # network (and the ``_encode_emb_text_fit`` boundary encoder) are the correct consumers. ``feature_names_in_`` (set below from the full pre-narrow column list)
    # still counts them so the sklearn ``n_features_in_`` contract matches the user's input width; the passthrough indices are re-appended to ``support_`` at
    # fit-end. Default ON (a corrective mechanism; the legacy crash/drop was silently wrong); set ``embedding_passthrough=False`` for the legacy behaviour.
    self._passthrough_features_ = []
    if getattr(self, "embedding_passthrough", True) and isinstance(X, pd.DataFrame):
        from mlframe.feature_selection.filters._mrmr_passthrough import detect_passthrough_columns

        _emb_cols, _text_cols = detect_passthrough_columns(
            X,
            detect_embeddings=getattr(self, "embedding_passthrough_detect_embeddings", True),
            detect_text=getattr(self, "embedding_passthrough_detect_text", True),
        )
        _passthrough = list(_emb_cols) + [c for c in _text_cols if c not in _emb_cols]
        if _passthrough:
            self._passthrough_features_ = _passthrough
            # Column-subset selection shares the underlying column buffers (no row copy) - RAM-safe on 100+ GB frames. The original full column order is recovered
            # at fit-end from ``feature_names_in_`` (built from the pre-narrow list below) so the re-appended passthrough indices land at their true positions.
            _passthrough_set = set(_passthrough)
            _keep_cols = [c for c in (X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)) if c not in _passthrough_set]
            self._passthrough_full_columns_ = X.columns.tolist() if hasattr(X.columns, "tolist") else list(X.columns)
            X = X[_keep_cols]
            if verbose:
                logger.info(
                    "MRMR.fit: routing %d non-scalar column(s) THROUGH feature selection unchanged (embeddings=%s, text=%s); they bypass the MI screen and reach the estimator raw.",
                    len(_passthrough),
                    _emb_cols,
                    _text_cols,
                )

    # NULLABLE-DTYPE DENSIFICATION (gaps_fe_masking-09). A pandas masked-array frame (Int64 / Float64 / boolean +
    # pd.NA) is NOT what the screen / FE-pair numba kernels and the ``dtype.kind=="f"`` NaN guard expect:
    # ``DataFrame.to_numpy()`` on a mixed nullable frame yields object cells holding pd.NA (NOT float64+NaN), so
    # numeric FE families (e.g. conditional_gate) silently skip those columns and the SELECTION diverges from the
    # dense-float64 fit. Densify masked numeric / boolean columns to float64 (pd.NA -> NaN, semantically lossless)
    # so every downstream path is dtype-agnostic. Categorical / string extension columns are left untouched for
    # categorize_dataset (their ``dtype.kind`` is 'O' / 'U', not in the masked numeric set). Default ON: a
    # corrective mechanism (the legacy silent column-skip was wrong), no flag.
    if isinstance(X, pd.DataFrame):
        _nullable_num = [c for c in X.columns if pd.api.types.is_extension_array_dtype(X[c].dtype) and getattr(X[c].dtype, "kind", "O") in ("i", "u", "f", "b")]
        if _nullable_num:
            # A single ``assign`` of every nullable column materialises all the float64 arrays before building the
            # frame (peak ~2x the nullable-column bytes); above the threshold densify one column per ``assign`` so
            # each intermediate frame is freed and peak extra RAM stays ~one column. ``assign`` returns a new frame
            # either way, so the caller's frame is never mutated - the densification stays RAM-safe on 100+ GB frames.
            if len(X) * len(_nullable_num) * 8 <= _NULLABLE_DENSIFY_EAGER_MAX_BYTES:
                X = X.assign(**{c: X[c].astype("float64") for c in _nullable_num})
            else:
                for _nc in _nullable_num:
                    X = X.assign(**{_nc: X[_nc].astype("float64")})
            if verbose:
                logger.info(
                    "MRMR.fit: densified %d nullable masked column(s) to float64 (NaN-preserving): %s",
                    len(_nullable_num),
                    _nullable_num[:8],
                )
    return X


def _categorical_var_names(_is_polars_input, X):
    """Names of the categorical columns of X (string, categorical, enum, boolean dtypes), read from the schema."""
    if _is_polars_input:
        # Polars schema-driven detection; mirrors categorize_dataset's _is_pl_cat.
        import polars as _pl

        _CAT_DTYPES_FOR_VARS = {_pl.Utf8, _pl.String, _pl.Categorical, _pl.Boolean}
        categorical_vars_names = [name for name, dt in X.schema.items() if dt in _CAT_DTYPES_FOR_VARS or (hasattr(_pl, "Enum") and isinstance(dt, _pl.Enum))]
    else:
        categorical_vars_names = X.head().select_dtypes(include=("category", "object", "string", "bool")).columns.values.tolist()
    return categorical_vars_names


def _inject_targets(self, y, X):
    """Append the target column(s) to X as ``<prefix>_<i>`` so discretisation bins them with the features. Polars frames get a new
    frame sharing buffers; a pandas frame is modified in place and registered for cleanup, so a later raise in ``fit``
    still strips the injected columns from the caller's frame. Returns the frame, whether it is Polars, and the target names."""
    from mlframe.feature_selection.filters.mrmr.shared import target_to_numpy_values as _target_to_numpy_values

    target_prefix = self._resolve_target_prefix()
    y_shape = y.shape
    if len(y_shape) == 2:
        y_shape = y_shape[1]
    else:
        y_shape = 1
    target_names = [target_prefix + "_" + str(i) for i in range(y_shape)]

    vals = _target_to_numpy_values(y)
    vals = self._coerce_target_dtype(vals)

    # Native Polars support - no `.to_pandas()` copy. Production frames are 100+ GB; full materialization
    # would OOM. Use Polars-native ops when the input is pl.DataFrame.
    _is_polars_input = False
    try:
        import polars as pl  # local alias; safe even if pl is already imported module-scope

        _is_polars_input = isinstance(X, pl.DataFrame)
    except ImportError:
        pass

    # Track the caller-visible pandas frame so the ``finally`` below can always drop the injected target columns even if
    # ``fit`` raises mid-way (e.g. categorize_dataset / screen_predictors / cat-FE step). Pre-fix code dropped only on
    # the happy path, so a raised exception left ``targ_*`` columns on the caller's frame; downstream pipelines then
    # baked them into ``feature_names_in_`` and crashed on ``transform``.
    _caller_pandas_frame = None
    if _is_polars_input:
        # Polars is immutable; with_columns returns a new frame sharing buffers with X - no data copy.
        target_series = [pl.Series(name, vals[:, i] if vals.ndim == 2 else vals) for i, name in enumerate(target_names)]
        X = X.with_columns(target_series)
    else:
        # Multilabel target (N, K): pass through unchanged so each column maps to its target_names entry.
        # Previous .reshape(-1, 1) only worked for 1-D y; crashed on multilabel with "Must have equal len keys
        # and value when setting with an ndarray".
        _caller_pandas_frame = X
        if vals.ndim == 2:
            X.loc[:, target_names] = vals
        else:
            X.loc[:, target_names] = vals.reshape(-1, 1)
        # Register cleanup with the public ``fit`` wrapper so any later raise still strips ``targ_*``.
        self._pandas_frame_for_target_cleanup = _caller_pandas_frame
        self._target_names_for_cleanup = list(target_names)
    return X, _is_polars_input, target_names
