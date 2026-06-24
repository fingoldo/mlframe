"""``MRMR.validate*`` + ``MRMR.transform`` + helpers for ``mlframe.feature_selection.filters.mrmr``.

Split out of ``mrmr.py`` to keep the parent below the 1k-line monolith
threshold. The methods here are bound onto the ``MRMR`` class at the
parent's module bottom so ``self.transform(...)`` /
``self._validate_inputs(...)`` call sites continue to work unchanged.

Eagerly imports the parent module's ``MRMR`` class and shared symbols.
Safe: the parent loads ``MRMR`` BEFORE the bottom-of-module
``from ._mrmr_validate_transform import ...`` binding, so every name
referenced here is already on the parent at the time this sibling loads.
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")

# Recipe kinds whose ``src_names`` reference raw input columns only and never a chained engineered intermediate. For these, a source absent
# from the transform-time frame is a recipe-vs-X mismatch (raise); for every other kind a missing unproducible source is treated as a pruned
# chained producer and degrades to a NaN column. Keep additions here when a new raw-input-only consumer kind is added.
_RAW_SEED_ONLY_RECIPE_KINDS = frozenset({"mi_greedy_transform"})


def _validate_string_params(self):
    """Raise ValueError on bad constructor strings. Each branch lists the
    accepted values verbatim so the error message is actionable. fix audit
    row FS-P2-1."""
    _checks = (
        ("quantization_method", self._VALID_QUANTIZATION_METHODS),
        ("nan_strategy", self._VALID_NAN_STRATEGIES),
        ("mrmr_relevance_algo", self._VALID_MRMR_RELEVANCE_ALGOS),
        ("mrmr_redundancy_algo", self._VALID_MRMR_REDUNDANCY_ALGOS),
        ("fe_unary_preset", self._VALID_FE_UNARY_PRESETS),
        ("fe_binary_preset", self._VALID_FE_BINARY_PRESETS),
        ("cluster_aggregate_mode", self._VALID_CLUSTER_AGGREGATE_MODES),
        ("nbins_strategy", self._VALID_NBINS_STRATEGIES),
        ("mi_correction", self._VALID_MI_CORRECTIONS),
        ("redundancy_aggregator", self._VALID_REDUNDANCY_AGGREGATORS),
        ("stability_selection_method", self._VALID_STABILITY_SELECTION_METHODS),
        # 2026-05-30 Wave 9 — DCD distance / swap-method strings.
        ("dcd_distance", self._VALID_DCD_DISTANCES),
        ("dcd_swap_method", self._VALID_DCD_SWAP_METHODS),
        # additional_rfecv_selection_rule flows verbatim into RFECV's
        # n_features_selection_rule; validate it here so a typo fails at
        # fit() start, consistent with the other MRMR string params.
        ("additional_rfecv_selection_rule", self._VALID_RFECV_SELECTION_RULES),
    )
    # 2026-05-30 Wave 9 — DCD range checks gated on dcd_enable.
    if bool(getattr(self, "dcd_enable", False)):
        _d = getattr(self, "dcd_distance", "su")
        _tau_raw = getattr(self, "dcd_tau_cluster", 0.7)
        # Layer 47 (2026-05-31): ``dcd_tau_cluster='auto'`` opts into the
        # per-fit bimodality-detection calibration sweep in
        # ``make_dcd_state``. The string accepts only the literal lower-case
        # ``"auto"``; any other string is a configuration error.
        if isinstance(_tau_raw, str):
            if _tau_raw.lower() != "auto":
                raise ValueError(
                    f"MRMR: dcd_tau_cluster must be a float in (0, 1] or the "
                    f"literal string 'auto'; got {_tau_raw!r}."
                )
            # 'auto' string short-circuits the numeric range check.
            _tau = None
        else:
            _tau = float(_tau_raw)
            # Layer 46 (2026-05-31): ``"auto"`` (distance) returns max(SU, VI_sim) so the
            # score lives in [0, 1] just like SU; reuse the SU range check.
            if _d in ("su", "auto") and not (0.0 < _tau <= 1.0):
                raise ValueError(
                    f"MRMR: dcd_tau_cluster must be in (0, 1] for "
                    f"distance={_d!r}; got {_tau}."
                )
            if _d in ("vi", "sotoca_pla") and _tau <= 0.0:
                raise ValueError(
                    f"MRMR: dcd_tau_cluster must be > 0 for distance={_d!r}; "
                    f"got {_tau}."
                )
        # 2026-05-31 Layer 42: lower bound from 2 to 1. The threshold counts
        # cluster MEMBERS (not anchor + members), so threshold=1 fires the
        # PC1 swap on the strict 2-feature redundancy case (anchor + 1
        # perfect duplicate); threshold=2 (the new default) fires only when
        # the cluster grew anchor + >=2 members. Both are sane settings.
        if int(getattr(self, "dcd_cluster_size_threshold", 2)) < 1:
            raise ValueError(
                f"MRMR: dcd_cluster_size_threshold must be >= 1; got "
                f"{self.dcd_cluster_size_threshold}."
            )
        if float(getattr(self, "dcd_swap_gain_threshold", 0.05)) < 0.0:
            raise ValueError(
                f"MRMR: dcd_swap_gain_threshold must be >= 0; got "
                f"{self.dcd_swap_gain_threshold}."
            )
        _alpha = float(getattr(self, "dcd_swap_alpha", 0.05))
        if not (0.0 < _alpha <= 1.0):
            raise ValueError(
                f"MRMR: dcd_swap_alpha must be in (0, 1]; got {_alpha}."
            )
        if (bool(getattr(self, "dcd_postoc_compose", False)) and
                bool(getattr(self, "cluster_aggregate_enable", True))):
            import warnings as _w_dcd
            _w_dcd.warn(
                "MRMR: dcd_enable=True AND cluster_aggregate_enable=True AND "
                "dcd_postoc_compose=True will double-aggregate clusters. The "
                "post-hoc step will see almost no clusters DCD did not already "
                "process. Consider dcd_postoc_compose=False (the default).",
                UserWarning, stacklevel=3,
            )
    # 2026-05-29 Wave 7: AccuracyWarning for demoted nbins_strategy options.
    _demoted = getattr(self, "_DEMOTED_NBINS_STRATEGIES", ())
    _nbins_strat = getattr(self, "nbins_strategy", None)
    if _nbins_strat in _demoted:
        import warnings as _w
        _w.warn(
            f"MRMR: nbins_strategy={_nbins_strat!r} is DEMOTED to research-only. "
            f"F1-bench honest ranking by ``|err vs truth| + noise_floor`` "
            f"places these demoted methods last: Knuth (combined 0.213), "
            f"Bayesian Blocks (0.233), MAH/SCI (0.373, collapses to ~2 bins). "
            f"Recommended: 'mdlp' (combined 0.107, only TRUE zero noise floor) "
            f"for balanced production use; 'qs' (signal err 0.093 best, but "
            f"noise floor 0.123 inflates false positives) when no-signal "
            f"columns are absent. Opt-out via ``warnings.filterwarnings('ignore', "
            f"category=UserWarning, module='mlframe.feature_selection.filters')``.",
            UserWarning,
            stacklevel=3,
        )
    for _name, _valid in _checks:
        _val = getattr(self, _name, None)
        if _val is None:
            continue
        if not isinstance(_val, str):
            raise ValueError(
                f"MRMR: {_name} must be a string; got {type(_val).__name__}={_val!r}. "
                f"Valid values: {_valid}."
            )
        if _val not in _valid:
            raise ValueError(
                f"MRMR: {_name}={_val!r} is not a recognised value. "
                f"Valid values: {_valid}."
            )
    # 2026-06-01 Layer 85 — validate the orth default-scorer routing flag.
    # Kept outside the ``_checks`` loop because the attribute lives on the
    # MRMR class as ``_VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS`` (longer name
    # than the constants reused by the loop). Invalid value -> ValueError
    # listing every accepted scorer so the message is actionable.
    _default_scorer = getattr(self, "fe_hybrid_orth_default_scorer", None)
    if _default_scorer is not None:
        _valid_scorers = getattr(
            self, "_VALID_FE_HYBRID_ORTH_DEFAULT_SCORERS", None,
        )
        if _valid_scorers is not None:
            if not isinstance(_default_scorer, str):
                raise ValueError(
                    f"MRMR: fe_hybrid_orth_default_scorer must be a string; "
                    f"got {type(_default_scorer).__name__}={_default_scorer!r}. "
                    f"Valid values: {_valid_scorers}."
                )
            if _default_scorer not in _valid_scorers:
                raise ValueError(
                    f"MRMR: fe_hybrid_orth_default_scorer={_default_scorer!r} "
                    f"is not a recognised value. "
                    f"Valid values: {_valid_scorers}."
                )
    # cluster_aggregate_methods is a sequence; validate each element.
    _methods = getattr(self, "cluster_aggregate_methods", None)
    if _methods is not None:
        for _m in _methods:
            if _m not in self._VALID_CLUSTER_AGGREGATE_METHODS:
                raise ValueError(
                    f"MRMR: cluster_aggregate_methods contains {_m!r}, not a recognised value. "
                    f"Valid values: {self._VALID_CLUSTER_AGGREGATE_METHODS}."
                )

# Input validation contract: explicit guards for memory-exhaustion shapes, malformed dtypes,
# +/-inf values, single-class y, and polars LazyFrame / Expr edge cases. Each guard raises ValueError or warns.
# All-constant features are NOT rejected here: zero-variance columns survive validation and surface as MI=0
# in the screening loop, which is the documented downstream behaviour.
def _validate_inputs(self, X, y):
    # Validate string-valued constructor params on every fit. We intentionally
    # do NOT validate inside __init__ to preserve sklearn-style "no work in
    # __init__" semantics (clone() must not raise).
    self._validate_string_params()
    import warnings as _w
    n_rows = getattr(X, "shape", (None,))[0]
    if n_rows is not None:
        n_cols = X.shape[1] if len(X.shape) > 1 else 1
        if n_rows == 0:
            raise ValueError("MRMR.fit: empty input (n_rows=0)")
        if n_rows == 1:
            raise ValueError("MRMR.fit: cannot fit on a single row")
        if isinstance(n_cols, int):
            # MRMR's binned-frame working set is roughly ``n_rows * n_cols * 4`` bytes (int32 per cell). The previous absolute 1e9 cell ceiling rejected datasets that comfortably fit in RAM on a modern 128 GB+ host while letting through wide-but-not-as-wide frames on a tiny 16 GB box. Compare to ``psutil.virtual_memory().available * 0.5`` -- half of free RAM is the standard "safe working set" headroom for one stage of the pipeline.
            _footprint_bytes = n_rows * n_cols * 4
            try:
                import psutil as _psutil
                _available_bytes = int(_psutil.virtual_memory().available)
            except Exception:
                _available_bytes = 0
            _headroom_bytes = _available_bytes // 2
            if _headroom_bytes > 0 and _footprint_bytes > _headroom_bytes:
                raise ValueError(
                    f"MRMR.fit: refusing to allocate for n*p={n_rows * n_cols:_} "
                    f"(~{_footprint_bytes / 1e9:.2f} GB int32 working set) on a host with "
                    f"{_available_bytes / 1e9:.2f} GB available RAM; threshold is half of available "
                    f"(~{_headroom_bytes / 1e9:.2f} GB). Subsample, split the dataset, or free RAM "
                    "headroom before fitting."
                )
    if self.quantization_nbins > 1000:
        raise ValueError(f"quantization_nbins={self.quantization_nbins} > 1000 likely OOMs")
    if self.interactions_max_order > 5:
        raise ValueError(f"interactions_max_order={self.interactions_max_order} > 5 explodes combinatorially")
    if getattr(self, "fe_max_steps", 0) > 20:
        raise ValueError(f"fe_max_steps={self.fe_max_steps} > 20 unlikely to converge")
    # Polars edge cases.
    try:
        import polars as _pl
        if isinstance(X, _pl.LazyFrame):
            _w.warn("MRMR.fit autocollecting LazyFrame; pass DataFrame to skip this copy.", stacklevel=3)
            X = X.collect()
        if hasattr(_pl, "Expr") and isinstance(X, _pl.Expr):
            raise ValueError("MRMR.fit cannot accept polars Expr; materialise via .select(...) first")
        if isinstance(X, _pl.DataFrame):
            # Polars struct columns are not supported.
            struct_cols = [name for name, dt in X.schema.items() if str(dt).startswith("Struct")]
            if struct_cols:
                raise ValueError(f"MRMR.fit: polars Struct columns not supported: {struct_cols}")
    except ImportError:
        pass
    # Pandas: duplicate column names.
    if hasattr(X, "columns"):
        cols = list(X.columns)
        if len(cols) != len(set(cols)):
            from collections import Counter
            dups = [c for c, n in Counter(cols).items() if n > 1]
            raise ValueError(f"MRMR.fit: duplicate column names not supported: {dups}")
    # Numeric-column extraction for NaN / Inf validation. Object-dtype frames (numeric + cat/string mixed) used to slip through because the whole frame was
    # converted to object-dtype where dtype.kind != "f"; scan numeric columns explicitly instead.
    try:
        try:
            import polars as _pl
            if isinstance(X, _pl.DataFrame):
                _num_cols = [n for n, d in X.schema.items() if d.is_numeric()]
                if _num_cols:
                    _arr = X.select(_num_cols).to_numpy()
            elif hasattr(X, "select_dtypes"):
                _arr = X.select_dtypes(include=["number"]).to_numpy()
            else:
                _arr = X.to_numpy()
        except ImportError:
            _arr = X.to_numpy() if hasattr(X, "to_numpy") else None
        if _arr is not None and _arr.dtype.kind == "f":
            if np.isinf(_arr).any():
                raise ValueError(
                    "MRMR.fit: input X contains +/-inf values. Replace or drop these rows before fitting; the discretization step produces undefined bins on inf."
                )
            # NaN is allowed and routed through `self.nan_strategy` (default
            # "separate_bin": NaN rows get an honest dedicated bin instead of
            # being merged into bin-0 or imputed silently). transform()
            # preserves NaN in the returned X for downstream NaN-aware models
            # (catboost, lightgbm, xgboost histogram tree).
        # Object-dtype columns can smuggle a Python ``float('inf')`` past the float-only scan above (they are excluded from select_dtypes("number")
        # and the polars numeric set). Scan them too when present so the same undefined-bin failure is caught at the boundary, not deep in discretisation.
        if hasattr(X, "select_dtypes"):
            _obj = X.select_dtypes(include=["object"])
            if _obj.shape[1] > 0:
                _obj_arr = _obj.to_numpy()
                _obj_floats = np.frompyfunc(lambda v: isinstance(v, float) and np.isinf(v), 1, 1)(_obj_arr).astype(bool)
                if _obj_floats.any():
                    raise ValueError(
                        "MRMR.fit: input X contains +/-inf values in object-dtype column(s). Replace or drop these rows before fitting; the discretization step produces undefined bins on inf."
                    )
    except ValueError:
        raise  # re-raise our own ValueError
    except Exception:
        logger.debug("MRMR.fit: inf/NaN input validation scan failed unexpectedly; skipping the guard.", exc_info=True)
    # All-same y: raise (symmetric with RFECV.fit's single-class y validation). Constant y has H(y)=0 so
    # every MI(X_j, y) = 0; the entire MRMR pipeline produces zero-information output.
    # Multilabel y is (N, K): require that AT LEAST ONE label column has variation
    # (a single dead label is normal; all dead labels means the whole y is constant).
    try:
        _y_arr = np.asarray(y)
        if _y_arr.ndim == 2:
            _per_col_unique = [
                len(np.unique(_y_arr[:, _j])) for _j in range(_y_arr.shape[1])
            ]
            _y_is_constant = max(_per_col_unique) == 1 if _per_col_unique else True
        else:
            _y_is_constant = len(np.unique(_y_arr)) == 1
        if _y_is_constant:
            raise ValueError(
                "MRMR.fit: target y has only 1 unique value. H(y)=0 "
                "so all features have MI(X_j, y)=0 by construction. "
                "Drop or rebuild y before fitting."
            )
    except ValueError:
        raise  # re-raise our own ValueError
    except Exception:
        logger.debug("MRMR.fit: constant-y validation scan failed unexpectedly; skipping the guard.", exc_info=True)
    return X



def transform(self, X, y=None):
    """Apply the fitted MRMR selection + engineered-recipe replay to ``X``.

    Returns the column-subset / engineered-frame matching the layout MRMR
    produced at fit time. Raises :class:`sklearn.exceptions.NotFittedError`
    if called before :meth:`MRMR.fit`. Bound onto the ``MRMR`` class at the
    parent module's bottom so ``self.transform(X)`` call sites work
    unchanged.
    """
    # Lazy import: ``.mrmr`` re-imports this module at its bottom for method
    # binding -> any top-level ``from .mrmr import ...`` here creates a hard
    # import cycle that ``tests/test_meta/test_no_import_cycles.py`` flags.
    from .mrmr import ENSURE_ARROW_DF_SUPPORT
    # Unfitted -> NotFittedError (sklearn-canonical); previously returned X unchanged, masking config bugs.
    if not hasattr(self, "support_") or not hasattr(self, "feature_names_in_"):
        from sklearn.exceptions import NotFittedError
        raise NotFittedError(
            "This MRMR instance is not fitted yet. Call 'fit' before "
            "using 'transform'."
        )
    # 2026-05-30 Wave 9.1 fix (loop iter 19): sklearn ``n_features_in_``
    # contract. Pre-fix ``transform()`` accepted ndarray (and any non-
    # DataFrame array) with wrong column count and silently sliced
    # ``X[:, support_]`` from whatever positions support_ pointed at,
    # returning garbage columns. Confirmed live: fit on 4-col DataFrame
    # then transform 3-col, 5-col, 7-col ndarrays all returned shape
    # (n, k) with no error or warning. The pandas path's
    # column-name validation at lines 297-308 protected the DataFrame
    # surface but the positional ndarray path was naked. sklearn's
    # canonical contract requires raising ValueError on shape mismatch.
    _n_features_in = getattr(self, "n_features_in_", None)
    if _n_features_in is None:
        # Fallback to feature_names_in_ length for legacy estimators
        # missing the n_features_in_ attribute.
        _n_features_in = (
            len(self.feature_names_in_)
            if hasattr(self, "feature_names_in_") else None
        )
    if _n_features_in is not None and hasattr(X, "shape") and len(X.shape) >= 2:
        if int(X.shape[1]) != int(_n_features_in):
            # When X is a pandas / polars DataFrame the column-name validation
            # at lines ~297-308 raises a more actionable ``RuntimeError`` that
            # names the missing columns. Skip the bare shape check on named-
            # column frames so the column-name path can fire (the wrappers
            # audit + edge-coverage tests pin RuntimeError on column drift);
            # for plain ndarrays without column names this is the only signal.
            _is_named_frame = (
                (pd is not None and isinstance(X, pd.DataFrame))
                or hasattr(X, "schema")  # polars DataFrame / LazyFrame
            )
            if not _is_named_frame:
                raise ValueError(
                    f"X has {int(X.shape[1])} features, but MRMR is expecting "
                    f"{int(_n_features_in)} features as input."
                )
    support = self.support_
    recipes = getattr(self, "_engineered_recipes_", [])

    # Fast-path: when MRMR selected every input column AND produced zero engineered recipes, transform()
    # is the identity. Return X unchanged to avoid a full-copy X[selected_cols] and to let the caller detect
    # the no-op (checked via ``_mlframe_identity_equivalent`` downstream).
    if not recipes and hasattr(X, "shape"):
        _support_arr = np.asarray(support)
        if len(_support_arr) > 0 and isinstance(_support_arr.flat[0], (bool, np.bool_)):
            _n_selected = int(np.count_nonzero(_support_arr))
        else:
            _n_selected = len(_support_arr)
        if _n_selected == X.shape[1]:
            return X

    # Empty-base-support: if no base AND no engineered recipes, return legacy empty output. Recipes but no
    # base falls through and only the engineered cols come out.
    if len(support) == 0 and not recipes:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, []]
        else:
            return X[:, np.array([], dtype=np.intp)]

    if isinstance(X, pd.DataFrame):
        if ENSURE_ARROW_DF_SUPPORT:
            # Use column names to support Arrow-backed DataFrames (from polars zero-copy conversion).
            # Arrow-backed DFs don't support .iloc[:, integer_array] reliably.
            selected_cols = [self.feature_names_in_[i] for i in support]
            # Fuzz-caught: in a multi-model suite where a fitted MRMR is reused across models, val_df passed
            # to transform can have a different column set than the train_df MRMR fit on (e.g. after
            # _filter_categorical_features narrowed the frame). Detect column drift explicitly so we raise
            # with actionable context instead of an unhelpful KeyError.
            missing = [c for c in selected_cols if c not in X.columns]
            if missing:
                # Raise on column drift (symmetric with RFECV.transform); silent intersection masked
                # downstream column-set bugs. Callers wanting degradation can catch and intersect themselves.
                raise RuntimeError(
                    f"MRMR.transform: {len(missing)}/{len(selected_cols)} "
                    f"selected columns missing from input X ({missing[:8]}). "
                    f"The fitted support_ no longer matches the input's "
                    f"physical columns; an upstream step (constant-col "
                    f"removal / imputer drop / OD filter) is mutating the "
                    f"column set BETWEEN fit and transform. Investigate."
                )
            base_out = X[selected_cols]
        else:
            base_out = X.iloc[:, support]
        return self._append_engineered(base_out, X, recipes)
    elif hasattr(X, "schema") and hasattr(X, "columns"):
        # Polars DataFrame. ``support`` indexes the FIT-TIME feature set positionally, but the polars frame
        # passed at transform time may be narrower / reordered (e.g. a multi-model suite reuses one fitted MRMR
        # across models after an upstream step narrowed the frame). Positional ``X[:, support]`` then indexes the
        # wrong columns or raises ``IndexError`` when a fit-time index exceeds the narrower input width. Mirror the
        # pandas-by-name branch: remap support -> fit-time names, validate them against the input, select by name.
        _support_idx = np.asarray(support)
        if _support_idx.dtype == bool:
            _support_idx = np.flatnonzero(_support_idx)
        elif not np.issubdtype(_support_idx.dtype, np.integer):
            _support_idx = _support_idx.astype(np.intp)
        selected_cols = [self.feature_names_in_[i] for i in _support_idx]
        missing = [c for c in selected_cols if c not in X.columns]
        if missing:
            raise RuntimeError(
                f"MRMR.transform: {len(missing)}/{len(selected_cols)} "
                f"selected columns missing from input X ({missing[:8]}). "
                f"The fitted support_ no longer matches the input's "
                f"physical columns; an upstream step (constant-col "
                f"removal / imputer drop / OD filter) is mutating the "
                f"column set BETWEEN fit and transform. Investigate."
            )
        base_out = X.select(selected_cols)
    else:
        # Plain ndarray: ``support`` indexes columns positionally (the shape check above guards the width), so it
        # must be an integer (or boolean) array. An EMPTY full-mode selection (all signal folded into engineered
        # recipes) stored via ``np.array([])`` is float64 and would raise ``IndexError: arrays used as indices must
        # be of integer (or boolean) type`` here -- harmless for fresh fits after the int64 dtype fix in
        # _mrmr_fit_impl, but old pickles can still carry a float empty array, so coerce defensively. Boolean masks
        # pass through untouched (np.intp cast would corrupt them) -- only non-bool arrays are normalised to int.
        _support_idx = np.asarray(support)
        if _support_idx.dtype != bool and not np.issubdtype(_support_idx.dtype, np.integer):
            _support_idx = _support_idx.astype(np.intp)
        base_out = X[:, _support_idx]

    out = self._append_engineered(base_out, X, recipes)
    # When X is polars and Pipeline has set_output(transform="pandas"), sklearn's PandasAdapter calls
    # pd.DataFrame(out, ...) which (unlike polars' own .to_pandas()) does NOT preserve Arrow-backed
    # dtypes: pl.Enum / pl.Categorical collapse to object; pl.Float32 turns to object-of-strings
    # ('1.23' as text), and downstream HGB/XGB/SimpleImputer raise "could not convert string to float".
    # Convert ourselves via polars' Arrow-preserving .to_pandas() whenever the consumer expects pandas.
    try:
        from sklearn.utils._set_output import _get_output_config
        _cfg = _get_output_config("transform", estimator=self)
        _want_pandas = (_cfg.get("dense") or "default") == "pandas"
    except Exception:
        _want_pandas = False
    if _want_pandas:
        try:
            import polars as _pl
            if isinstance(out, _pl.DataFrame):
                out = out.to_pandas()
        except ImportError:
            pass
    return out

def _append_engineered(self, base_out, X, recipes):
    """Append engineered-recipe columns onto ``base_out``.

    Inputs:
    - ``base_out``: DataFrame / ndarray already restricted to the base
      ``support_`` columns. Caller's dtype is preserved.
    - ``X``: full input frame (DataFrame, ndarray, or polars). Recipe
      replay reads source columns from here BY NAME, so X must
      contain at least every ``recipe.src_names`` entry.
    - ``recipes``: list of EngineeredRecipe to replay.

    Behaviour:
    - Returns ``base_out`` unchanged when ``recipes`` is empty (legacy
      path, zero overhead).
    - For pandas / polars input, engineered cols are appended as
      named columns.
    - For ndarray input, engineered cols are stacked as additional
      numeric columns; column names are not preserved (caller is
      expected to use ``get_feature_names_out`` for naming).
    """
    if not recipes:
        return base_out

    # Lazy import keeps import-time cost off MRMR users who never engage FE.
    from .engineered_recipes import apply_recipe

    # K-way recipes ship a chained-lookup payload (extras ``chain_lookups`` / ``chain_nuniqs``) so they
    # replay on test data alongside pair recipes. The only filter is the legacy ``requires_refit_for_replay``
    # flag retained for OLD pickles that pre-date the chain payload.
    replayable = [
        r for r in recipes
        if r.extra.get("chain_lookups") is not None
        or not r.extra.get("requires_refit_for_replay")
    ]
    if len(replayable) < len(recipes) and self.verbose:
        logger.info(
            "MRMR.transform: skipping %d legacy k-way recipe(s) "
            "without chained-lookup payload (pre-D3 pickle). Re-fit "
            "to materialise the chain.",
            len(recipes) - len(replayable),
        )
    if not replayable:
        return base_out
    recipes = replayable
    # Engineered-recipe replay resolves source columns by NAME (e.g. cluster_aggregate looks
    # up ``recipe.src_names`` via filters/engineered_recipes._extract_column). A plain 2-D
    # ndarray has no column-name index, so wrap it in a pandas DataFrame using fit-time
    # feature_names_in_ before replay. base_out is already projected so we keep its shape;
    # this view is only used for engineered-column resolution.
    _X_for_recipes = X
    if isinstance(X, np.ndarray) and X.dtype.names is None and hasattr(self, "feature_names_in_"):
        try:
            _X_for_recipes = pd.DataFrame(X, columns=list(self.feature_names_in_))
        except (ValueError, TypeError) as _wrap_err:
            # A length/shape mismatch means X does not match fit-time feature_names_in_. Falling back
            # to the nameless ndarray makes every src-names-by-name recipe resolve against an unnamed
            # frame -> wrong engineered columns. Surface it at WARNING instead of masking it silently.
            logger.warning(
                "MRMR.transform: could not wrap input ndarray in fit-time feature_names_in_ (%s: %s); "
                "engineered-recipe replay will run on an unnamed frame and may produce wrong columns.",
                type(_wrap_err).__name__, _wrap_err,
            )
            _X_for_recipes = X
    # Recipe replay must support multi-level chaining: a spline / fourier / hybrid
    # extra-basis recipe can carry ``src_names=('x__He2',)`` referencing a sibling
    # ``orth_univariate`` recipe rather than a raw input column. The fit-time pipeline
    # appends each level's columns to the augmented frame before the next level runs,
    # so the recipe order recorded in ``_engineered_recipes_`` is already a valid
    # topological order. We mirror that here: materialise each recipe's output into
    # a per-row working frame so subsequent recipes can resolve their ``src_names``.
    # Falls back transparently to the legacy single-pass behaviour for recipes whose
    # src_names point only at raw input columns.
    engineered_cols = []
    if isinstance(_X_for_recipes, pd.DataFrame):
        chained = _X_for_recipes
        # Dependency-aware replay: a chained recipe (e.g. modular-of-cross,
        # spline-on-He2) references an earlier engineered column via src_names.
        # The recorded order is USUALLY topological, but cross-family chaining
        # (cat_pair_cross consumed by a numeric_decompose / modular recipe) can
        # record the consumer before its producer. Resolve in passes: apply only
        # recipes whose src_names already exist in ``chained``, defer the rest,
        # loop until no progress -- then apply any genuinely-unresolvable
        # remainder in recorded order (surfaces a real missing-source KeyError
        # rather than an ordering artefact).
        _results: dict = {}
        _pending = list(recipes)
        _unresolved: set = set()

        def _unresolved_sources(r) -> list:
            """Source names a recipe still needs from ``chained`` that are NOT
            available. NESTED-ENGINEERED PARENTS (2026-06-08): a ``unary_binary``
            recipe whose operand is itself engineered carries that parent's recipe
            in ``extra['nested_parent_a'|'nested_parent_b']`` and recomputes it
            recursively in ``apply_recipe`` -- so that side is self-resolving and
            must NOT be treated as a missing column dependency (its engineered
            ``src_name`` is never present in the raw-only transform frame)."""
            _src = tuple(getattr(r, "src_names", ()) or ())
            _extra = getattr(r, "extra", {}) or {}
            _nested_by_pos = (_extra.get("nested_parent_a"), _extra.get("nested_parent_b"))
            out = []
            for _pos, s in enumerate(_src):
                if s in chained.columns:
                    continue
                if _pos < 2 and _nested_by_pos[_pos] is not None:
                    # Resolved recursively via the stored parent recipe -- BUT that parent
                    # may itself reference a SEPARATE engineered column (its own src) that
                    # is not yet in ``chained`` (e.g. a nested binned_numeric_agg whose
                    # group_col is an adaptive Fourier/chirp column produced by another
                    # recipe). Surface those TRANSITIVE deps so the scheduler waits for the
                    # producing recipe to replay first, instead of KeyError-ing on the
                    # missing column. (2026-06-21: exposed once orth-FE/extra-basis is ON
                    # by default and binned_agg feeds-forward on a chirp operand.)
                    out.extend(_unresolved_sources(_nested_by_pos[_pos]))
                    continue
                out.append(s)
            return out

        while _pending:
            _progress = False
            _still: list = []
            for r in _pending:
                if not _unresolved_sources(r):
                    col = apply_recipe(r, chained)
                    _results[r.name] = col
                    chained = chained.assign(**{r.name: col})
                    _progress = True
                else:
                    _still.append(r)
            if not _progress:
                for r in _still:
                    _missing = _unresolved_sources(r)
                    # Raise-vs-NaN is keyed on the recipe kind's data-flow contract. Raw-seed-only kinds (``mi_greedy_transform``) consume
                    # input columns exclusively -- a missing source is a genuine recipe-vs-X mismatch (corrupted/stale recipe, or wrong X)
                    # and must fail loudly and name the column. Chained-capable kinds (modular / numeric_decompose / numeric_rounding /
                    # orth_spline / cross families) may reference an engineered intermediate that fit-time pruning dropped its producer for;
                    # that source is unreconstructable at transform, so emit a NaN column rather than crash.
                    if r.kind in _RAW_SEED_ONLY_RECIPE_KINDS:
                        raise KeyError(
                            f"MRMR.transform: recipe {r.name!r} (kind={r.kind}) references "
                            f"source column(s) {_missing} that are absent from X. This kind consumes "
                            f"input columns only, so the recipe set does not match the input frame "
                            f"(corrupted/stale recipe, or wrong X)."
                        )
                    logger.warning(
                        "MRMR.transform: recipe %r (kind=%s) references unresolved engineered "
                        "source(s) %s; emitting a neutral zero column (feature effectively dropped from replay).",
                        r.name, r.kind, _missing,
                    )
                    # The producer of the engineered intermediate was pruned at fit time, so this chained recipe is
                    # unreconstructable. The output width is fixed by ``get_feature_names_out`` (the recipe count), so
                    # the column cannot be physically removed without a shape mismatch -- emit a zero-variance column
                    # instead. A NaN placeholder (the prior behaviour) propagates into every downstream estimator and
                    # hard-crashes the ones that reject NaN (LogisticRegression et al.); a constant 0.0 carries no
                    # signal (so it is "dropped" in effect) yet is accepted by every estimator.
                    _results[r.name] = np.zeros(len(chained), dtype=np.float64)
                    chained = chained.assign(**{r.name: _results[r.name]})
                    _unresolved.add(r.name)
                _still = []
            _pending = _still
        engineered_cols = [_results[r.name] for r in recipes]
    else:
        # ndarray / polars path: best-effort single pass (recipes that reference
        # engineered intermediates aren't expressible without a name-indexed frame).
        engineered_cols = [apply_recipe(r, _X_for_recipes) for r in recipes]
    if isinstance(base_out, pd.DataFrame):
        # ``copy=False`` would risk mutating caller's view (base_out is a view into pandas X). Build a narrow
        # new frame: engineered cols are fresh ndarrays anyway, only base cols share buffers with X.
        # Name the engineered output columns through the SAME value-preserving canonicaliser
        # get_feature_names_out uses (abs(div(sqr(a),neg(b))) -> abs(div(sqr(a),b))) so the two
        # stay byte-for-byte in sync (the all-or-nothing collision guard keeps widths equal). The
        # internal ``chained``/``_results`` replay above still keys off the RAW r.name, so replay
        # is unaffected. Build column-by-column (positional) to tolerate any duplicate display name.
        from .engineered_recipes._recipe_name_simplify import simplified_recipe_names
        _disp_names = simplified_recipe_names(recipes)
        engineered_df = pd.DataFrame(
            dict(zip(range(len(_disp_names)), engineered_cols)),
            index=base_out.index,
        )
        engineered_df.columns = _disp_names
        return pd.concat([base_out, engineered_df], axis=1)

    # Try polars first if available (avoid hard import).
    try:
        import polars as _pl
        if isinstance(base_out, _pl.DataFrame):
            return base_out.with_columns([
                _pl.Series(r.name, col) for r, col in zip(recipes, engineered_cols)
            ])
    except ImportError:
        pass

    # ndarray fallback: hstack engineered cols. Names are lost but row order matches get_feature_names_out.
    # 2026-05-30 Wave 9.1 fix (loop iter 15): promote BOTH sides to the
    # common dtype via np.result_type instead of forcing engineered cols
    # into ``base_out.dtype``. Pre-fix when base_out was an integer
    # ndarray (the common case for selected categorical / binned
    # features), every engineered recipe (target_encoding,
    # cluster_aggregate, hermite_pair, factorize_merge, ...) returning
    # float64 was silently truncated to 0 via int cast - downstream
    # models saw constant columns, AUC collapsed, no warning. The
    # pandas / polars paths above don't have this bug since they
    # preserve per-column dtype. fit_transform(ndarray) thus diverged
    # from fit(pd.DataFrame).transform(ndarray), violating sklearn
    # contract.
    engineered_arr = (
        np.column_stack(engineered_cols)
        if engineered_cols
        else np.empty((base_out.shape[0], 0))
    )
    common_dtype = np.result_type(base_out.dtype, engineered_arr.dtype)
    if base_out.size == 0:
        return engineered_arr.astype(common_dtype, copy=False)
    return np.hstack([
        base_out.astype(common_dtype, copy=False),
        engineered_arr.astype(common_dtype, copy=False),
    ])

