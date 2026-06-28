"""The main ``fit`` method for ``CompositeTargetDiscovery``.

Split out of ``composite_discovery.py`` to keep the parent below the 1k-line
monolith threshold. ``fit`` is bound back onto the
``CompositeTargetDiscovery`` class at the parent's module bottom, so call
sites that invoke ``disc.fit(...)`` continue to work unchanged.
"""

from __future__ import annotations

import logging
import os
import threading
from timeit import default_timer as timer
from typing import Any, Sequence

import numpy as np

from ..spec import CompositeSpec
from .forward_stepwise import forward_stepwise_multi_base
from .screening import (
    _aggregate_mi_per_feature,
    _aggregate_mi_per_feature_excluding,
    _extract_column_array,
    _is_polars_df,
    _mi_per_feature_knn,
    _mi_per_feature_prebinned,
    _mi_to_target,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
    _prebin_feature_columns_cached,
    _prebin_feature_columns_lazy,
    _sample_indices,
)
from ..transforms import (
    UnknownTransformError,
    _linear_residual_fit,
    _linear_residual_multi_fit,
    compose_target_name,
    get_transform,
)
from ._fit_ram import _phase_ram_report, _process_mem_mb  # noqa: F401 -- _process_mem_mb re-exported for back-compat
from ._eval import build_unary_base_context, eval_one_transform
from ._eval_stats import (
    apply_alpha_drift_gate,
    apply_fdr_control_to_candidates,
    apply_linear_residual_diff_collapse,
    near_collinear_keep_mask,
)

logger = logging.getLogger(__name__)

# Sentinel base key for the dedicated UNARY (``requires_base=False``)
# evaluation context. Unary transforms ignore the base column entirely, so they
# are scored ONCE against the FULL feature matrix (no base dropped) rather than
# bound to an arbitrary first base. The empty string is also the
# ``CompositeTargetEstimator`` default ``base_column`` for base-less specs, and
# ``compose_target_name(..., base="")`` renders the base-free 2-segment name.
_UNARY_BASE_SENTINEL = ""


def fit(
    self,
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    time_ordering: Any = None,
) -> "CompositeTargetDiscovery":  # noqa: F821 -- forward ref to parent class
    """Discover composite-target specs.

    Parameters
    ----------
    df
        Pandas or polars frame containing ``target_col`` and
        ``feature_cols`` as columns.
    target_col
        Column name of the regression target.
    feature_cols
        Candidate feature columns. Base candidates are drawn from
        this set when ``config.base_candidates="auto"``.
    train_idx
        Row indices to use for fitting transform params and
        scoring. **Required** -- no implicit "use full df" shortcut.
    val_idx, test_idx
        Stored on the instance for later integrity checks; never
        touched during fit.
    time_ordering
        Optional per-row sortable key (timestamps / a monotone index)
        aligned to ``df`` rows. When given, the MI-screening sample is
        SORTED by time so the tiny-model CV uses a forward-walk
        (TimeSeriesSplit) instead of a shuffled K-fold -- the canonical
        ``lag(y)`` base is non-monotone so the old base-monotonicity
        heuristic never fired and the screen leaked future->past on
        temporal data. ``None`` keeps the legacy base-monotonicity
        auto-detection.
    """
    if not self.config.enabled:
        self.specs_: list[CompositeSpec] = []
        self.report_: list[dict[str, Any]] = []
        self.train_idx_ = np.asarray(train_idx)
        self._df_ref = df
        self._target_col = target_col
        return self

    train_idx = np.asarray(train_idx)
    # A boolean mask is a common idiom but detonates later with a cryptic IndexError (sampling reads the mask LENGTH as the row count); normalise it up front and reject non-integer dtypes loudly.
    if train_idx.dtype == bool:
        train_idx = np.flatnonzero(train_idx)
    elif not np.issubdtype(train_idx.dtype, np.integer):
        raise TypeError("train_idx must be integer positions or a boolean mask, got dtype %r" % train_idx.dtype)

    def _normalise_idx(idx: Any, name: str) -> np.ndarray:
        arr = np.asarray(idx)
        if arr.dtype == bool:
            return np.flatnonzero(arr)
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError("%s must be integer positions or a boolean mask, got dtype %r" % (name, arr.dtype))
        return arr

    val_idx = None if val_idx is None else _normalise_idx(val_idx, "val_idx")
    test_idx = None if test_idx is None else _normalise_idx(test_idx, "test_idx")

    # Leakage discipline: the class documents val/test integrity as its core discipline but fit performed no check; overlapping train/test rows silently fit params + MI screens on holdout. O(n log n), negligible vs MI screening.
    if test_idx is not None and np.intersect1d(train_idx, test_idx).size:
        raise ValueError("[CompositeTargetDiscovery] train_idx overlaps test_idx -- leakage.")
    if val_idx is not None and np.intersect1d(train_idx, val_idx).size:
        raise ValueError("[CompositeTargetDiscovery] train_idx overlaps val_idx -- leakage.")
    if np.unique(train_idx).size != train_idx.size:
        logger.warning("[CompositeTargetDiscovery] duplicated train_idx rows bias MI estimates.")
    if train_idx.size and int(train_idx.max()) >= len(df):
        raise ValueError("[CompositeTargetDiscovery] train_idx max %d out of bounds for df of %d rows." % (int(train_idx.max()), len(df)))

    # Stash the identifiers BEFORE the early-return paths so
    # _filter_features (which reads ``self._target_col``) and
    # iter_transform (which reads ``self._df_ref``) work even on
    # the no-spec degenerate cases.
    self._target_col = target_col
    self._df_ref = df

    # Post-selection-inference holdout (winner's-curse de-bias, SA27): carve a never-touched
    # holdout BEFORE screening, then REBIND ``train_idx`` to the screening pool so every
    # downstream consumer is holdout-excluded with no per-site change (carve_screening_holdout).
    from ._honest_holdout import carve_screening_holdout

    train_idx, _honest_holdout_idx = carve_screening_holdout(self, train_idx)

    if train_idx.size < 50:
        logger.warning(
            "[CompositeTargetDiscovery] train_idx has only %d rows; " "MI estimates unreliable. Discovery yields no specs.",
            train_idx.size,
        )
        self.specs_ = []
        self.report_ = []
        return self

    t0 = timer()
    # Per-fit() RAM telemetry state. Each sub-phase report logs delta vs
    # prev + cumulative vs entry; opt out by setting
    # MLFRAME_DISCOVERY_RAM_PROFILER=0 (the helper checks the env once at
    # entry so the rest of the fit() body never tests the flag again).
    import os as _os

    _ram_state: dict = {}
    _ram_profiler_on = _os.environ.get("MLFRAME_DISCOVERY_RAM_PROFILER", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "entry")

    # Pull target on train rows. We never touch val/test.
    y_full = _extract_column_array(df, target_col)
    y_train = y_full[train_idx]

    # Auto-boost mi_n_strata on heavy-tail y. The
    # default 10 strata produces unstable MI estimates when the
    # tail dominates the signal -- one or two tail rows per bin.
    # Detect via skew or kurtosis on train; if either is high,
    # bump n_strata to ``mi_n_strata_heavy_tail`` (default 30).
    # User-configured ``mi_n_strata`` remains the floor; we only
    # bump when the auto-detected boost is HIGHER.
    y_finite_for_check = y_train[np.isfinite(y_train)]
    if y_finite_for_check.size >= 100:
        y_std = float(y_finite_for_check.std())
        if y_std > 1e-12:
            z_centered = (y_finite_for_check - y_finite_for_check.mean()) / y_std
            # z**3 / z**4 via chained mul over np.power dispatch (~3x;
            # same antipattern as iter138 _target_distribution_analyzer).
            z2 = z_centered * z_centered
            skew = float(np.mean(z2 * z_centered))
            kurt = float(np.mean(z2 * z2) - 3.0)
            if abs(skew) > 2.0 or kurt > 5.0:
                boost = int(
                    getattr(
                        self.config,
                        "mi_n_strata_heavy_tail",
                        30,
                    )
                )
                cur_n_strata = int(
                    getattr(
                        self.config,
                        "mi_n_strata",
                        10,
                    )
                )
                if boost > cur_n_strata:
                    # Mutate config in-place ONLY if we own a
                    # copy (avoid leaking into callers' shared
                    # config). model_copy is safe here because
                    # discovery already gets a per-target config
                    # clone in core.py when hint is enabled.
                    try:
                        new_cfg = self.config.model_copy(update={"mi_n_strata": boost})
                        self.config = new_cfg
                        logger.info(
                            "[CompositeTargetDiscovery] heavy-tail y " "detected (skew=%.2f, kurt=%.2f); boosted " "mi_n_strata %d -> %d.",
                            skew,
                            kurt,
                            cur_n_strata,
                            boost,
                        )
                    except Exception:
                        pass  # leave at user-configured value.

    # Filter feature_cols by name patterns AND constancy on train.
    usable_features = self._filter_features(df, feature_cols, y_train, train_idx)
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "filter_features_done")

    # Resolve base candidates.
    base_candidates = self._resolve_base_candidates(
        df,
        target_col,
        usable_features,
        y_train,
        train_idx,
    )
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "resolve_base_candidates_done")

    # Pre-discovery base-target leakage guard (config.detect_base_leakage); see _fit_temporal.apply_base_leakage_guard.
    if getattr(self.config, "detect_base_leakage", True) and time_ordering is not None and base_candidates:
        from ._fit_temporal import apply_base_leakage_guard

        base_candidates = apply_base_leakage_guard(self, df, base_candidates, train_idx, y_train, time_ordering)

    if not base_candidates:
        logger.warning(
            "[CompositeTargetDiscovery] no usable base candidates after " "forbidden-pattern / corr / ptp / numeric filters. " "Discovery yields no specs."
        )
        self.specs_ = []
        self.report_ = []
        return self

    # Down-sample for MI screening. Stratified-quantile when
    # configured -- guarantees per-bin coverage on heavy-tail y.
    sample_idx = _sample_indices(
        train_idx.size,
        self.config.mi_sample_n,
        self.config.random_state,
        strategy=getattr(self.config, "mi_sample_strategy", "random"),
        y=y_train,
        n_strata=getattr(self.config, "mi_n_strata", 10),
    )
    train_idx_screen = train_idx[sample_idx]

    # Time-awareness: when the caller supplies an explicit ``time_ordering``,
    # SORT the screening sample into time order so the downstream tiny-model CV
    # is a genuine forward-walk (TimeSeriesSplit). The old heuristic inferred
    # time-awareness from base MONOTONICITY, which never fires for the canonical
    # non-monotone ``lag(y)`` base -> shuffled K-fold leaked future->past on
    # temporal data. Sorting once here makes every per-spec / raw-baseline /
    # stepwise tiny-CV time-correct without per-call ordering logic.
    from ._fit_temporal import order_screen_by_time

    train_idx_screen, sample_idx, self._screen_time_ordered_ = order_screen_by_time(train_idx_screen, sample_idx, time_ordering)
    y_screen = y_full[train_idx_screen]

    # Bin-MI floors every value to 0.0 when the screening sample has fewer than
    # 5*nbins finite rows (joint-histogram cells too sparse), so top-K ranking
    # silently degenerates to the rerank/alphabetical tiebreaker. Warn rather
    # than auto-shrink nbins (which would change the MI numerics).
    if self.config.mi_estimator == "bin":
        _eff_n = int(train_idx_screen.size)
        _min_n = 5 * int(self.config.mi_nbins)
        if _eff_n < _min_n:
            logger.warning(
                "[CompositeTargetDiscovery] screening sample %d < 5*mi_nbins(%d): "
                "bin-MI is inactive (all 0.0); spec ranking is deferred to the "
                "rerank/tiebreaker. Raise mi_sample_n or lower mi_nbins.",
                _eff_n,
                int(self.config.mi_nbins),
            )

    # mi_y baseline is computed PER-BASE because the X-without-base
    # feature set differs per candidate. Comparing MI(T, X_no_base)
    # against MI(y, X) (full X) confounds two effects: target
    # transformation AND removal of the dominant feature. We want
    # only the first effect, so both halves use the same feature
    # set: X without the base column.

    # Stash the per-candidate base arrays so the multi-base forward-stepwise extension (run after kept_specs is finalised) can pick from the SAME pool of MI-ranked bases that the single-base discovery considered. Keyed by column name; values are train-row-restricted ndarrays.
    self._auto_base_pool: dict[str, np.ndarray] = {}

    # Score each (base, transform).
    # Unary y-transforms (``requires_base=False``) ignore the base column, so each routes through ONE dedicated context (``_UNARY_BASE_SENTINEL``) scored against the FULL feature matrix (no base dropped) with an empty-string, base-free spec name -- not bound to / scored against / named after an arbitrary "first" base as before (which made the unary's mi_gain shift with irrelevant auto-base ranking and claim a nonexistent base dependence). The set tracks which unary names are already evaluated so later base-loop iterations skip the redundant re-fit; bivariate + chain transforms still iterate per base.
    _unary_evaluated: set[str] = set()

    candidates: list[dict[str, Any]] = []

    # Hoist the per-base setup OUT of the candidate
    # evaluation loop so a single parallel dispatch can span all
    # (base, transform) pairs. The previous serial outer loop over
    # bases bottlenecked total parallelism at ``n_transforms`` per
    # base; flattening lifts the cap to ``sum(transforms_per_base)``
    # and lets ``discovery_n_jobs`` saturate even when a single base
    # has fewer eligible transforms than CPU cores. Per-base setup
    # itself (column extraction, X-without-base matrix, pre-binning,
    # ``mi_y_for_base``) stays serial because it writes
    # ``self._auto_base_pool`` and is cheap relative to MI compute.
    # Build the full screen-sized feature matrix ONCE across all bases, then
    # per-base slice out the base column via np.delete. Avoids 10x polars->numpy
    # column extraction (and 10x prebinning if mi_estimator='bin') on the same
    # usable_features set. The polars columns themselves don't change between
    # iterations - only the choice of which one is the "base" does.
    _usable_features_list = list(usable_features)
    _col_index = {c: i for i, c in enumerate(_usable_features_list)}
    _bin_estimator = self.config.mi_estimator == "bin"
    _dedup_x_remaining = bool(getattr(self.config, "dedup_x_remaining_for_mi_baseline", True))
    # Lazy-prebin gate: on the bin estimator the downstream MI uses only the
    # int16/int32 CODE matrix -- the float32 (n, F) plane feeds nothing but the
    # prebinning itself (and dedup, when on). On a POLARS carrier large enough
    # that the float plane is the dominant transient we can therefore skip
    # building it entirely: pull + bin one column at a time
    # (``_prebin_feature_columns_lazy``) so peak extra RAM is ONE column, not
    # the whole plane. BIT-IDENTICAL codes (shared per-column kernel). Gated to
    # bin + dedup-off (dedup needs the float matrix) + polars + a size floor;
    # ndarray / small / knn / dedup-on inputs keep the eager path. Override via
    # MLFRAME_DISCOVERY_LAZY_PREBIN=0|1 (force off / on; ignores the gate).
    _lazy_force = os.environ.get("MLFRAME_DISCOVERY_LAZY_PREBIN", "").strip().lower()
    _lazy_n_floor = int(os.environ.get("MLFRAME_DISCOVERY_LAZY_PREBIN_MIN_N", "50000"))
    _lazy_eligible = _bin_estimator and not _dedup_x_remaining and _is_polars_df(df) and len(_usable_features_list) > 0
    if _lazy_force in ("0", "false", "no", "off"):
        _use_lazy_prebin = False
    elif _lazy_force in ("1", "true", "yes", "on"):
        _use_lazy_prebin = _lazy_eligible
    else:
        _use_lazy_prebin = _lazy_eligible and train_idx_screen.size >= _lazy_n_floor
    _prebin_use_cache = os.environ.get("MLFRAME_PREBIN_CACHE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    if _use_lazy_prebin:
        # Defer column extraction: never materialise the (n, F) float plane.
        _full_x_matrix = None
        _full_x_prebinned = _prebin_feature_columns_lazy(
            df,
            _usable_features_list,
            train_idx_screen,
            nbins=int(self.config.mi_nbins),
        )
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "lazy_prebin_features_done")
    else:
        _full_x_matrix = self._build_feature_matrix(
            df,
            _usable_features_list,
            train_idx_screen,
        )
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "build_full_x_matrix_done")
        # Cache-consulting prebin: codes are deterministic on (matrix bytes, nbins), so a re-discovery
        # on the SAME screen sample + nbins with a different config (transforms / rerank / re-enabled
        # bin estimator) reuses the bit-identical codes instead of recomputing the per-column quantile
        # binning. Opt out via MLFRAME_PREBIN_CACHE=0 (force fresh recompute, no store).
        _full_x_prebinned = (
            _prebin_feature_columns_cached(
                _full_x_matrix,
                nbins=int(self.config.mi_nbins),
                use_cache=_prebin_use_cache,
            )
            if _bin_estimator
            else None
        )
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "prebin_features_done")

    # Per-feature MI(y, x_j) is INDEPENDENT of which base column is excluded, so
    # compute the full-feature vector ONCE and derive each base's mi_y by
    # excluding that base's column (mean/sum over the survivors) -- instead of
    # re-binning + re-MI'ing the shared columns per base candidate. Bit-identical
    # on BOTH the prebinned (mi_estimator='bin') path (_per_feat_y_full below) and
    # the knn path (_per_feat_y_knn_full below).
    _mi_aggregation = getattr(self.config, "mi_aggregation", "mean")
    _per_feat_y_full = (
        _mi_per_feature_prebinned(
            _full_x_prebinned,
            y_screen,
            nbins=int(self.config.mi_nbins),
        )
        if _full_x_prebinned is not None
        else None
    )
    # knn analogue: per-column MI(y, x_j) is likewise base-invariant, but the Kraskov estimator dominates
    # wall time (~0.45s/column at the 100k screen sample), so re-running the full per-column sweep per base
    # candidate is the dominant redundant cost on the knn path. Compute the vector ONCE over the full float
    # matrix and derive each base's mi_y by aggregating over its surviving (base-dropped, dedup-kept) original
    # column indices -- bit-identical because each column's MI is independent of which others are present.
    _per_feat_y_knn_full = (
        _mi_per_feature_knn(
            _full_x_matrix,
            y_screen,
            n_neighbors=self.config.mi_n_neighbors,
            random_state=self.config.random_state,
        )
        if (not _bin_estimator and _full_x_matrix is not None)
        else None
    )

    # Dedup ``x_remaining`` before the MI baseline. A near-duplicate
    # sibling of the removed base inflates ``MI(y, x_remaining)`` (it re-carries
    # the base's info) without helping ``MI(T, x_remaining)``, biasing
    # ``mi_gain`` DOWN for exactly the lag-family bases discovery wants. The
    # keep-mask is computed per base on the (base-dropped) screen matrix and
    # applied identically to ``x_remaining_matrix`` / ``_x_prebinned`` and the
    # decomposed per-feature MI vector so both halves of ``mi_gain`` score the
    # same de-duplicated feature set. Gated + threshold-tunable via config; a
    # strict no-op when no surviving pair exceeds the threshold.
    _dedup_corr_thr = float(getattr(self.config, "dedup_x_remaining_corr_threshold", 0.99))
    _base_contexts: dict[str, dict[str, Any]] = {}
    for base in base_candidates:
        base_train = _extract_column_array(df, base)[train_idx]
        self._auto_base_pool[base] = base_train
        base_screen = base_train[sample_idx]
        if base in _col_index:
            _drop_idx = _col_index[base]
            _x_prebinned = np.delete(_full_x_prebinned, _drop_idx, axis=1) if _full_x_prebinned is not None else None
            if _use_lazy_prebin:
                # No float plane on the lazy path -- the base-dropped float matrix
                # is never read by the bin-estimator eval (it consumes only the
                # prebinned codes). Carry a zero-row float32 proxy of the right
                # WIDTH so the ``x_remaining_matrix.shape[1]`` index/empty checks
                # and the eval body's shape reads stay correct without allocating
                # the (n, F-1) plane. dedup is off on this gate, so the float
                # values are provably unused.
                _rem_cols = _x_prebinned.shape[1] if _x_prebinned is not None else 0
                x_remaining_matrix = np.empty((0, _rem_cols), dtype=np.float32)
            else:
                x_remaining_matrix = np.delete(_full_x_matrix, _drop_idx, axis=1)
            # Original-column indices that survive base-drop (used to derive the knn mi_y baseline from the
            # precomputed base-invariant per-feature vector); dedup prunes this in lockstep with x_remaining_matrix.
            _surviving_orig_idx = np.delete(np.arange(_full_x_matrix.shape[1]), _drop_idx) if _full_x_matrix is not None else None
            if _dedup_x_remaining and not _use_lazy_prebin and x_remaining_matrix.shape[1] > 1:
                _keep = near_collinear_keep_mask(
                    x_remaining_matrix,
                    corr_threshold=_dedup_corr_thr,
                )
                if not _keep.all():
                    x_remaining_matrix = x_remaining_matrix[:, _keep]
                    if _x_prebinned is not None:
                        _x_prebinned = _x_prebinned[:, _keep]
                    if _surviving_orig_idx is not None:
                        _surviving_orig_idx = _surviving_orig_idx[_keep]
        else:
            # Invariant: every base in usable_features is in _col_index, so this arm is unreachable; keeping the base in its own x_remaining would leak it into the MI baseline, so skip rather than mis-score.
            logger.error(
                "[CompositeTargetDiscovery] invariant violated: base %r not in usable_features index; skipping (would leak base into its own x_remaining).",
                base,
            )
            continue
        if x_remaining_matrix.shape[1] == 0:
            continue
        _mi_kwargs: dict[str, Any] = dict(
            nbins=int(self.config.mi_nbins),
            aggregation=getattr(self.config, "mi_aggregation", "mean"),
        )
        if _x_prebinned is not None:
            if _per_feat_y_full is not None and base in _col_index:
                # Decompose: aggregate the precomputed per-feature MI over all
                # features except the base column (bit-identical to re-MI'ing
                # x_remaining vs y, since per-feature MI is base-invariant). The
                # exclude-aware aggregate masks out the base entry in place, so
                # no per-base (n, F-1) np.delete copy is materialised for the
                # baseline -- only the held-alive transform-consumer matrices remain.
                mi_y_for_base = _aggregate_mi_per_feature_excluding(
                    _per_feat_y_full,
                    _mi_aggregation,
                    _drop_idx,
                )
            elif _per_feat_y_full is not None:
                mi_y_for_base = _aggregate_mi_per_feature(
                    _per_feat_y_full,
                    _mi_aggregation,
                )
            else:
                mi_y_for_base = _mi_to_target_prebinned(
                    _x_prebinned,
                    y_screen,
                    **_mi_kwargs,
                )
        elif _per_feat_y_knn_full is not None and _surviving_orig_idx is not None:
            # knn baseline from the precomputed base-invariant per-feature vector: aggregate over the surviving
            # original-column indices. Bit-identical to _mi_to_target(x_remaining_matrix, y_screen, knn) -- the same
            # set of single-column MI(y, x_j) values (each on its own per-pair-finite rows), aggregated in the same
            # mean/sum reduction -- without re-running ~50 Kraskov estimators per base.
            mi_y_for_base = _aggregate_mi_per_feature(
                _per_feat_y_knn_full[_surviving_orig_idx],
                _mi_aggregation,
            )
        else:
            mi_y_for_base = _mi_to_target(
                x_remaining_matrix,
                y_screen,
                n_neighbors=self.config.mi_n_neighbors,
                random_state=self.config.random_state,
                estimator=self.config.mi_estimator,
                **_mi_kwargs,
            )
        _base_contexts[base] = dict(
            base_train=base_train,
            base_screen=base_screen,
            x_remaining_matrix=x_remaining_matrix,
            _x_prebinned=_x_prebinned,
            mi_y_for_base=mi_y_for_base,
            _mi_kwargs=_mi_kwargs,
            # Shrunk-domain ``mi_y_compare`` memo shared by all transforms on this base (they share the ``valid_screen`` mask); lock guards the eval threads.
            _mi_y_compare_memo={},
            _mi_y_compare_memo_lock=threading.Lock(),
        )

    # Dedicated UNARY context (full feature matrix, sentinel base) so unary
    # (``requires_base=False``) transforms are scored ONCE against full X and
    # their mi_gain is invariant to auto-base ranking order. Built in the
    # ``_eval`` sibling to keep this module under the LOC threshold; see
    # ``build_unary_base_context`` for the full rationale.
    # On the lazy path the float plane is None; the unary context (bin
    # estimator) reads ``full_x_matrix`` only for its ``.shape[1]`` width guard
    # and the dead ``x_remaining_matrix`` store, so hand it a zero-row proxy of
    # the full column count -- never read for values on this gate.
    _unary_full_x = _full_x_matrix
    if _use_lazy_prebin:
        _full_width = _full_x_prebinned.shape[1] if _full_x_prebinned is not None else 0
        _unary_full_x = np.empty((0, _full_width), dtype=np.float32)
    _unary_ctx = build_unary_base_context(
        full_x_matrix=_unary_full_x,
        full_x_prebinned=_full_x_prebinned,
        per_feat_y_full=_per_feat_y_full,
        y_screen=y_screen,
        n_train=train_idx.size,
        sample_idx=sample_idx,
        mi_aggregation=_mi_aggregation,
        mi_nbins=int(self.config.mi_nbins),
        mi_n_neighbors=self.config.mi_n_neighbors,
        random_state=self.config.random_state,
        mi_estimator=self.config.mi_estimator,
    )
    if _unary_ctx is not None:
        _base_contexts[_UNARY_BASE_SENTINEL] = _unary_ctx

    # Build flat (base, transform_name, transform) work list. Base-dependent
    # transforms iterate per base normally. Unary (``requires_base=False``)
    # transforms route to the dedicated ``_UNARY_BASE_SENTINEL`` context exactly
    # ONCE (full-X scoring, base-free name) instead of being bound to whichever
    # real base they happened to pair with first. ``_unary_evaluated`` still
    # dedups so each unary appears once. Keeping the build serial outside the
    # parallel dispatch preserves deterministic (base, transform) ordering.
    _unary_context_available = _UNARY_BASE_SENTINEL in _base_contexts
    _work_items: list[tuple[str, str, Any]] = []
    for base in base_candidates:
        if base not in _base_contexts:
            continue
        for transform_name in self.config.transforms:
            try:
                transform = get_transform(transform_name)
            except UnknownTransformError as exc:
                logger.warning(
                    "[CompositeTargetDiscovery] %s; skipping.",
                    exc,
                )
                continue
            if not transform.requires_base:
                if transform_name in _unary_evaluated:
                    continue
                _unary_evaluated.add(transform_name)
                # Score the unary against the FULL-X sentinel context, not the
                # current loop's ``base``. Fall back to the real base only if
                # the sentinel context could not be built (degenerate empty
                # feature matrix) so the unary still gets evaluated.
                _unary_base = _UNARY_BASE_SENTINEL if _unary_context_available else base
                _work_items.append((_unary_base, transform_name, transform))
                continue
            _work_items.append((base, transform_name, transform))

    # Single parallel dispatch over the flat
    # ``_work_items`` list. Joblib preserves input order so
    # ``candidates`` ends up in (base, transform) iteration order
    # identical to the legacy nested-loop serial path. joblib
    # threading backend keeps closure capture cheap (no pickling),
    # which is critical for the large ``x_remaining_matrix`` /
    # ``_x_prebinned`` arrays the body reads. Most of the compute
    # (transform.fit / transform.forward / _mi_to_target_prebinned
    # / bootstrap MI loop) is numpy / numba which releases the GIL,
    # so threading scales close to linearly up to cpu_count.
    # 0 = auto: cap at the number of work items and cpu_count. 1 = serial.
    _n_jobs_raw = getattr(self.config, "discovery_n_jobs", 1)
    _n_jobs_raw = 1 if _n_jobs_raw is None else int(_n_jobs_raw)
    if _n_jobs_raw == 0:
        import os as _os

        _n_jobs_disc = max(1, min(len(_work_items), _os.cpu_count() or 1))
    else:
        _n_jobs_disc = max(1, _n_jobs_raw)
    if _n_jobs_disc > 1 and len(_work_items) > 1:
        from joblib import Parallel as _Parallel, delayed as _delayed

        _results = _Parallel(
            n_jobs=_n_jobs_disc,
            backend="threading",
            prefer="threads",
        )(
            _delayed(eval_one_transform)(
                self,
                _b,
                _tn,
                _t,
                base_contexts=_base_contexts,
                y_train=y_train,
                y_screen=y_screen,
                target_col=target_col,
            )
            for _b, _tn, _t in _work_items
        )
    else:
        _results = [
            eval_one_transform(
                self,
                _b,
                _tn,
                _t,
                base_contexts=_base_contexts,
                y_train=y_train,
                y_screen=y_screen,
                target_col=target_col,
            )
            for _b, _tn, _t in _work_items
        ]
    for _r in _results:
        if _r:
            candidates.extend(_r)
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "transforms_evaluated")

    # Family-wise FDR control across the candidate family before the eps gate (see ``apply_fdr_control_to_candidates``); no-op when the bootstrap is disabled.
    if bool(getattr(self.config, "mi_gain_fdr_control", True)):
        apply_fdr_control_to_candidates(
            candidates,
            alpha=float(getattr(self.config, "mi_gain_fdr_alpha", 0.10)),
        )

    # Filter + sort.
    kept_specs: list[CompositeSpec] = []
    for entry in candidates:
        spec: CompositeSpec | None = entry.get("spec")
        if spec is None:
            continue  # already a reject
        if entry.get("fdr_dropped"):
            continue  # family-wise FDR control already rejected this spec.
        # Gate compares LCB (lower CI bound), not point estimate,
        # when bootstrap is enabled. Falls back to point estimate
        # when LCB unavailable.
        mi_gain_for_gate = entry.get("mi_gain_lcb", spec.mi_gain)
        if mi_gain_for_gate <= self.config.eps_mi_gain:
            entry["reason"] = f"mi_gain={spec.mi_gain:.4f} <= eps={self.config.eps_mi_gain:.4f}"
            continue
        kept_specs.append(spec)
        entry["kept"] = True

    # Plugin MI quantises to a fixed grid, so tied mi_gain is realistic; the spec-name secondary key makes top-K deterministic across runs.
    # Known minor inconsistency: the gate above admits on mi_gain_lcb but this ranks on the point mi_gain, so under bootstrap a high-variance
    # big-point/low-LCB spec can outrank a stable better-LCB one (the lcb is not carried on the spec). Default bootstrap_n=0, so ranking is exact.
    # WINNER'S CURSE (SA27): mi_gain is the SELECTION score (max over many candidates) -- optimistically
    # biased, NOT a calibrated generalisation gain. The de-bias is the post-selection holdout re-score below
    # (``apply_honest_holdout``); use mi_gain only as the ranking key here, read ``honest_holdout_gain`` for
    # a generalisation estimate.
    kept_specs.sort(key=lambda s: (-s.mi_gain, getattr(s, "name", "")))
    kept_specs = kept_specs[: self.config.top_k_after_mi]

    # Rolling-origin alpha-drift Chow test for linear_residual specs (lifted to
    # ``_eval_stats`` to keep this file under the monolith threshold).
    kept_specs = apply_alpha_drift_gate(
        self,
        kept_specs,
        df=df,
        train_idx=train_idx,
        y_full=y_full,
        extract_column_array=_extract_column_array,
        linear_residual_fit=_linear_residual_fit,
    )

    # Collapse redundant linear_residual -> diff when alpha ~ 1 and beta ~ 0 (linear_residual
    # has zero information advantage over diff but carries 2 fitted params); lifted to
    # ``_eval_stats`` to keep this file under the monolith threshold.
    kept_specs = apply_linear_residual_diff_collapse(
        self, kept_specs, df=df, train_idx=train_idx, y_train=y_train,
        extract_column_array=_extract_column_array,
    )

    # Phase B: tiny-model rerank. Re-rank the MI-survivors by
    # CV-RMSE on the y-scale (the actual prediction objective).
    # Skip when ``screening == "mi"`` -- callers who want only
    # MI ranking pay zero rerank cost.
    if kept_specs and self.config.screening in ("tiny_model", "hybrid") and self.config.tiny_screening_models in ("single_lgbm", "per_family"):
        kept_specs = self._tiny_model_rerank(
            kept_specs=kept_specs,
            df=df,
            target_col=target_col,
            usable_features=usable_features,
            train_idx=train_idx,
            y_full=y_full,
        )
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "tiny_model_rerank_done")

    if not kept_specs:
        mode = self.config.fail_on_no_gain
        msg = f"[CompositeTargetDiscovery] no candidate cleared mi_gain > " f"{self.config.eps_mi_gain} on target='{target_col}'."
        if mode == "raise":
            raise RuntimeError(msg)
        logger.warning(msg + f" (fail_on_no_gain={mode!r})")

    # Multi-base forward-stepwise auto-promotion of linear_residual specs. After single-base discovery + raw-y baseline gate + tiny-model rerank, look at each kept ``linear_residual`` spec and try greedily adding more bases from the auto-base candidate pool. When the marginal RMSE reduction clears ``multi_base_min_marginal_rmse_gain`` (default 0.02 = 2%), upgrade the spec to ``linear_residual_multi`` with the expanded base list. Measure-first benchmark in ``benchmarks/composite_multi_base_benchmark.py`` validates: geo-mean gain 83% on positive scenarios, no-harm on negative scenarios -> auto-promote=True. Gated by ``self.config.multi_base_enabled``; opt-out via config.
    # Auto-skip multi-base promotion when the base pool is uniformly highly-correlated: stacking two
    # near-identical bases into a multi-base residual adds NO orthogonal signal but DOUBLES the
    # base-shift amplification of the inverse on unseen groups. (On the prod TVT pool every base was
    # >=0.999 correlated -- a multi-base upgrade there would have made the collapse strictly worse.)
    _multibase_pool_corr_skip = False
    if (kept_specs and getattr(self.config, "multi_base_enabled", False)
            and getattr(self, "_auto_base_pool", None)):
        _pool_corr_thresh = float(getattr(self.config, "multi_base_skip_when_pool_corr_above", 0.98))
        if _pool_corr_thresh < 1.0:
            try:
                _pa = [np.asarray(v, dtype=np.float64).ravel()
                       for v in self._auto_base_pool.values() if v is not None]
                _pa = [a for a in _pa if a.size > 2 and float(a.std()) > 0]
                if len(_pa) >= 2:
                    _M = np.vstack(_pa)
                    _C = np.corrcoef(_M)
                    _off = _C[~np.eye(_C.shape[0], dtype=bool)]
                    _off = np.abs(_off[np.isfinite(_off)])
                    if _off.size and float(_off.mean()) > _pool_corr_thresh:
                        _multibase_pool_corr_skip = True
                        logger.info(
                            "[CompositeTargetDiscovery] multi-base promotion SKIPPED: base pool is "
                            "uniformly highly-correlated (mean |pair-corr|=%.4f > %.4g) -- a multi-base "
                            "residual would add no orthogonal signal and double the inverse's base-shift "
                            "amplification.", float(_off.mean()), _pool_corr_thresh,
                        )
            except Exception:  # noqa: BLE001 -- the corr guard is a heuristic; never abort discovery on it
                _multibase_pool_corr_skip = False
    if (kept_specs and getattr(self.config, "multi_base_enabled", False)
            and getattr(self, "_auto_base_pool", None) and not _multibase_pool_corr_skip):
        _multi_max_k = int(getattr(self.config, "multi_base_max_k", 3))
        _multi_min_gain = float(getattr(self.config, "multi_base_min_marginal_rmse_gain", 0.02))
        _cv_sel_mode = str(getattr(self.config, "cv_selector_mode", "mean"))
        _cv_sel_alpha = float(getattr(self.config, "cv_selector_alpha", 1.0))
        _cv_sel_conf = float(getattr(self.config, "cv_selector_confidence", 0.9))
        _cv_sel_qlevel = float(getattr(self.config, "cv_selector_quantile_level", 0.9))
        _cv_persist = bool(getattr(self.config, "cv_persist_fold_scores", False))
        _upgraded_specs: list[CompositeSpec] = []
        # Hoist the (base_column, pool_signature) -> pool_arrays
        # build outside the per-spec loop so K linear_residual specs that
        # share the same auto_base_pool + base_column do ONE pool build
        # (and one _extract_column_array call), not K. Cache key includes
        # the pool signature (frozenset of pool keys) so config-driven
        # pool changes invalidate cleanly.
        _pool_arrays_cache: dict[tuple[str, frozenset], dict[str, np.ndarray]] = {}
        _base_pool_keys_frozen = frozenset(self._auto_base_pool.keys())
        _y_train_local = y_train
        for _spec in kept_specs:
            if _spec.transform_name != "linear_residual":
                _upgraded_specs.append(_spec)
                continue
            _cache_key = (_spec.base_column, _base_pool_keys_frozen)
            _pool_arrays = _pool_arrays_cache.get(_cache_key)
            if _pool_arrays is None:
                # Build candidate pool: the auto-base candidates (top-K MI-ranked bases) PLUS the spec's own seed base.
                _pool_cols = list(self._auto_base_pool.keys())
                if _spec.base_column not in _pool_cols:
                    _pool_cols.append(_spec.base_column)
                # Materialise arrays once (the pool stores arrays).
                _pool_arrays = {c: self._auto_base_pool.get(c) for c in _pool_cols if self._auto_base_pool.get(c) is not None}
                if _spec.base_column not in _pool_arrays:
                    _pool_arrays[_spec.base_column] = _extract_column_array(df, _spec.base_column)[train_idx]
                _pool_arrays_cache[_cache_key] = _pool_arrays
            try:
                _kept_bases, _fwd_diag = forward_stepwise_multi_base(
                    _y_train_local,
                    _pool_arrays,
                    seed_bases=[_spec.base_column],
                    max_k=_multi_max_k,
                    min_marginal_rmse_gain=_multi_min_gain,
                    cv_selector_mode=_cv_sel_mode,
                    cv_selector_alpha=_cv_sel_alpha,
                    cv_selector_confidence=_cv_sel_conf,
                    cv_selector_quantile_level=_cv_sel_qlevel,
                    cv_persist_fold_scores=_cv_persist,
                )
            except Exception as _multi_err:
                logger.warning(
                    "[CompositeTargetDiscovery] multi-base forward-stepwise failed on spec=%s: %s. Keeping single-base spec.",
                    _spec.name,
                    _multi_err,
                )
                _upgraded_specs.append(_spec)
                continue
            if len(_kept_bases) <= 1:
                # No additional bases survived the gate; keep the original single-base spec.
                _upgraded_specs.append(_spec)
                continue
            # Upgrade: fit the linear_residual_multi joint OLS on the kept base set and stamp a NEW spec with extra_base_columns populated.
            _base_matrix = np.column_stack([_pool_arrays[n] for n in _kept_bases])
            _multi_params = _linear_residual_multi_fit(_y_train_local, _base_matrix)
            _new_name = compose_target_name(target_col, "linear_residual_multi", "+".join(_kept_bases))
            _upgraded_spec = CompositeSpec(
                name=_new_name,
                target_col=target_col,
                transform_name="linear_residual_multi",
                base_column=_kept_bases[0],
                fitted_params=_multi_params,
                mi_gain=_spec.mi_gain,
                mi_y=_spec.mi_y,
                mi_t=_spec.mi_t,
                valid_domain_frac=_spec.valid_domain_frac,
                n_train_rows=_spec.n_train_rows,
                extra_base_columns=tuple(_kept_bases[1:]),
            )
            _upgraded_specs.append(_upgraded_spec)
            # The upgraded spec carries a NEW name (``...-linear_residual_multi-<bases>``); carry the
            # seed's tiny-rerank CV-RMSE over to it so the raw-y baseline gate and the public
            # ``tiny_rerank_scores_`` diagnostic can look it up by the new name. The multi-base composite
            # is built by ADDING bases to a seed that already cleared the raw-y baseline gate, and every
            # added base only reduced the joint-OLS residual, so the seed's score is a valid conservative
            # stand-in. Without this the new name has no score entry and any ``tiny_rerank_scores_[name]``
            # lookup KeyErrors (regressed the sklearn-matrix composite sweep).
            _seed_score = getattr(self, "_tiny_rerank_scores", None)
            if isinstance(_seed_score, dict) and _spec.name in _seed_score:
                _seed_score[_new_name] = _seed_score[_spec.name]
            _accepted_steps = [d for d in _fwd_diag if d.get("accepted")]
            logger.info(
                "[CompositeTargetDiscovery.multi_base] upgraded spec='%s' -> '%s' with %d base(s); accepted_steps=%s",
                _spec.name,
                _new_name,
                len(_kept_bases),
                [(d["candidate_added"], f"{d['marginal_gain'] * 100:.1f}%") for d in _accepted_steps],
            )
        # Two seeds can converge on the same multi-base set yet emit name 'X+Y' vs 'Y+X' for one identical joint-OLS transform; dedup on the unordered base set so we don't train + ensemble two perfectly-correlated members.
        _seen_base_sets: set[tuple[str, frozenset]] = set()
        _deduped_specs: list[CompositeSpec] = []
        for _s in _upgraded_specs:
            _set_key = (
                _s.transform_name,
                frozenset((_s.base_column,) + tuple(getattr(_s, "extra_base_columns", ()) or ())),
            )
            if _set_key in _seen_base_sets:
                continue
            _seen_base_sets.add(_set_key)
            _deduped_specs.append(_s)
        kept_specs = _deduped_specs
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "forward_stepwise_done")

    # Opt-in discovery steps (region-adaptive / interaction-base / auto-chain). Gated by config flags defaulting True (each has test-confirmed value); set all False for a no-op leaving kept_specs byte-identical to the pre-hook flow. Heavy logic lives in the ``_opt_in_steps`` sibling (LOC threshold); it returns extra appendable specs (auto-chain) + stashes per-step artefacts on the instance. The cheap gate check + no-op artefact init both live in the sibling.
    if kept_specs and (
        getattr(self.config, "region_adaptive_enabled", False)
        or getattr(self.config, "interaction_base_discovery_enabled", False)
        or getattr(self.config, "auto_chain_discovery_enabled", False)
    ):
        from ._opt_in_steps import run_optional_discovery_steps

        _extra = run_optional_discovery_steps(self, df, target_col, usable_features, train_idx, kept_specs, self.config)
        kept_specs = list(kept_specs) + list(_extra) if _extra else kept_specs
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "opt_in_steps_done")

    # y-scale group-aware holdout gate. Drop specs whose predict-T -> invert-to-y pipeline collapses
    # on a group-disjoint holdout (the prod failure the forward-only MI / i.i.d. honest-holdout never
    # sees). No-op without group ids. Runs BEFORE the honest re-score so the (heavier) MI re-score only
    # touches survivors.
    if kept_specs and getattr(self.config, "yscale_holdout_gate_enabled", True):
        from ._yscale_holdout_gate import apply_yscale_holdout_gate

        kept_specs = apply_yscale_holdout_gate(
            self, df, target_col, kept_specs, usable_features, train_idx, y_full,
            val_idx=val_idx,
        )
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "yscale_holdout_gate_done")

    # Honest holdout re-score (SA27). The winner set is now FINAL; re-score ONLY these
    # survivors on the holdout the discovery never touched (see ``apply_honest_holdout``).
    if kept_specs and _honest_holdout_idx is not None and _honest_holdout_idx.size:
        from ._honest_holdout import apply_honest_holdout

        apply_honest_holdout(
            self, df, target_col, kept_specs, usable_features,
            train_idx, _honest_holdout_idx, y_full,
        )
        if _ram_profiler_on:
            _phase_ram_report(_ram_state, "honest_holdout_rescore_done")

    elapsed = timer() - t0
    logger.info(
        "[CompositeTargetDiscovery] target='%s' discovered %d spec(s) " "from %d candidate(s) in %.2fs",
        target_col,
        len(kept_specs),
        len(candidates),
        elapsed,
    )
    if _ram_profiler_on:
        _phase_ram_report(_ram_state, "fit_exit")

    # Alpha-drift WARNINGs only for the SURVIVING specs. Inline emits during scoring are at DEBUG; the user sees a single, actionable warning at the end of discovery rather than a wall of warnings for specs that the raw-y baseline gate / Wilcoxon filter dropped anyway.
    _drift_flags = getattr(self, "_alpha_drift_flags", {})
    if _drift_flags and kept_specs:
        _drift_threshold = float(
            getattr(
                self.config,
                "alpha_drift_z_threshold",
                3.0,
            )
        )
        _surviving_drift = [
            (s.name, _drift_flags[s.name]) for s in kept_specs if s.name in _drift_flags and _drift_flags[s.name].get("z_score", 0.0) > _drift_threshold
        ]
        if _surviving_drift:
            for _spec_name, _info in _surviving_drift:
                logger.warning(
                    "[CompositeTargetDiscovery] alpha drift "
                    "detected for KEPT spec=%s (alpha first-half="
                    "%.4f, second-half=%.4f, z=%.2f > %.2f). "
                    "Concept drift -- linear_residual may "
                    "underperform on test. Set "
                    "reject_on_alpha_drift=True in "
                    "CompositeTargetDiscoveryConfig to drop "
                    "automatically.",
                    _spec_name,
                    _info["alpha_first_half"],
                    _info["alpha_second_half"],
                    _info["z_score"],
                    _drift_threshold,
                )

    # Reconcile the report ``kept`` flag against the FINAL surviving specs.
    # ``entry['kept']`` is stamped True at the eps_mi_gain gate, but the specs
    # then pass through top_k_after_mi trim, the alpha-drift gate, the
    # linear_residual->diff collapse, the tiny-model rerank, and multi-base
    # name-swaps -- none of which write back to the candidate entries. Without
    # this pass ``report()`` claims kept=True for specs that were actually
    # dropped (or renamed) downstream, contradicting its "all evaluated
    # candidates with their final disposition" contract. Reconcile by spec name
    # against ``kept_specs``; a multi-base upgrade swaps a seed
    # ``linear_residual`` for a ``linear_residual_multi`` of a NEW name, so its
    # seed entry is recorded as upgraded (not silently dropped).
    _final_kept_names = {getattr(s, "name", None) for s in kept_specs}
    _multi_seed_primaries = {s.base_column for s in kept_specs if s.transform_name == "linear_residual_multi"}
    for _entry in candidates:
        _espec = _entry.get("spec")
        if _espec is None:
            continue  # already a reject row; reason already set.
        _ename = getattr(_espec, "name", None)
        if _ename in _final_kept_names:
            _entry["kept"] = True
            continue
        # Spec did NOT survive to the final set. Flip kept and record why,
        # unless the eps gate already rejected it (kept was never set True).
        if _entry.get("kept"):
            if _espec.transform_name == "linear_residual" and _espec.base_column in _multi_seed_primaries:
                _entry["reason"] = "upgraded into a linear_residual_multi spec " "(multi-base forward-stepwise)"
            else:
                _entry["reason"] = (
                    "dropped after the MI gate by a downstream filter "
                    "(top_k_after_mi trim / alpha-drift / "
                    "linear_residual->diff collapse / tiny-model rerank / "
                    "multi-base dedup)"
                )
            _entry["kept"] = False

    # Stash the data signature the specs were fit on so a later ``discover_incremental(prior_result, new_df, ...)`` warm-start compares it against the appended frame without recomputing. Failures non-fatal -- the incremental path recomputes new_sig regardless; empty prior_sig just skips the byte-identical fast path.
    try:
        from ..cache import data_signature as _data_signature

        self._fit_data_signature = _data_signature(df, target_col, feature_cols)
    except Exception:  # noqa: BLE001 -- signature is an optimisation, never load-bearing
        self._fit_data_signature = ""

    # Bookkeeping. (target_col + df_ref + train_idx already stashed.)
    self.specs_ = kept_specs
    self.report_ = [self._entry_to_report(e) for e in candidates]
    self.val_idx_ = val_idx
    self.test_idx_ = test_idx
    self.elapsed_seconds_ = elapsed
    return self
