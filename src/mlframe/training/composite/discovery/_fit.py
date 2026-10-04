"""The main ``fit`` method for ``CompositeTargetDiscovery``.

Split out of ``composite_discovery.py`` to keep the parent below the 1k-line
monolith threshold. ``fit`` is bound back onto the
``CompositeTargetDiscovery`` class at the parent's module bottom, so call
sites that invoke ``disc.fit(...)`` continue to work unchanged.
"""

from __future__ import annotations

import os
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from . import CompositeTargetDiscovery

from .screening import (
    _extract_column_array,
    _is_polars_df,
    _mi_per_feature_knn,
    _mi_per_feature_prebinned,
    _prebin_feature_columns_cached,
    _prebin_feature_columns_lazy,
    _sample_indices,
)
from ._fit_ram import _phase_ram_report, _process_mem_mb  # noqa: F401 -- _process_mem_mb re-exported for back-compat
from ._fit_helpers import maybe_boost_mi_strata_for_heavy_tail, no_base_candidates_report_entry, take_screen_matrix
from mlframe.utils.env_flags import env_flag
from types import SimpleNamespace as _SimpleNamespace

# Sentinel base key for the dedicated UNARY (``requires_base=False``)
# evaluation context. Unary transforms ignore the base column entirely, so they
# are scored ONCE against the FULL feature matrix (no base dropped) rather than
# bound to an arbitrary first base. The empty string is also the
# ``CompositeTargetEstimator`` default ``base_column`` for base-less specs, and
# ``compose_target_name(..., base="")`` renders the base-free 2-segment name.
from ._fit_steps import (  # noqa: F401  -- carved helpers
    logger,
    _UNARY_BASE_SENTINEL,
    _apply_honest_holdout_stages,
    _evaluate_work_items,
    _mi_y_baseline,
    _point_mass_skips_logged,
    _reason_from_ledger,
    _fit_step1_strict_no_op,
    _fit_step2_full_column_count,
    _fit_step3_opt_discovery_steps,
    _fit_step4_seed_entry_recorded,
)


def _screen_matrices(self, df, _usable_features_list, train_idx_screen, *, _bin_estimator, _dedup_x_remaining, _ram_state):
    """``(use_lazy_prebin, full_x_matrix, full_x_prebinned, streamed_dedup)`` for the screen sample.

    The lazy path (see the gate comment in ``fit``) returns no float matrix and, with dedup on, the streamed collinearity
    the per-base dedup reads instead; the eager path returns the float matrix and its (cached) prebinned codes.
    """
    _streamed_dedup = None
    _lazy_force = os.environ.get("MLFRAME_DISCOVERY_LAZY_PREBIN", "").strip().lower()
    _lazy_n_floor = int(os.environ.get("MLFRAME_DISCOVERY_LAZY_PREBIN_MIN_N", "50000"))
    _lazy_eligible = _bin_estimator and _is_polars_df(df) and len(_usable_features_list) > 0
    if _lazy_force in ("0", "false", "no", "off"):
        _use_lazy_prebin = False
    elif _lazy_force in ("1", "true", "yes", "on"):
        _use_lazy_prebin = _lazy_eligible
    else:
        _use_lazy_prebin = _lazy_eligible and train_idx_screen.size >= _lazy_n_floor
    _prebin_use_cache = env_flag("MLFRAME_PREBIN_CACHE", default=True)
    if _use_lazy_prebin:
        # Defer column extraction: never materialise the (n, F) float plane.
        _full_x_matrix = None
        _full_x_prebinned: np.ndarray | None = _prebin_feature_columns_lazy(
            df,
            _usable_features_list,
            train_idx_screen,
            nbins=int(self.config.mi_nbins),
        )
        _streamed_dedup = None
        if _dedup_x_remaining:
            from ._lazy_dedup import StreamedCollinearity

            _streamed_dedup = StreamedCollinearity(
                df, _usable_features_list, train_idx_screen,
                corr_threshold=float(getattr(self.config, "dedup_x_remaining_corr_threshold", 0.99)),
            )
        if _ram_state is not None:
            _phase_ram_report(_ram_state, "lazy_prebin_features_done")
    else:
        _full_x_matrix = take_screen_matrix(
            self, df,
            _usable_features_list,
            train_idx_screen,
        )
        if _ram_state is not None:
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
        if _ram_state is not None:
            _phase_ram_report(_ram_state, "prebin_features_done")
    return _use_lazy_prebin, _full_x_matrix, _full_x_prebinned, _streamed_dedup

def fit(
    self: "CompositeTargetDiscovery",
    df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray | None = None,
    test_idx: np.ndarray | None = None,
    time_ordering: Any = None,
    val_df: Any = None,
    val_y: np.ndarray | None = None,
) -> "CompositeTargetDiscovery":
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
    st = _SimpleNamespace()  # long-lived locals of this function (see the stage helpers below)
    if not self.config.enabled:
        self.specs_ = []
        self.report_ = []
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
        """Apply the same boolean-mask-to-positions normalisation used for train_idx above to val_idx/test_idx, raising loudly on non-integer, non-boolean dtypes."""
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
    # Per-fit rejection ledger: every gate appends a structured {spec, stage, reason, numbers} row so
    # "why was MY spec rejected?" is queryable from ``rejection_ledger`` instead of only the logs.
    from ._rejection_ledger import ledger_init
    ledger_init(self)
    # Per-fit too: the drift gate clears these only when a linear_residual survived, so a re-fit of one instance
    # (stability replicates, the stacked second pass, per-group reuse) that keeps none would warn about the previous
    # fit's specs at the end of this one.
    self._alpha_drift_flags = {}

    # Post-selection-inference holdout (winner's-curse de-bias, SA27): carve a never-touched
    # holdout BEFORE screening, then REBIND ``train_idx`` to the screening pool so every
    # downstream consumer is holdout-excluded with no per-site change (carve_screening_holdout).
    from ._honest_holdout import carve_screening_holdout

    # Snapshot the pre-carve, full train_idx for the opt-in per-group discovery path below (it needs
    # every group's rows to split by group and carve ITS OWN per-group honest holdout -- reusing the
    # already-carved outer/global screening pool would either starve small groups or leak the outer
    # holdout rows into a per-group screen).
    st._orig_train_idx = train_idx

    train_idx, st._honest_holdout_idx = carve_screening_holdout(self, train_idx)

    if train_idx.size < 50:
        logger.warning(
            "[CompositeTargetDiscovery] train_idx has only %d rows; " "MI estimates unreliable. Discovery yields no specs.",
            train_idx.size,
        )
        self.specs_ = []
        self.report_ = []
        return self

    st.t0 = timer()
    # Per-fit() RAM telemetry state. Each sub-phase report logs delta vs
    # prev + cumulative vs entry; opt out by setting
    # MLFRAME_DISCOVERY_RAM_PROFILER=0 (the helper checks the env once at
    # entry so the rest of the fit() body never tests the flag again).
    st._ram_state = {}
    st._ram_profiler_on = env_flag("MLFRAME_DISCOVERY_RAM_PROFILER", default=True)
    if st._ram_profiler_on:
        _phase_ram_report(st._ram_state, "entry")

    # Pull target on train rows. We never touch val/test.
    st.y_full = _extract_column_array(df, target_col)
    st.y_train = st.y_full[train_idx]

    # Auto-boost mi_n_strata on heavy-tail y (skew/kurtosis); carved to _fit_helpers to keep this module under 1k LOC.
    maybe_boost_mi_strata_for_heavy_tail(self, st.y_train)

    # Filter feature_cols by name patterns AND constancy on train.
    st.usable_features = self._filter_features(df, feature_cols, st.y_train, train_idx)
    if st._ram_profiler_on:
        _phase_ram_report(st._ram_state, "filter_features_done")

    # knn-MI cost guard, BEFORE base resolution (auto-base ranking + its permutation null is ~21 knn sweeps per column):
    # probe the Kraskov cost on a screen-sized sample and downgrade knn -> bin (warning) when the extrapolated sweep exceeds
    # ``knn_mi_budget_seconds``. Swaps only a per-fit config copy; see ``_knn_budget`` for why T-side caching is not an option.
    if self.config.mi_estimator == "knn" and getattr(self.config, "knn_mi_auto_downgrade", True):
        from ._knn_budget import maybe_downgrade_knn_estimator

        maybe_downgrade_knn_estimator(self, df, st.usable_features, train_idx, st.y_train)

    # Resolve base candidates.
    st.base_candidates = self._resolve_base_candidates(
        df,
        target_col,
        st.usable_features,
        st.y_train,
        train_idx,
    )
    if st._ram_profiler_on:
        _phase_ram_report(st._ram_state, "resolve_base_candidates_done")

    # Pre-discovery base-target leakage guard (config.detect_base_leakage); see _fit_temporal.apply_base_leakage_guard.
    # Seeded unconditionally so a caller can tell "the guard was inert" from "fit never reached the guard".
    self._base_leakage_guard_ran_ = False
    self._leaky_bases_dropped_ = []
    if getattr(self.config, "detect_base_leakage", True) and time_ordering is not None and st.base_candidates:
        from ._fit_temporal import apply_base_leakage_guard

        st.base_candidates = apply_base_leakage_guard(self, df, st.base_candidates, train_idx, st.y_train, time_ordering)

    # Add strictly-causal per-group lag / trailing / expanding bases of y AFTER the leakage guard (causal by construction,
    # must not be stripped as leakage); adds to usable_features + base_candidates. No-op unless a group key is configured.
    from ._grouped_causal_bases import maybe_add_grouped_causal_bases
    df, st.usable_features, st.base_candidates = maybe_add_grouped_causal_bases(
        self, df, target_col, st.usable_features, st.base_candidates, train_idx,
    )
    # Interaction bases (a*b beating both parents on MI) become base candidates here, before screening, so a composite on
    # one is selected and gated like any other base; outside discovery the name resolves from its parents.
    from ._interaction_specs import add_interaction_bases
    df, st.usable_features, st.base_candidates = add_interaction_bases(self, df, st.usable_features, st.base_candidates, train_idx, st.y_train)
    self._df_ref = df  # engineered columns must be visible to downstream gates that read self._df_ref.

    if not st.base_candidates:
        logger.warning(
            "[CompositeTargetDiscovery] no usable base candidates after " "forbidden-pattern / corr / ptp / numeric filters. " "Discovery yields no specs."
        )
        self.specs_ = []
        # Never return a silent, unexplained empty report: one diagnostic entry says WHY discovery found nothing.
        self.report_ = no_base_candidates_report_entry()
        return self

    # Down-sample for MI screening. Stratified-quantile when
    # configured -- guarantees per-bin coverage on heavy-tail y.
    st.sample_idx = _sample_indices(
        train_idx.size,
        self.config.mi_sample_n,
        self.config.random_state,
        strategy=getattr(self.config, "mi_sample_strategy", "stratified_quantile"),
        y=st.y_train,
        n_strata=getattr(self.config, "mi_n_strata", 10),
    )
    st.train_idx_screen = train_idx[st.sample_idx]

    # Time-awareness: when the caller supplies an explicit ``time_ordering``,
    # SORT the screening sample into time order so the downstream tiny-model CV
    # is a genuine forward-walk (TimeSeriesSplit). The old heuristic inferred
    # time-awareness from base MONOTONICITY, which never fires for the canonical
    # non-monotone ``lag(y)`` base -> shuffled K-fold leaked future->past on
    # temporal data. Sorting once here makes every per-spec / raw-baseline /
    # stepwise tiny-CV time-correct without per-call ordering logic.
    from ._fit_temporal import order_screen_by_time

    st.train_idx_screen, st.sample_idx, self._screen_time_ordered_ = order_screen_by_time(st.train_idx_screen, st.sample_idx, time_ordering)
    # Keep the key itself, not just the flag: consumers that draw their OWN sample (the tiny rerank, the drift gate)
    # must re-apply the order, otherwise they run a "forward walk" over whatever order their rows happen to be in.
    self._time_ordering_ = time_ordering
    st.y_screen = st.y_full[st.train_idx_screen]

    # Bin-MI floors every value to 0.0 when the screening sample has fewer than
    # 5*nbins finite rows (joint-histogram cells too sparse), so top-K ranking
    # silently degenerates to the rerank/alphabetical tiebreaker. Warn rather
    # than auto-shrink nbins (which would change the MI numerics).
    if self.config.mi_estimator == "bin":
        _eff_n = int(st.train_idx_screen.size)
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
    self._auto_base_pool = {}

    # Score each (base, transform).
    # Unary y-transforms (``requires_base=False``) ignore the base column, so each routes through ONE dedicated context (``_UNARY_BASE_SENTINEL``) scored against the FULL feature matrix (no base dropped) with an empty-string, base-free spec name -- not bound to / scored against / named after an arbitrary "first" base as before (which made the unary's mi_gain shift with irrelevant auto-base ranking and claim a nonexistent base dependence). The set tracks which unary names are already evaluated so later base-loop iterations skip the redundant re-fit; bivariate + chain transforms still iterate per base.
    st._unary_evaluated = set()

    st.candidates = []

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
    st._usable_features_list = list(st.usable_features)
    st._col_index = {c: i for i, c in enumerate(st._usable_features_list)}
    st._bin_estimator = self.config.mi_estimator == "bin"
    st._dedup_x_remaining = bool(getattr(self.config, "dedup_x_remaining_for_mi_baseline", True))
    # Lazy-prebin gate: on the bin estimator the downstream MI uses only the
    # int16/int32 CODE matrix -- the float32 (n, F) plane feeds nothing but the
    # prebinning itself (and dedup, when on). On a POLARS carrier large enough
    # that the float plane is the dominant transient we can therefore skip
    # building it entirely: pull + bin one column at a time
    # (``_prebin_feature_columns_lazy``) so peak extra RAM is ONE column, not
    # the whole plane. BIT-IDENTICAL codes (shared per-column kernel). Gated to
    # bin + polars + a size floor; ndarray / small / knn inputs keep the eager path. Dedup runs on a row-block Gram
    # (``_lazy_dedup.StreamedCollinearity``) instead of the float plane. Override via
    # MLFRAME_DISCOVERY_LAZY_PREBIN=0|1 (force off / on; ignores the gate).
    st._use_lazy_prebin, st._full_x_matrix, st._full_x_prebinned, st._streamed_dedup = _screen_matrices(
        self, df, st._usable_features_list, st.train_idx_screen, _bin_estimator=st._bin_estimator, _dedup_x_remaining=st._dedup_x_remaining,
        _ram_state=st._ram_state if st._ram_profiler_on else None,
    )

    # Per-feature MI(y, x_j) is INDEPENDENT of which base column is excluded, so
    # compute the full-feature vector ONCE and derive each base's mi_y by
    # excluding that base's column (mean/sum over the survivors) -- instead of
    # re-binning + re-MI'ing the shared columns per base candidate. Bit-identical
    # on BOTH the prebinned (mi_estimator='bin') path (_per_feat_y_full below) and
    # the knn path (_per_feat_y_knn_full below).
    st._mi_aggregation = getattr(self.config, "mi_aggregation", "mean")
    st._per_feat_y_full = (
        _mi_per_feature_prebinned(
            st._full_x_prebinned,
            st.y_screen,
            nbins=int(self.config.mi_nbins),
        )
        if st._full_x_prebinned is not None
        else None
    )
    # knn analogue: per-column MI(y, x_j) is likewise base-invariant, but the Kraskov estimator dominates
    # wall time (~0.45s/column at the 100k screen sample), so re-running the full per-column sweep per base
    # candidate is the dominant redundant cost on the knn path. Compute the vector ONCE over the full float
    # matrix and derive each base's mi_y by aggregating over its surviving (base-dropped, dedup-kept) original
    # column indices -- bit-identical because each column's MI is independent of which others are present.
    st._per_feat_y_knn_full = (
        _mi_per_feature_knn(
            st._full_x_matrix,
            st.y_screen,
            n_neighbors=self.config.mi_n_neighbors,
            random_state=self.config.random_state,
        )
        if (not st._bin_estimator and st._full_x_matrix is not None)
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
    st._dedup_corr_thr = float(getattr(self.config, "dedup_x_remaining_corr_threshold", 0.99))
    st._base_contexts = {}
    _fit_step1_strict_no_op(self, st, df, train_idx)
    _fit_step2_full_column_count(self, st, train_idx, target_col, df)

    # Opt-in discovery steps (region-adaptive / interaction-base / auto-chain). Gated by config flags defaulting True (each has test-confirmed value); set all False for a no-op leaving kept_specs byte-identical to the pre-hook flow. Heavy logic lives in the ``_opt_in_steps`` sibling (LOC threshold); it returns extra appendable specs (auto-chain) + stashes per-step artefacts on the instance. The cheap gate check + no-op artefact init both live in the sibling.
    _fit_step3_opt_discovery_steps(self, st, df, target_col, train_idx, val_df, val_y)
    st._multi_seed_primaries = {s.base_column for s in st.kept_specs if s.transform_name == "linear_residual_multi"}
    _fit_step4_seed_entry_recorded(self, st)

    # The data signature the specs were fit on is read only by ``discover_incremental``, but it cost 84-308 ms at 200k x 50
    # (seconds on wide polars frames) on every fit, stability replicate and per-group fit. Record what it needs and let
    # ``fit_data_signature()`` compute it on first use; pickling computes it before the frame reference is dropped. The row count lets it re-score only appended rows.
    # A signature the caller already computed on this very frame, target and feature list is taken as is.
    st._seed = getattr(self, "_fit_data_signature_seed", None)
    st._seed_ok = st._seed is not None and st._seed[0] == id(df) and st._seed[1] == target_col and st._seed[2] == tuple(feature_cols)
    self._fit_data_signature = st._seed[3] if (st._seed is not None and st._seed_ok) else None
    self._fit_data_signature_seed = None
    self._fit_data_signature_inputs, self._fit_n_rows = (target_col, list(feature_cols)), len(df)

    # Bookkeeping. (target_col + df_ref + train_idx already stashed.)
    self.specs_ = st.kept_specs
    self.report_ = [self._entry_to_report(e) for e in st.candidates]
    self.val_idx_ = val_idx
    self.test_idx_ = test_idx
    self.elapsed_seconds_ = st.elapsed

    # Opt-in per-group/per-cluster discovery (reopened REJECTED decision -- see the module docstring
    # near the "Per-cluster composite" comment in ``discovery/__init__.py``). Runs AFTER the global fit
    # above so ``specs_`` (the default consumer-facing attribute) is populated exactly as before
    # regardless of this flag -- small/unseen groups fall back to it. Entirely behind the flag: when
    # ``per_group_discovery_enabled`` is False (the default) this block never executes, so the default
    # code path above is untouched byte-for-byte.
    if getattr(self.config, "per_group_discovery_enabled", False):
        from ._per_group import run_per_group_discovery

        self.specs_by_group_ = run_per_group_discovery(
            self,
            df,
            target_col,
            feature_cols,
            st._orig_train_idx,
            val_idx,
            test_idx,
            time_ordering,
            val_df,
            val_y,
        )
    return self
