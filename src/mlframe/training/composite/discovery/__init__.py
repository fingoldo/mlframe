"""CompositeTargetDiscovery: main entry-point class that auto-finds the best (base, transform) pairs for a regression target. Orchestrates: base candidate ranking via residualised-MI, transform screening over the registry, optional tiny-model rerank, multi-base forward-stepwise auto-promotion, validation gating, and CompositeProvenance generation. composite.py re-exports CompositeTargetDiscovery for full back-compat."""

from __future__ import annotations

import hashlib
import logging
import math
import re
import warnings
from datetime import datetime
from timeit import default_timer as timer
from typing import (
    Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd

# Module-level scipy import so external introspection (e.g. the
# test_m3_spearman_demoter_uses_rankdata regression sensor) can confirm the
# rankdata path is wired in - argsort-of-argsort fallback gives wrong ranks on
# ties, so a revert MUST be caught at the sensor level. Graceful fallback
# preserved for installs without scipy.
try:
    from scipy.stats import rankdata
except ImportError:  # pragma: no cover - scipy is a hard dep in pyproject; allow graceful fallback
    rankdata = None

from ..spec import CompositeSpec
from .auto_detect import (
    detect_time_column_candidates,
    sort_df_by_time_column,
    detect_group_column_candidates,
)
from .bayesian import bayesian_alpha_fit
from ..cache import (
    DiscoveryCache,
    data_signature,
    make_discovery_cache_key,
)
from ..ensemble import (
    CompositeCrossTargetEnsemble,
    _is_monotone_nondecreasing,
    compute_oof_holdout_predictions,
    derive_seeds,
    detect_gpu_in_use,
    env_signature,
)
from ..estimator import CompositeTargetEstimator
from ..ensemble.feature_stacking import (
    composite_oof_predictions,
    composite_predictions_as_feature,
)
from .forward_stepwise import forward_stepwise_multi_base
from ..transforms.interaction_bases import generate_interaction_bases
from ..provenance import (
    CompositeProvenance,
    report_to_markdown,
)
from .screening import (
    _build_tiny_model,
    _extract_column_array,
    _is_numeric_column,
    _mi_pair_bin,
    _mi_per_feature_knn,
    _mi_per_feature_y_fixed,
    _mi_to_target,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
    _prebin_feature_columns_cached,
    _per_bin_rmse,
    _residualise,
    _safe_abs_corr_all,
    _safe_corr,
    _sample_indices,
    _silence_tiny_model_output,
    _tiny_cv_rmse_raw_y,
    _tiny_cv_rmse_raw_y_multiseed,
    _tiny_cv_rmse_y_scale,
    _tiny_cv_rmse_y_scale_multiseed,
)
from ..ensemble.stacking import (
    max_off_diagonal_correlation,
    residual_correlation_matrix,
    stacking_aware_gate,
)
from ..streaming import streaming_alpha_check_and_refit
from ..transforms import (
    DomainViolationError,
    Transform,
    UnknownTransformError,
    _linear_residual_fit,
    _linear_residual_multi_fit,
    _TRANSFORMS_REGISTRY,
    compose_target_name,
    get_transform,
    list_transforms,
)

logger = logging.getLogger(__name__)


class CompositeTargetDiscovery:
    """Auto-find the best (base, transform) pairs for a regression target.

    Workflow
    --------
    1. Resolve base candidates (auto via residualised-MI ranking, OR
       user-supplied list). Apply the forbidden-pattern + corr + ptp
       filters to drop columns that are leakage-prone (target encoding,
       derived-from-y, near-constant).
    2. For each (base, transform) pair, fit transform-specific params
       on **train_idx only**, compute T on the train sample, and score
       MI(T, X \\ {base}) against MI(y, X).
    3. Filter by ``min_valid_domain_frac`` and ``eps_mi_gain``, sort
       by MI gain descending, keep top ``top_k_after_mi``.

    Leakage discipline (CRITICAL)
    -----------------------------
    Every fitted parameter (alpha/beta for linear_residual, MAD for
    logratio, eps for ratio, MI bin edges for screening, y-clip
    quantiles, etc.) is computed from rows in ``train_idx`` ONLY. Test
    and validation rows are NEVER touched at fit. The unit test
    ``test_alpha_train_only_changes_with_train_idx`` proves this:
    fitting on two different ``train_idx`` slices of the same df
    yields different alpha, while fitting on the same train_idx
    yields identical alpha. If you ever change the implementation to
    add a "use full df for X" shortcut, that test will fail.
    """

    # ``fit`` (bound externally, see below) sets these fitted attributes; declared here so mypy
    # can type-check reads that run before/without a preceding assignment visible in this class
    # body, and so the sibling modules implementing ``fit`` can assign them without an inline
    # annotation (mypy forbids ``x.attr: T = v`` when ``x`` isn't literally a class-body ``self``).
    specs_: list
    specs_by_group_: dict[Any, list[CompositeSpec]]  # opt-in per-group results; see ``_per_group.py``
    report_: list[dict[str, Any]]
    _auto_base_pool: dict[str, np.ndarray]
    _tiny_rerank_scores: dict[str, float]
    _auto_chains_diag: list
    _alpha_drift_flags: dict[str, dict[str, float]]
    train_idx_: np.ndarray
    val_idx_: np.ndarray | None
    test_idx_: np.ndarray | None
    elapsed_seconds_: float
    _df_ref: Any
    _screen_time_ordered_: bool
    _fit_data_signature: str
    # Sweep-shared honest holdout set by ``fit_with_stability_check`` (consumed by
    # ``carve_screening_holdout``); ``None`` outside a stability sweep.
    _stability_shared_holdout_idx: np.ndarray | None
    stability_counts_: dict[str, int]

    def __init__(self, config: Any) -> None:
        if isinstance(config, dict):
            from ...configs import CompositeTargetDiscoveryConfig
            config = CompositeTargetDiscoveryConfig(**config)
        self.config = config
        self._patterns_compiled: list[re.Pattern] = [re.compile(p) for p in config.forbidden_base_patterns]
        self.specs_by_group_ = {}

    def __getstate__(self) -> dict[str, Any]:
        # fit() pins the (potentially 100+ GB) source frame on ``_df_ref`` and a
        # per-base column pool on ``_auto_base_pool`` for the post-fit
        # iter_transform path. Pickling/deep-copy would otherwise serialise the
        # whole frame and pin it against GC -- exclude both from the pickled
        # state (re-pass ``df`` to iter_transform after unpickling).
        state = self.__dict__.copy()
        state.pop("_df_ref", None)
        state.pop("_auto_base_pool", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._df_ref = None
        self._auto_base_pool = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ``fit`` and the private helpers below are implemented in sibling modules and bound onto this
    # class at the bottom of this module (see the assignments after the class body); declared here
    # as Callable attributes so mypy resolves calls to them without duplicating their signatures.
    fit: Callable[..., "CompositeTargetDiscovery"]
    fit_stacked: Callable[..., Any]
    fit_stacked_on_residual: Callable[..., Any]
    _tiny_model_rerank: Callable[..., Any]
    _auto_base: Callable[..., list[str]]
    _filter_features: Callable[..., Any]
    fit_with_stability_check: Callable[..., "CompositeTargetDiscovery"]
    _target_col: str

    def iter_transform(self, df: Any) -> Iterator[tuple[str, np.ndarray]]:
        """Yield ``(spec_name, T_values)`` per discovered spec, applied
        to ALL rows of ``df``. Streaming generator: we never
        materialise more than one T column at a time -- on a 4M-row
        frame with K=8 specs that saves ~250 MB peak.

        Rows that fail ``domain_check`` get ``NaN`` in T, so downstream
        target-aware filters drop them automatically when fitting the
        per-spec model. The wrapper at predict time uses its own
        ``y_train_median`` fallback for those rows.
        """
        if not getattr(self, "specs_", None):
            return
        target_col = self._target_col
        y_full = _extract_column_array(df, target_col)
        for spec in self.specs_:
            # Multi-base specs (linear_residual_multi, auto-promoted by default
            # via multi_base_enabled) carry extra_base_columns; build the full
            # (n, K) base matrix so domain_check/forward see all K columns. A
            # 1-D pull of base_column alone would raise "base has 1 columns but
            # fitted alphas has K entries" -- the other consumers (ensemble)
            # already stack; this public generator was the one that did not.
            transform = get_transform(spec.transform_name)
            extra = tuple(getattr(spec, "extra_base_columns", ()) or ())
            if not transform.requires_base:
                # Unary specs carry an empty ``base_column`` sentinel and ignore
                # base entirely. ``_extract_column_array(df, "")`` would crash;
                # the adapter's domain_check/forward never read base, so pass None
                # (domain_check gates on y, forward ignores base).
                base_full = None
            elif extra:
                base_full = np.column_stack([_extract_column_array(df, c) for c in (spec.base_column, *extra)])
            else:
                base_full = _extract_column_array(df, spec.base_column)
            valid = transform.domain_check(y_full, base_full)  # type: ignore[arg-type]  # base_full is None only when requires_base=False, whose domain_check tolerates it
            t = np.full(y_full.shape[0], np.nan, dtype=np.float64)
            if valid.any():
                # Unary specs have ``base_full is None`` (the transform ignores
                # base); pass None straight through rather than slicing.
                _base_valid = None if base_full is None else base_full[valid]
                t[valid] = transform.forward(
                    y_full[valid], _base_valid, spec.fitted_params,
                )
            yield spec.name, t

    # Per-cluster composite (REOPENED, was a REJECTED design decision -- see ``discovery/_per_group.py``
    # module docstring for the full rationale): opt-in via ``config.per_group_discovery_enabled``.

    # fit_stacked / fit_stacked_on_residual are bound onto this class from
    # sibling _composite_discovery_stacked at module bottom.

    # fit_with_stability_check is implemented in ``_stability_check.py`` and
    # bound onto this class at the bottom of this module (carved out to keep
    # this facade under its 750-LOC budget -- see test_composite_discovery_facade.py).

    def export_specs(self) -> list[dict[str, Any]]:
        """Plain-dict snapshot of discovered specs for ``metadata`` storage."""
        return [
            {
                "name": s.name,
                "target_col": s.target_col,
                "transform_name": s.transform_name,
                "base_column": s.base_column,
                "fitted_params": dict(s.fitted_params),
                "mi_gain": s.mi_gain,
                "mi_y": s.mi_y,
                "mi_t": s.mi_t,
                "valid_domain_frac": s.valid_domain_frac,
                "n_train_rows": s.n_train_rows,
                # Multi-base specs (linear_residual_multi from forward-stepwise
                # auto-promotion): omitting this field stranded every
                # downstream consumer with only the primary base column,
                # causing transform.forward/inverse to raise "base has 1
                # columns but fitted alphas has K entries" in
                # _phase_dummy_baselines, _phase_composite_post (OOF holdout),
                # and any post-train wrapping path that re-applies the
                # transform.
                "extra_base_columns": tuple(getattr(s, "extra_base_columns", ()) or ()),
                # Post-selection-inference honest gain (SA27). ``mi_gain`` above is the
                # in-screen SELECTION score (optimistically biased by the winner's curse);
                # ``honest_holdout_gain`` is the SAME gain re-scored on a holdout the discovery
                # never touched -- the de-biased generalisation estimate. ``None`` when the
                # holdout pass did not run (disabled / too few rows / degenerate re-score),
                # in which case downstream callers fall back to ``mi_gain``.
                "honest_holdout_gain": getattr(s, "honest_holdout_gain", None),
                "honest_holdout_mi_t": getattr(s, "honest_holdout_mi_t", None),
                "honest_holdout_mi_y": getattr(s, "honest_holdout_mi_y", None),
                "honest_holdout_n_rows": getattr(s, "honest_holdout_n_rows", None),
                # OOS predictive-error re-score on the same holdout (the RMSE analogue of the MI gain;
                # ``None`` when the honest RMSE gate did not run). ``rmse_gain`` = raw - spec, so
                # positive = the composite predicts y better out-of-sample than the raw-y tiny baseline.
                "honest_holdout_rmse": getattr(s, "honest_holdout_rmse", None),
                "honest_holdout_raw_rmse": getattr(s, "honest_holdout_raw_rmse", None),
                "honest_holdout_rmse_gain": getattr(s, "honest_holdout_rmse_gain", None),
            }
            for s in getattr(self, "specs_", [])
        ]

    def report(self) -> list[dict[str, Any]]:
        """All evaluated candidates including rejected ones with reasons.

        Inner per-candidate dicts are defensively deep-copied: a shallow
        ``list(...)`` over a list of dicts decouples the outer list but returns
        the inner dicts (incl. ``score`` / ``reason`` / base column metadata) by
        REFERENCE, so a caller doing ``discovery.report()[0]["score"] = 999``
        would mutate the persisted internal record and later ``report()`` calls
        would return the corrupted value.
        """
        return [dict(r) for r in getattr(self, "report_", [])]

    @property
    def rejection_ledger(self) -> list[dict]:
        """Per-spec rejection rows {spec_name, base_column, transform_name, stage, reason, numbers}, one per
        (spec, rejecting-stage) drop, appended by every discovery gate. Answers "why was MY spec rejected?" -- the
        downstream gates (alpha-drift / linres-collapse / tiny-rerank / structural-fragility / y-scale holdout /
        raw-dominance skip) previously recorded their verdicts only in local lists that were logged and discarded.
        """
        return [dict(r) for r in getattr(self, "rejection_ledger_", [])]

    @property
    def tiny_rerank_scores_(self) -> dict[str, float]:
        """Per-spec tiny CV-RMSE on y-scale (after Phase B rerank).

        Empty when ``screening="mi"`` or rerank didn't run. Keyed by
        spec name. Useful for surfacing "why did this composite get
        kept / rejected" diagnostics.
        """
        return dict(getattr(self, "_tiny_rerank_scores", {}))

    @property
    def honest_holdout_gains_(self) -> dict[str, float | None]:
        """Per-spec honest (post-selection) holdout gain, keyed by spec name (SA27).

        The de-biased generalisation gain re-scored on a holdout the discovery never
        touched. ``None`` for a spec whose holdout re-score did not run (the holdout
        was disabled, too few valid holdout rows, or the transform raised). Use this
        -- NOT the in-screen ``mi_gain`` -- for generalisation claims; the in-screen
        gain is the optimistically-biased winner's-curse selection score.
        """
        return {getattr(s, "name", ""): getattr(s, "honest_holdout_gain", None) for s in getattr(self, "specs_", [])}

    @property
    def raw_y_baseline_rmse_(self) -> float:
        """Tiny CV-RMSE of a model trained directly on raw y on the
        same screening sample / folds / family used by Phase B rerank.

        ``nan`` when the raw-y baseline gate didn't run
        (``require_beats_raw_baseline=False``, screening="mi", or
        degenerate sample).
        """
        return float(getattr(self, "_raw_y_baseline_rmse", float("nan")))

    def filter_drops(self) -> list[dict[str, Any]]:
        """Columns that were filtered out before MI ranking, with reason
        and the offending value (corr, ptp, n_finite). Useful for audit
        when discovery seems to "miss" an obvious base candidate -- the
        most common cause is a corr-threshold false positive on a
        legitimate autoregressive lag feature.

        Same shape as ``report()`` above: inner dicts are defensively copied to
        prevent caller mutation from poisoning the persisted internal state.
        """
        return [dict(d) for d in getattr(self, "_filter_drops", [])]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    # _filter_features is bound onto this class from sibling
    # _composite_discovery_filter at module bottom.

    def _resolve_base_candidates(
        self,
        df: Any,
        target_col: str,
        usable_features: Sequence[str],
        y_train: np.ndarray,
        train_idx: np.ndarray,
    ) -> list[str]:
        """Return the base candidates to evaluate.

        For ``base_candidates="auto"``, rank features by *structural*
        MI gain: for each feature ``x``, residualise ``y`` against
        ``x`` (remove the linear contribution) and score
        ``MI(residual, X \\ {x})``. Features whose autoregressive
        contribution genuinely *opens up* the rest of the feature
        space rank higher than features that just correlate with ``y``
        through a global trend.
        """
        config = self.config
        cap = getattr(config, "max_base_candidates", None)
        if isinstance(config.base_candidates, str) and config.base_candidates == "auto":
            # ``auto_base_top_k`` already bounds the auto path via the full ``_auto_base`` machinery
            # (structural boost / null-perm filter / near-copy exclusion / dedup); the cap only needs
            # to trim FURTHER when it is tighter, so it is cheapest to just cap ``auto_base_top_k``
            # for this call rather than re-rank a second time post-hoc.
            if cap is not None and int(cap) > 0 and int(cap) < int(getattr(config, "auto_base_top_k", 3)):
                _saved_cfg = self.config
                try:
                    self.config = config.model_copy(update={"auto_base_top_k": int(cap)})
                    return self._auto_base(df, usable_features, y_train, train_idx)
                finally:
                    self.config = _saved_cfg
            return self._auto_base(df, usable_features, y_train, train_idx)
        # Explicit list. Keep only entries that survived feature filters.
        explicit = list(config.base_candidates)
        kept = [c for c in explicit if c in usable_features]
        if len(kept) != len(explicit):
            dropped = sorted(set(explicit) - set(kept))
            logger.warning(
                "[CompositeTargetDiscovery] explicit base_candidates dropped " "by filters (forbidden/constant/non-numeric/leak-corr): %s",
                dropped,
            )
        # Early pruning of an over-long EXPLICIT base grid (``max_base_candidates``). Each extra base
        # multiplies the whole per-(base, transform) MI screen, and the explicit path had no cap at
        # all. Rank the survivors by a CHEAP, DIRECT per-pair MI(y, x) on the screening sample (the
        # same primitive ``_auto_base`` starts from) -- deliberately NOT the full ``_auto_base``
        # pipeline: that pipeline's null-permutation filter + near-copy-of-y exclusion + demoters can
        # legitimately drop every candidate (a small/correlated explicit pool routinely fails the
        # permutation-null z-test or the |corr(base,y)|>0.9995 near-copy gate), which would make the
        # CAP -- an early-pruning optimisation with no business rejecting bases -- silently produce
        # ZERO base candidates. A first attempt reused ``_auto_base`` directly and hit exactly that:
        # on a 12-base synthetic grid it returned 0 candidates and, because the heavy null-perm /
        # structural-boost machinery still ran on the full uncapped pool before pruning, was 6x
        # SLOWER than no cap at all (measured in ``_benchmarks/bench_base_candidate_cap.py``).
        if cap is not None and int(cap) > 0 and len(kept) > int(cap):
            try:
                ranked = self._rank_bases_by_mi_for_cap(df, kept, y_train, train_idx)[: int(cap)]
            except Exception as _cap_err:  # -- pruning is an optimisation; keep order-truncation fallback
                logger.warning(
                    "[CompositeTargetDiscovery] max_base_candidates ranking failed (%s); truncating the explicit list in given order.",
                    _cap_err,
                )
                ranked = kept[: int(cap)]
            logger.info(
                "[CompositeTargetDiscovery] max_base_candidates=%d pruned the explicit base grid %d -> %d: %s",
                int(cap), len(kept), len(ranked), ranked,
            )
            return ranked
        return kept

    def _rank_bases_by_mi_for_cap(
        self, df: Any, candidates: Sequence[str], y_train: np.ndarray, train_idx: np.ndarray,
    ) -> list[str]:
        """Cheap per-pair MI(y, x) ranking used ONLY to prune an over-long base grid (``max_base_candidates``).

        Deliberately simpler than ``_auto_base``: no permutation-null filter, no near-copy-of-y
        exclusion, no structural boost / demote / dedup -- those are SELECTION-quality gates with
        their own semantics, and running them here would let a pure early-pruning cap reject bases
        the actual (uncapped) discovery pipeline would have kept (or accepted zero from a small
        pool), and pays their cost before any pruning happens. This method exists solely to pick
        the top-``max_base_candidates`` most-informative bases via the same MI primitive
        ``_auto_base`` starts from, cheaply.
        """
        from .screening import _mi_per_feature_y_fixed_per_col

        cfg = self.config
        sample_idx = _sample_indices(
            train_idx.size, cfg.mi_sample_n, cfg.random_state,
            strategy=getattr(cfg, "mi_sample_strategy", "random"), y=y_train,
            n_strata=getattr(cfg, "mi_n_strata", 10),
        )
        train_idx_screen = train_idx[sample_idx]
        y_screen = y_train[sample_idx]
        x_matrix = self._build_feature_matrix(df, list(candidates), train_idx_screen)
        mi = _mi_per_feature_y_fixed_per_col(x_matrix, y_screen, nbins=int(cfg.mi_nbins))
        ranked = sorted(zip(mi.tolist(), candidates), key=lambda t: -t[0])
        return [c for _m, c in ranked]

    # ``_auto_base`` is implemented in ``_composite_discovery_auto_base.py``
    # and bound onto this class at the bottom of this module.

    # ``_tiny_model_rerank`` is implemented in ``_composite_discovery_tiny_rerank.py``
    # and bound onto this class at the bottom of this module.

    def _build_feature_matrix(
        self, df: Any, cols: Sequence[str], idx: np.ndarray,
    ) -> np.ndarray:
        """Materialise a 2-D ndarray of the requested columns at the
        requested rows. Used only for MI screening on the small
        sample slice -- never on the full frame."""
        if not cols:
            return np.zeros((idx.size, 0), dtype=np.float64)
        # Gather only the sampled rows (O(len(idx)) per column) instead of
        # materialising the full column then slicing -- this is the MI-screening
        # sample, a small fraction of a 4M+ row frame over ~500 columns.
        cols_arrays = [_extract_column_array(df, c, rows=idx) for c in cols]
        return np.column_stack(cols_arrays)

    def _reject(
        self, base: str, transform_name: str, mi_y: float, valid_frac: float,
        reason: str,
    ) -> dict[str, Any]:
        """Build a rejected-candidate work-item entry (``spec=None``) carrying the reason for the rejection ledger."""
        return {
            "spec": None,
            "kept": False,
            "rejected": True,
            "base": base,
            "transform_name": transform_name,
            "valid_domain_frac": valid_frac,
            "mi_y": mi_y,
            "reason": reason,
        }

    def _entry_to_report(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Flatten a rejected or surviving work-item entry into the public per-candidate report row."""
        spec = entry.get("spec")
        if spec is None:
            return {
                "name": f"__{entry['transform_name']}__{entry['base']}",
                "kept": False,
                "rejected": True,
                "reason": entry["reason"],
                "base_column": entry["base"],
                "transform_name": entry["transform_name"],
                "mi_gain": float("nan"),
                "valid_domain_frac": entry.get("valid_domain_frac", float("nan")),
            }
        return {
            "name": spec.name,
            "kept": entry.get("kept", False),
            "rejected": False,
            "reason": entry.get("reason", ""),
            "base_column": spec.base_column,
            "transform_name": spec.transform_name,
            "mi_gain": spec.mi_gain,
            "mi_y": spec.mi_y,
            "mi_t": spec.mi_t,
            "valid_domain_frac": spec.valid_domain_frac,
            "n_train_rows": spec.n_train_rows,
            # In-screen ``mi_gain`` above is the optimistic SELECTION score; the honest
            # holdout gain (None for rejected / non-survivor entries that were never re-scored)
            # is the de-biased generalisation estimate -- report it alongside, never instead.
            "honest_holdout_gain": getattr(spec, "honest_holdout_gain", None),
        }


# Bind the carved-out methods onto the class. Each carved sibling exposes the
# method as a module-level function taking ``self`` as the first argument; the
# binding here makes ``CompositeTargetDiscovery._tiny_model_rerank`` resolve
# to that function so ``self._tiny_model_rerank(...)`` call sites keep
# working unchanged.
from ._tiny_rerank import _tiny_model_rerank as _tiny_model_rerank_impl
from ._auto_base import _auto_base as _auto_base_impl
from ._fit import fit as _fit_impl
from ._filter import _filter_features as _filter_features_impl
from ._stacked import (
    fit_stacked as _fit_stacked_impl,
    fit_stacked_on_residual as _fit_stacked_on_residual_impl,
)
from ._per_group_discovery import (
    route_spec_column_by_group as route_spec_column_by_group,
)
from ._stability_check import fit_with_stability_check as _fit_with_stability_check_impl
CompositeTargetDiscovery._tiny_model_rerank = _tiny_model_rerank_impl
CompositeTargetDiscovery._auto_base = _auto_base_impl
CompositeTargetDiscovery.fit = _fit_impl
CompositeTargetDiscovery._filter_features = _filter_features_impl
CompositeTargetDiscovery.fit_stacked = _fit_stacked_impl
CompositeTargetDiscovery.fit_stacked_on_residual = _fit_stacked_on_residual_impl
CompositeTargetDiscovery.fit_with_stability_check = _fit_with_stability_check_impl


# ----------------------------------------------------------------------
# Re-exports from the post-split sub-modules. Existing code that did
# ``from mlframe.training.composite import detect_group_column_candidates`` etc.
# continues to work without change. New code should prefer the direct
# sub-module import (`mlframe.training.composite_auto_detect`) for
# clearer dependency boundaries.
# ----------------------------------------------------------------------
from .auto_detect import (
    _GROUP_DETECT_DEFAULT_MIN_UNIQUE,
    _GROUP_DETECT_DEFAULT_MAX_UNIQUE,
    _GROUP_DETECT_DEFAULT_MIN_SIZE_RATIO,
)
from ..cache import (
    _DISCOVERY_SIGNATURE_SAMPLE_N,
)
from ..transforms.interaction_bases import (
    _INTERACTION_OPS_DEFAULT,
)

# Dependent helper re-exports.
from .forward_stepwise import (
    _MULTI_BASE_DEFAULT_MAX_K,
    _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN,
)
from ..streaming import (
    _STREAMING_DEFAULT_Z_THRESHOLD,
    _STREAMING_DEFAULT_MIN_BUFFER_N,
)
from .bayesian import (
    _BAYESIAN_ALPHA_DEFAULT_N_BOOTSTRAP,
    _BAYESIAN_ALPHA_DEFAULT_CI_LEVEL,
)

# Incremental / warm-start discovery re-export. Thin facade over
# ``_incremental.incremental_discovery_check`` so callers can warm-start a prior
# discovery result on an appended frame without reaching into the private
# sibling. Train-only / no frame copy (the helper reads only a bounded row sample
# via narrow column pulls).
from ._incremental import (
    IncrementalDecision,
    incremental_discovery_check,
)


def discover_incremental(
    prior_result: "CompositeTargetDiscovery",
    new_df: Any,
    target_col: str,
    feature_cols: Sequence[str],
    config: Any | None = None,
    **kwargs: Any,
) -> IncrementalDecision:
    """Warm-start a prior :class:`CompositeTargetDiscovery` on an appended frame.

    Cheaply decide REUSE (prior specs still hold -- skip the full re-screen) vs
    REDISCOVER (DGP drifted -- caller should run ``fit``) by re-scoring each kept
    spec's MI gain on a bounded sample of ``new_df``. Pulls the prior kept specs
    + the ``data_signature`` they were fit on off ``prior_result``; ``config``
    defaults to the prior result's own config.

    Returns an :class:`IncrementalDecision`; ``reuse=True`` carries the prior
    specs unchanged, ``reuse=False`` carries ``specs=None`` (run a full ``fit``).
    Thin wrapper -- all extra kwargs (``sample_n`` / ``min_surviving_fraction`` /
    ``eps_mi_gain``) forward to :func:`incremental_discovery_check`.
    """
    prior_specs = list(getattr(prior_result, "specs_", []) or [])
    prior_sig = getattr(prior_result, "_fit_data_signature", "") or ""
    if config is None:
        config = getattr(prior_result, "config", None)
    return incremental_discovery_check(
        prior_specs, prior_sig, new_df, target_col, feature_cols, config, **kwargs,
    )


# ----------------------------------------------------------------------
