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
    rankdata = None  # type: ignore[assignment]

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

    def __init__(self, config: Any) -> None:
        if isinstance(config, dict):
            from ...configs import CompositeTargetDiscoveryConfig
            config = CompositeTargetDiscoveryConfig(**config)
        self.config = config
        self._patterns_compiled: list[re.Pattern] = [
            re.compile(p) for p in config.forbidden_base_patterns
        ]

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

    # ``fit`` is implemented in ``_composite_discovery_fit.py`` and bound
    # onto this class at the bottom of this module.

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
                base_full = np.column_stack(
                    [_extract_column_array(df, c)
                     for c in (spec.base_column, *extra)]
                )
            else:
                base_full = _extract_column_array(df, spec.base_column)
            valid = transform.domain_check(y_full, base_full)
            t = np.full(y_full.shape[0], np.nan, dtype=np.float64)
            if valid.any():
                # Unary specs have ``base_full is None`` (the transform ignores
                # base); pass None straight through rather than slicing.
                _base_valid = None if base_full is None else base_full[valid]
                t[valid] = transform.forward(
                    y_full[valid], _base_valid, spec.fitted_params,
                )
            yield spec.name, t

    # Per-cluster composite (REJECTED design decision):
    # Original proposal: when dataset has 50+ entities (group_id / customer_id /
    # segment) with >= 200 rows each, discovery COULD run per-cluster + global
    # fallback via ``linear_residual_grouped``. User judged this premature:
    # "10-15 values per cluster too few for stable per-cluster discovery".
    # Revisit ONLY when production data shows 500+ rows per cluster on average.
    # No action required until that data shape appears -- this is a closed
    # design decision, not an open TODO.

    # fit_stacked / fit_stacked_on_residual are bound onto this class from
    # sibling _composite_discovery_stacked at module bottom.

    def fit_with_stability_check(
        self,
        df: Any,
        target_col: str,
        feature_cols: Sequence[str],
        train_idx: np.ndarray,
        val_idx: np.ndarray | None = None,
        test_idx: np.ndarray | None = None,
        *,
        n_bootstrap_runs: int = 5,
        min_keep_fraction: float = 0.6,
        subsample_fraction: float = 0.5,
    ) -> CompositeTargetDiscovery:
        """Run :meth:`fit` ``n_bootstrap_runs`` times on DECORRELATED reseeds and per-run row subsamples, keeping only specs that survive in at least ``min_keep_fraction * n_bootstrap_runs`` runs.

        Filters "lucky split" wins where a single seed happens to find a spec that does not generalise. Default thresholds (5 runs, 60% majority, 50% subsample) match the standard stability-selection literature (Meinshausen-Buhlmann), whose procedure draws each replicate on a *random half* of the rows -- not merely a reseed of the same sample.

        Returns ``self``. After the call, ``self.specs_`` is the stable subset and ``self.stability_counts_`` maps each name to its survival count.

        Decorrelation rationale
        -----------------------
        Two defects made the pre-fix "bootstrap" runs near-duplicates rather
        than independent replicates, so the gate barely filtered anything:

        1. **Seed-stride collision.** The per-run stride was
           ``base_seed + i*7919``. The inner multi-seed sweep
           (``_screening_tiny._tiny_cv_rmse_*_multiseed``) strides the SAME
           7919 as ``base_random_state + s_idx*7919``. So run ``i``'s reseed
           landed exactly on run ``i-1``'s second inner seed -> the
           "independent" runs shared their CV draws on a 7919-aligned ladder,
           correlating the very replicates the gate assumes are independent.
           Fixed by deriving each run's master seed via the sha256-based
           :func:`derive_seeds` (no arithmetic relationship to the inner
           ``*7919`` ladder), which cannot collide with the multi-seed stride.
        2. **No row subsample.** Every run reused the *identical* ``train_idx``,
           so the only variation was the seed -- a spec found on one sample was
           almost always re-found on the same sample. Meinshausen-Buhlmann
           stability selection draws each replicate on a random subsample of
           the rows; we now draw a ``subsample_fraction`` (default 0.5) slice of
           ``train_idx`` per run with a per-run-seeded RNG. ``val_idx`` /
           ``test_idx`` are passed through untouched (never resampled -- fit
           only ever reads ``train_idx`` rows). Set ``subsample_fraction=1.0``
           to recover the legacy reseed-only behaviour.

        Perf note: the decorrelation additions are a per-call ``derive_seeds``
        (n_runs sha256 hashes) plus one ``np.random.choice(replace=False)+sort``
        per run. Measured ~400 us total for 5 runs at n_train=400 and ~16 ms for
        a single 50% draw at n_train=400k -- negligible vs one ``fit()`` (MI
        screening + tiny-model CV over the whole sample, seconds). No actionable
        speedup; the draw is intrinsically O(n_train) and is the cheapest part of
        each replicate.
        """
        if n_bootstrap_runs <= 1:
            return self.fit(df, target_col, feature_cols, train_idx, val_idx, test_idx)
        from collections import Counter

        # Per-run reseeding (and any mid-fit heavy-tail mi_n_strata boost that swaps self.config for a model_copy) must mutate only a config we own:
        # otherwise the final restore would write back the swapped copy and leave the caller's shared config permanently reseeded, poisoning later targets.
        _saved_cfg = self.config
        self.config = self.config.model_copy()
        base_seed = int(self.config.random_state)
        # Decorrelate run seeds from the inner multi-seed ``*7919`` ladder
        # (defect 1 above): sha256-derive one master seed per run keyed on the
        # base seed + run index, so no run's reseed can land on another run's
        # inner CV seed. Masked to int32 to stay a valid numpy/config seed.
        _run_seeds = derive_seeds(base_seed, [f"stability_run_{i}" for i in range(int(n_bootstrap_runs))])
        train_idx = np.asarray(train_idx)
        _n_train = int(train_idx.size)
        # M-B subsample size: clamp to [2, n_train]. A degenerate (<2 rows)
        # subsample is meaningless for fitting transform params, so fall back to
        # the full train_idx in that pathological case.
        _frac = float(subsample_fraction)
        _sub_n = _n_train if _frac >= 1.0 else max(2, int(round(_frac * _n_train)))
        _sub_n = min(_sub_n, _n_train)
        keep_counter: Counter = Counter()
        spec_by_name: dict[str, CompositeSpec] = {}
        for i in range(int(n_bootstrap_runs)):
            _run_seed = int(_run_seeds[f"stability_run_{i}"]) & 0x7FFFFFFF
            self.config.random_state = _run_seed
            # Per-run row subsample (defect 2 above). A dedicated RNG seeded
            # from the same decorrelated run seed keeps the draw reproducible
            # for a given base seed while making each run a genuinely different
            # row population. Sorted to preserve any time/order semantics the
            # caller's train_idx carried (fit reads rows positionally).
            if _sub_n < _n_train:
                _run_rng = np.random.default_rng(_run_seed)
                _run_train_idx = np.sort(
                    _run_rng.choice(train_idx, size=_sub_n, replace=False)
                )
            else:
                _run_train_idx = train_idx
            try:
                self.fit(df, target_col, feature_cols, _run_train_idx, val_idx, test_idx)
            except Exception as _exc:
                logger.warning(
                    "[CompositeTargetDiscovery.stability] bootstrap run %d failed: %s",
                    i, _exc,
                )
                continue
            for spec in self.specs_:
                keep_counter[spec.name] += 1
                spec_by_name.setdefault(spec.name, spec)
        # Restore the caller's original config and write the stable spec set.
        self.config = _saved_cfg
        threshold = max(1, int(min_keep_fraction * n_bootstrap_runs))
        stable_names = [n for n, c in keep_counter.items() if c >= threshold]
        self.specs_ = [spec_by_name[n] for n in stable_names if n in spec_by_name]
        self.stability_counts_ = dict(keep_counter)
        logger.info(
            "[CompositeTargetDiscovery.stability] n_runs=%d, threshold=%d/%d, "
            "subsample=%d/%d rows (frac=%.2f). Kept %d spec(s); counts: %s",
            n_bootstrap_runs, threshold, n_bootstrap_runs,
            _sub_n, _n_train, _frac,
            len(self.specs_), dict(keep_counter),
        )
        return self

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
        return {
            getattr(s, "name", ""): getattr(s, "honest_holdout_gain", None)
            for s in getattr(self, "specs_", [])
        }

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
        if isinstance(config.base_candidates, str) and config.base_candidates == "auto":
            return self._auto_base(df, usable_features, y_train, train_idx)
        # Explicit list. Keep only entries that survived feature filters.
        explicit = list(config.base_candidates)
        kept = [c for c in explicit if c in usable_features]
        if len(kept) != len(explicit):
            dropped = sorted(set(explicit) - set(kept))
            logger.warning(
                "[CompositeTargetDiscovery] explicit base_candidates dropped "
                "by filters (forbidden/constant/non-numeric/leak-corr): %s", dropped,
            )
        return kept

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
from ._tiny_rerank import _tiny_model_rerank as _tiny_model_rerank_impl  # noqa: E402
from ._auto_base import _auto_base as _auto_base_impl  # noqa: E402
from ._fit import fit as _fit_impl  # noqa: E402
from ._filter import _filter_features as _filter_features_impl  # noqa: E402
from ._stacked import (  # noqa: E402
    fit_stacked as _fit_stacked_impl,
    fit_stacked_on_residual as _fit_stacked_on_residual_impl,
)
CompositeTargetDiscovery._tiny_model_rerank = _tiny_model_rerank_impl
CompositeTargetDiscovery._auto_base = _auto_base_impl
CompositeTargetDiscovery.fit = _fit_impl
CompositeTargetDiscovery._filter_features = _filter_features_impl
CompositeTargetDiscovery.fit_stacked = _fit_stacked_impl
CompositeTargetDiscovery.fit_stacked_on_residual = _fit_stacked_on_residual_impl


# ----------------------------------------------------------------------
# Re-exports from the post-split sub-modules. Existing code that did
# ``from mlframe.training.composite import detect_group_column_candidates`` etc.
# continues to work without change. New code should prefer the direct
# sub-module import (`mlframe.training.composite_auto_detect`) for
# clearer dependency boundaries.
# ----------------------------------------------------------------------
from .auto_detect import (  # noqa: E402,F401
    _GROUP_DETECT_DEFAULT_MIN_UNIQUE,
    _GROUP_DETECT_DEFAULT_MAX_UNIQUE,
    _GROUP_DETECT_DEFAULT_MIN_SIZE_RATIO,
)
from ..cache import (  # noqa: E402,F401
    _DISCOVERY_SIGNATURE_SAMPLE_N,
)
from ..transforms.interaction_bases import (  # noqa: E402,F401
    _INTERACTION_OPS_DEFAULT,
)

# Dependent helper re-exports.
from .forward_stepwise import (  # noqa: E402,F401
    _MULTI_BASE_DEFAULT_MAX_K,
    _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN,
)
from ..streaming import (  # noqa: E402,F401
    _STREAMING_DEFAULT_Z_THRESHOLD,
    _STREAMING_DEFAULT_MIN_BUFFER_N,
)
from .bayesian import (  # noqa: E402,F401
    _BAYESIAN_ALPHA_DEFAULT_N_BOOTSTRAP,
    _BAYESIAN_ALPHA_DEFAULT_CI_LEVEL,
)

# Incremental / warm-start discovery re-export. Thin facade over
# ``_incremental.incremental_discovery_check`` so callers can warm-start a prior
# discovery result on an appended frame without reaching into the private
# sibling. Train-only / no frame copy (the helper reads only a bounded row sample
# via narrow column pulls).
from ._incremental import (  # noqa: E402,F401
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
