"""CompositeTargetDiscovery: main entry-point class that auto-finds the best (base, transform) pairs for a regression target. Orchestrates: base candidate ranking via residualised-MI, transform screening over the registry, optional tiny-model rerank, multi-base forward-stepwise auto-promotion, validation gating, and CompositeProvenance generation. Split out of composite.py to isolate discovery internals from the lightweight wrapper / spec / provenance surface; composite.py re-exports CompositeTargetDiscovery at its bottom for full back-compat."""


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
# M3 fix is wired in - argsort-of-argsort fallback gives wrong ranks on
# ties, so a revert MUST be caught at the sensor level. Graceful fallback
# preserved for installs without scipy.
try:
    from scipy.stats import rankdata
except ImportError:  # pragma: no cover - scipy is a hard dep in pyproject; allow graceful fallback
    rankdata = None  # type: ignore[assignment]

from .composite_spec import CompositeSpec
from .composite_auto_detect import (
    detect_time_column_candidates,
    sort_df_by_time_column,
    detect_group_column_candidates,
)
from .composite_bayesian import bayesian_alpha_fit
from .composite_cache import (
    DiscoveryCache,
    data_signature,
    make_discovery_cache_key,
)
from .composite_ensemble import (
    CompositeCrossTargetEnsemble,
    _is_monotone_nondecreasing,
    compute_oof_holdout_predictions,
    derive_seeds,
    detect_gpu_in_use,
    env_signature,
)
from .composite_estimator import CompositeTargetEstimator
from .composite_feature_stacking import (
    composite_oof_predictions,
    composite_predictions_as_feature,
)
from .composite_forward_stepwise import forward_stepwise_multi_base
from .composite.transforms.interaction_bases import generate_interaction_bases
from .composite_provenance import (
    CompositeProvenance,
    report_to_markdown,
)
from .composite_screening import (
    _build_tiny_model,
    _extract_column_array,
    _is_numeric_column,
    _mi_pair_bin,
    _mi_per_feature_y_fixed,
    _mi_to_target,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
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
from .composite_stacking import (
    max_off_diagonal_correlation,
    residual_correlation_matrix,
    stacking_aware_gate,
)
from .composite_streaming import streaming_alpha_check_and_refit
from .composite.transforms import (
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
            from .configs import CompositeTargetDiscoveryConfig
            config = CompositeTargetDiscoveryConfig(**config)
        self.config = config
        self._patterns_compiled: list[re.Pattern] = [
            re.compile(p) for p in config.forbidden_base_patterns
        ]

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
            base_full = _extract_column_array(df, spec.base_column)
            transform = get_transform(spec.transform_name)
            valid = transform.domain_check(y_full, base_full)
            t = np.full(y_full.shape[0], np.nan, dtype=np.float64)
            if valid.any():
                t[valid] = transform.forward(
                    y_full[valid], base_full[valid], spec.fitted_params,
                )
            yield spec.name, t

    # Per-cluster composite (REJECTED -- explicit user decision 2026-05-18):
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
    ) -> CompositeTargetDiscovery:
        """Run :meth:`fit` ``n_bootstrap_runs`` times with different random seeds and keep only specs that survive in at least ``min_keep_fraction * n_bootstrap_runs`` runs.

        Filters "lucky split" wins where a single seed happens to find a spec that does not generalise. Default thresholds (5 runs, 60% majority) match the standard stability-selection literature (Meinshausen-Buhlmann).

        Returns ``self``. After the call, ``self.specs_`` is the stable subset and ``self.stability_counts_`` maps each name to its survival count.
        """
        if n_bootstrap_runs <= 1:
            return self.fit(df, target_col, feature_cols, train_idx, val_idx, test_idx)
        from collections import Counter

        base_seed = int(self.config.random_state)
        keep_counter: Counter = Counter()
        spec_by_name: dict[str, CompositeSpec] = {}
        for i in range(int(n_bootstrap_runs)):
            self.config.random_state = base_seed + i * 7919
            try:
                self.fit(df, target_col, feature_cols, train_idx, val_idx, test_idx)
            except Exception as _exc:
                logger.warning(
                    "[CompositeTargetDiscovery.stability] bootstrap run %d failed: %s",
                    i, _exc,
                )
                continue
            for spec in self.specs_:
                keep_counter[spec.name] += 1
                spec_by_name.setdefault(spec.name, spec)
        # Restore base seed and write the stable spec set.
        self.config.random_state = base_seed
        threshold = max(1, int(min_keep_fraction * n_bootstrap_runs))
        stable_names = [n for n, c in keep_counter.items() if c >= threshold]
        self.specs_ = [spec_by_name[n] for n in stable_names if n in spec_by_name]
        self.stability_counts_ = dict(keep_counter)
        logger.info(
            "[CompositeTargetDiscovery.stability] n_runs=%d, threshold=%d/%d. "
            "Kept %d spec(s); counts: %s",
            n_bootstrap_runs, threshold, n_bootstrap_runs,
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
                # transform. Reproduced by fuzz c0047 (multi-base
                # auto-promoted to linresM-num_1+num_dep).
                "extra_base_columns": tuple(getattr(s, "extra_base_columns", ()) or ()),
            }
            for s in getattr(self, "specs_", [])
        ]

    def report(self) -> list[dict[str, Any]]:
        """All evaluated candidates including rejected ones with reasons.

        Wave 26 P1 fix (2026-05-20): pre-fix did a shallow ``list(...)``
        over a list of dicts, so the outer list was decoupled but the
        inner per-candidate dicts (incl. ``score`` / ``reason`` / base
        column metadata) were returned by REFERENCE. A caller doing
        ``discovery.report()[0]["score"] = 999`` mutated the persisted
        internal record; subsequent ``report()`` calls returned the
        corrupted value.
        Sibling ``export_specs()`` above (~30 lines up) already builds
        fresh inner dicts via comprehension -- this was inconsistent
        defensive copying.
        """
        return [dict(r) for r in getattr(self, "report_", [])]

    @property
    def tiny_rerank_scores_(self) -> dict[str, float]:
        """Per-spec tiny CV-RMSE on y-scale (after Phase B rerank).

        Empty when ``screening="mi"`` or rerank didn't run. Keyed by
        spec name. Useful for surfacing "why did this composite get
        kept / rejected" diagnostics.
        """
        return dict(getattr(self, "_tiny_rerank_scores", {}))

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

        Wave 26 P1 fix (2026-05-20): same shape as ``report()`` above.
        Inner dicts are defensively copied to prevent caller mutation
        from poisoning the persisted internal state.
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
        cols_arrays = [_extract_column_array(df, c)[idx] for c in cols]
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
        }


# Bind the carved-out methods onto the class. Each carved sibling exposes the
# method as a module-level function taking ``self`` as the first argument; the
# binding here makes ``CompositeTargetDiscovery._tiny_model_rerank`` resolve
# to that function so ``self._tiny_model_rerank(...)`` call sites keep
# working unchanged.
from ._composite_discovery_tiny_rerank import _tiny_model_rerank as _tiny_model_rerank_impl  # noqa: E402
from ._composite_discovery_auto_base import _auto_base as _auto_base_impl  # noqa: E402
from ._composite_discovery_fit import fit as _fit_impl  # noqa: E402
from ._composite_discovery_filter import _filter_features as _filter_features_impl  # noqa: E402
from ._composite_discovery_stacked import (  # noqa: E402
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
from .composite_auto_detect import (  # noqa: E402,F401
    _GROUP_DETECT_DEFAULT_MIN_UNIQUE,
    _GROUP_DETECT_DEFAULT_MAX_UNIQUE,
    _GROUP_DETECT_DEFAULT_MIN_SIZE_RATIO,
)
from .composite_cache import (  # noqa: E402,F401
    _DISCOVERY_SIGNATURE_SAMPLE_N,
)
from .composite.transforms.interaction_bases import (  # noqa: E402,F401
    _INTERACTION_OPS_DEFAULT,
)

# Dependent helper re-exports.
from .composite_forward_stepwise import (  # noqa: E402,F401
    _MULTI_BASE_DEFAULT_MAX_K,
    _MULTI_BASE_DEFAULT_MIN_MARGINAL_GAIN,
)
from .composite_streaming import (  # noqa: E402,F401
    _STREAMING_DEFAULT_Z_THRESHOLD,
    _STREAMING_DEFAULT_MIN_BUFFER_N,
)
from .composite_bayesian import (  # noqa: E402,F401
    _BAYESIAN_ALPHA_DEFAULT_N_BOOTSTRAP,
    _BAYESIAN_ALPHA_DEFAULT_CI_LEVEL,
)


# ----------------------------------------------------------------------
