"""Thin facade for ``mlframe.models.ensembling``.

The implementation was split across themed sibling modules to drop the
monolith below 1k LOC:

  - ``_ensembling_base`` (shared helpers, constants, numba probe,
    StreamingAccumulator / Welford, ``combine_probs``, ``rrf_ensemble``,
    ``build_predictive_kwargs``, ``compute_high_correlation_pairs``,
    ``batch_numaggs``, ``enrich_ensemble_preds_with_numaggs``,
    ``_stacked_corrcoef``, ``_per_member_mae_std``).
  - ``_ensembling_quality_gate`` (``compute_member_quality_gate``).
  - ``_ensembling_predict`` (``ensemble_probabilistic_predictions`` +
    streaming variant).
  - ``_ensembling_process_method`` (``_process_single_ensemble_method``).
  - ``_ensembling_score`` (``score_ensemble``).

This file keeps ``EnsembleLeaderboard``, ``_build_votenrank_leaderboard_from_results``
and ``compare_ensembles`` because they reference ``score_ensemble``
(circular if pulled into a sibling) and they form the small remaining
public-facade surface.

Re-exports below preserve every historical
``from mlframe.models.ensembling import X`` import path.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Leaf helpers, constants, classes, numba probe.
from ._ensembling_base import (  # noqa: E402,F401
    SIMPLE_ENSEMBLING_METHODS,
    RANK_FUSION_METHODS,
    StreamingAccumulator,
    _WelfordAccumulator,
    _stacked_corrcoef,
    _per_member_mae_std,
    _per_member_mae_std_njit,
    _HAS_NUMBA_PER_MEMBER,
    batch_numaggs,
    enrich_ensemble_preds_with_numaggs,
    _rrf_aggregate_probs,
    rrf_ensemble,
    combine_probs,
    build_predictive_kwargs,
    compute_high_correlation_pairs,
)
# Siblings.
from ._ensembling_quality_gate import compute_member_quality_gate  # noqa: E402,F401
from ._ensembling_predict import (  # noqa: E402,F401
    ensemble_probabilistic_predictions,
    ensemble_probabilistic_predictions_streaming,
)
from ._ensembling_process_method import _process_single_ensemble_method  # noqa: E402,F401
from ._ensembling_score import score_ensemble  # noqa: E402,F401


class EnsembleLeaderboard:
    """Thin wrapper around ``votenrank.Leaderboard`` that exposes the per-flavour rank table plus a
    ``to_csv`` helper for the suite wrapper to materialise to disk. The wrapper also stores the raw
    metric table so a caller can re-rank with different methods without re-instantiating.

    REG-RRF-DROPPED: regression suites still pass through here; classification-only methods are
    discovered automatically (the source flavour names use the same internal naming convention as
    ``_process_single_ensemble_method``) and rank-fusion entries are excluded when ``is_regression``.
    """

    def __init__(self, table: "pd.DataFrame", lb: Any, is_regression: bool) -> None:
        self.table = table
        self.lb = lb
        self.is_regression = bool(is_regression)

    def rank_all(self, **kwargs):
        return self.lb.rank_all(**kwargs)

    def to_csv(self, path: str, **kwargs) -> None:
        # Persist the underlying score table; the rank-method table can be re-derived from it.
        self.table.to_csv(path, **kwargs)


def _build_votenrank_leaderboard_from_results(res: dict, *, is_regression: bool) -> Optional["EnsembleLeaderboard"]:
    """Construct an EnsembleLeaderboard from a `score_ensemble` result dict.

    Per-flavour rows are the ensemble flavour name (``"arithm"``, ``"harm"``, ...); columns are
    the metric labels harvested from each result's ``metrics`` mapping (``oof.<split>.<metric>``).
    Regression mode skips RRF / votenrank-incompatible flavours -- the rank-fusion ones have no
    rank semantic for continuous y.
    """
    rows: dict[str, dict[str, float]] = {}
    for _flavour, _result in res.items():
        if _flavour.startswith("_"):
            continue
        if is_regression and _flavour.lower().startswith("rrf"):
            continue
        _metrics = getattr(_result, "metrics", None) or (
            _result.get("metrics") if isinstance(_result, dict) else None
        )
        if not _metrics:
            continue
        _flat: dict[str, float] = {}
        for _split, _split_metrics in (_metrics or {}).items():
            if not isinstance(_split_metrics, dict):
                continue
            for _k, _v in _split_metrics.items():
                if isinstance(_v, (int, float, np.floating, np.integer)) and np.isfinite(float(_v)):
                    _flat[f"{_split}.{_k}"] = float(_v)
        if _flat:
            rows[_flavour] = _flat
    if not rows:
        return None
    table = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    # Build a votenrank.Leaderboard. Higher-is-better is the convention; classification flips for
    # error-style metrics is up to the caller (rank methods are unbiased under uniform flip).
    try:
        from mlframe.votenrank import Leaderboard

        lb = Leaderboard(table=table)
        return EnsembleLeaderboard(table=table, lb=lb, is_regression=is_regression)
    except Exception:
        return None


def compare_ensembles(
    ensembles: dict,
    sort_metric: str = "oof.1.integral_error",
    show_plot: bool = True,
    figsize: tuple = (15, 3),
) -> pd.DataFrame:
    # Default flipped from "val.*" to "oof.*": ``val`` is already burned for early-stopping (the model's last-iter
    # snapshot was chosen because it scored best on val), so re-using val to pick a flavour is selecting twice on
    # the same surface. ``oof`` is the cross_val_predict held-out signal -- never seen at fit, never used for ES.
    # Test-set sort still WARNs via logger (the obvious test-set selection bias is preserved); val.* sort now WARNs
    # via warnings.warn(UserWarning) so the message shows up even when the logger is silenced (debugger / scripts).
    import warnings as _warnings_mod
    if isinstance(sort_metric, str) and sort_metric.startswith("test."):
        logger.warning(
            "[compare_ensembles] sort_metric='%s' uses the TEST split; this re-introduces test-set "
            "selection bias. Prefer an 'oof.*' metric for ensemble selection.",
            sort_metric,
        )
    if isinstance(sort_metric, str) and sort_metric.startswith("val."):
        _warnings_mod.warn(
            f"[compare_ensembles] sort_metric='{sort_metric}' uses the VAL split; val is already burned for early "
            f"stopping, so selecting an ensemble flavour on it double-dips the same surface. Prefer 'oof.*' "
            f"(cross_val_predict held-out signal) for ensemble flavour selection.",
            UserWarning,
            stacklevel=2,
        )
    items = []
    for ens_name, ens_perf in ensembles.items():
        perf = copy.deepcopy(ens_perf.metrics)
        for set_name, set_perf in perf.items():
            if set_perf:
                for col in "feature_importances fairness_report robustness_report".split():
                    if col in set_perf:
                        del set_perf[col]
        ser = pd.json_normalize(perf).iloc[0, :]
        ser.name = ens_name
        items.append(ser)

    res = pd.DataFrame(items)
    if sort_metric in res:
        res = res.sort_values(sort_metric)

        if show_plot:
            if "test." in sort_metric:
                val_metric = sort_metric.replace("test.", "val.")
                if val_metric in res:
                    blank_metric = sort_metric.replace("test.", "")
                    ax = res.set_index(val_metric).sort_index()[sort_metric].plot(title=f"Ensembles {blank_metric}, val vs test", figsize=figsize)
                    ax.set_ylabel(sort_metric)
    return res
