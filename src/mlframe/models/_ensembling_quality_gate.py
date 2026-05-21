"""Member-quality gate for ``mlframe.models.ensembling``.

Split out of ``ensembling.py`` to keep the parent below the 1k-line monolith
threshold. ``compute_member_quality_gate`` is re-exported from the parent
so historical
``from mlframe.models.ensembling import compute_member_quality_gate``
imports continue to resolve.

The gate filters ensemble members by per-member MAE / STD relative to the
ensemble median, using the absolute / relative thresholds the caller
configures. ``score_ensemble`` calls it once per (flavour x split) cell.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ._ensembling_base import _per_member_mae_std

logger = logging.getLogger("mlframe.models.ensembling")


def compute_member_quality_gate(
    preds_list: Sequence,
    *,
    max_mae: float = 0.0,
    max_std: float = 0.0,
    max_mae_relative: float = 2.5,
    max_std_relative: float = 2.5,
    sample_weight: Optional[np.ndarray] = None,
    group_ids: Optional[np.ndarray] = None,
) -> Tuple[List[int], List[Tuple[int, str]], dict]:
    """Cross-member outlier filter for ensemble preds.

    Computes per-member MAE / STD against the cross-member median and
    returns the indices to keep + a list of (excluded_index, reason)
    tuples + a stats dict (median_mae, median_std, rel_mae_threshold,
    rel_std_threshold). Pure: no logging, no side effects.

    Use this from a SUITE-level scorer (e.g. ``score_ensemble``) to do
    the gate ONCE before iterating ensemble flavors -- the previous
    behaviour ran the same filter inside ``ensemble_probabilistic_
    predictions`` once per flavor x split, printing the same
    "ens member ... excluded ..." line ~20 times per suite call on
    a 4-model x 5-flavor x 2-split layout (regression suite, 2026-05
    prod log).

    Returns
    -------
    kept_indices : list[int]
        Indices into ``preds_list`` to keep. May equal all indices.
    excluded : list[tuple[int, str]]
        (member_index, human-readable-reason) for each dropped member.
    stats : dict
        ``{"median_mae", "median_std", "rel_mae_threshold",
          "rel_std_threshold", "per_member_mae", "per_member_std"}``.
    """
    n = len(preds_list)
    if n <= 1:
        # K=1 has no peer to compare against; gate is trivially a no-op.
        return list(range(n)), [], {}
    if n == 2:
        # C-P2-12: K=2 cannot drop an outlier (both members get identical |a-b|/2 distance from the
        # 2-element median, so no member is "more outlier" than the other). The legacy gate just
        # returned (kept-all, no-stats); this branch additionally stamps a ``k2_disagreement``
        # observability scalar = mean(|a-b|) so the suite caller / downstream metadata can surface
        # when the two members disagreed heavily even though no drop is possible. The blend
        # downstream is still arithmetic mean / harm / etc. of both members.
        try:
            _a = np.asarray(preds_list[0], dtype=np.float64)
            _b = np.asarray(preds_list[1], dtype=np.float64)
            if _a.shape == _b.shape:
                _disagreement = float(np.mean(np.abs(_a - _b)))
            else:
                _disagreement = float("nan")
        except Exception:
            _disagreement = float("nan")
        return list(range(n)), [], {"k2_disagreement": _disagreement, "filter_too_restrictive": False}

    arr = np.asarray(preds_list, dtype=np.float64)
    # Weighted median when sample_weight is supplied (numpy>=1.22). Falls back to unweighted on
    # older numpy or weight-shape mismatch. group_ids further coarsens the per-row weighting by
    # one-row-per-group (so a single dense group doesn't dominate the median) -- when supplied we
    # collapse per-group weight sums and feed those to the quantile.
    if sample_weight is not None:
        _sw = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if group_ids is not None and arr.ndim >= 2 and _sw.shape[0] == arr.shape[1]:
            _gids = np.asarray(group_ids).reshape(-1)
            if _gids.shape[0] == arr.shape[1]:
                _uniq, _inv = np.unique(_gids, return_inverse=True)
                _group_sum = np.zeros(_uniq.shape[0], dtype=np.float64)
                np.add.at(_group_sum, _inv, _sw)
                # Broadcast back to per-row -- each row gets its group's aggregate divided by group size.
                _group_size = np.zeros(_uniq.shape[0], dtype=np.float64)
                np.add.at(_group_size, _inv, 1.0)
                _sw = (_group_sum / np.where(_group_size > 0, _group_size, 1.0))[_inv]
    # ``np.nanmedian`` over ``np.nanquantile(arr, 0.5, ...)``: the q=0.5 codepath
    # in nanquantile dispatches through ``apply_along_axis``, which iterates the
    # non-axis dimensions in Python (200k rows -> 200k 1-D calls); nanmedian
    # uses numpy's dedicated C reduction. Bench at (K=3, N=200_000): 13.5 s ->
    # 49 ms (~275x); 3-D (K=3, N=200_000, C=4): 54 s -> 250 ms (~215x). Output
    # matches at machine epsilon (max abs diff 2.22e-16) since both ignore
    # NaNs by definition; we still drop into the unweighted path because the
    # cross-member median is uniformly weighted across the K-axis by
    # construction (row-level sample_weight only affects the downstream
    # per-member MAE aggregation, not this median).
    median_preds = np.nanmedian(arr, axis=0)
    # Vectorised per-member MAE/STD: collapses the explicit Python loop to a single broadcast
    # over (K, N, ...). diffs has shape (K, N[, C]); collapse all non-member axes to a per-member
    # scalar via mean / population-std. LOOP-MAE / PER-MEMBER-MAE-LOOP fix; bench script in
    # `_benchmarks/bench_ensemble_mae.py` shows ~5-50x speedup over the Python loop for K>=4.
    per_member_mae, per_member_std = _per_member_mae_std(arr, median_preds)

    # NO-SW: weighted aggregation of the per-member MAE / STD when sample_weight supplied. The
    # statistic is "how far is THIS member from the cross-member median, averaged over rows". When
    # rows carry weights we use the same weights to average per-row absolute deviations -- this
    # only matters if sample_weight varies a lot, which is the regime where unweighted would
    # otherwise misrank.
    if sample_weight is not None:
        _sw_b = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
        if arr.ndim >= 2 and _sw_b.shape[0] == arr.shape[1]:
            # Recompute per-member MAE using np.average to honour the weights.
            diffs = np.abs(arr - median_preds)
            if arr.ndim == 2:
                per_member_mae = np.array([float(np.average(diffs[i], weights=_sw_b)) for i in range(diffs.shape[0])])
                per_member_std = np.array([
                    float(np.sqrt(np.average((diffs[i] - per_member_mae[i]) ** 2, weights=_sw_b)))
                    for i in range(diffs.shape[0])
                ])
            else:
                # (K, N, C) -- average over (N, C) with broadcasted sample_weight on the N axis.
                per_member_mae = np.array([
                    float(np.average(diffs[i].mean(axis=-1), weights=_sw_b)) for i in range(diffs.shape[0])
                ])
                per_member_std = np.array([
                    float(np.sqrt(np.average((diffs[i].mean(axis=-1) - per_member_mae[i]) ** 2, weights=_sw_b)))
                    for i in range(diffs.shape[0])
                ])

    # Wave 21 P1: nanmedian so a NaN-bearing per-member statistic doesn't
    # silently make the threshold NaN -- pre-fix that NaN threshold then
    # made `tot_mae > rel_mae_threshold` return False for every member,
    # silently keeping members the gate was supposed to drop.
    median_mae = float(np.nanmedian(per_member_mae))
    median_std = float(np.nanmedian(per_member_std))
    rel_mae_threshold = max_mae_relative * median_mae if max_mae_relative > 0 else 0.0
    rel_std_threshold = max_std_relative * median_std if max_std_relative > 0 else 0.0

    kept: list = []
    excluded: list = []
    for i in range(n):
        tot_mae = float(per_member_mae[i])
        tot_std = float(per_member_std[i])
        abs_violation = (max_mae > 0 and tot_mae > max_mae) or (max_std > 0 and tot_std > max_std)
        rel_violation = (rel_mae_threshold > 0 and tot_mae > rel_mae_threshold) or (rel_std_threshold > 0 and tot_std > rel_std_threshold)
        if abs_violation or rel_violation:
            reason_parts = []
            if abs_violation:
                reason_parts.append(f"abs(mae>{max_mae}|std>{max_std})")
            if rel_violation:
                reason_parts.append(
                    f"rel(mae>{rel_mae_threshold:.4f}|std>{rel_std_threshold:.4f}; " f"median_mae={median_mae:.4f},median_std={median_std:.4f})"
                )
            excluded.append((i, f"mae={tot_mae:.4f}, std={tot_std:.4f} [{'; '.join(reason_parts)}]"))
        else:
            kept.append(i)
    # Defensive: if every member was excluded, the filter is too tight
    # for the data; fall back to the original list (else
    # ensemble_probabilistic_predictions returns a degenerate empty
    # ensemble downstream).
    if not kept:
        return (
            list(range(n)),
            [],
            {
                "median_mae": median_mae,
                "median_std": median_std,
                "rel_mae_threshold": rel_mae_threshold,
                "rel_std_threshold": rel_std_threshold,
                "per_member_mae": per_member_mae,
                "per_member_std": per_member_std,
                "filter_too_restrictive": True,
            },
        )
    return (
        kept,
        excluded,
        {
            "median_mae": median_mae,
            "median_std": median_std,
            "rel_mae_threshold": rel_mae_threshold,
            "rel_std_threshold": rel_std_threshold,
            "per_member_mae": per_member_mae,
            "per_member_std": per_member_std,
        },
    )


