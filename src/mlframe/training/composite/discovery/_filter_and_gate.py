"""Post-MI-screening candidate filter/sort/gate pipeline, lifted out of ``_fit.py`` to keep it
under the 1k-line monolith threshold. Called once from ``fit`` between the transform-evaluation
loop and the Phase B tiny-model rerank.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from . import CompositeTargetDiscovery

from ..spec import CompositeSpec
from ._eval_stats import (
    apply_alpha_drift_gate,
    apply_fdr_control_to_candidates,
    apply_linear_residual_diff_collapse,
)


def filter_sort_and_gate_candidates(
    self: "CompositeTargetDiscovery",
    candidates: list[dict],
    df: Any,
    train_idx: np.ndarray,
    y_full: np.ndarray,
    y_train: np.ndarray,
    extract_column_array: Callable[..., np.ndarray],
) -> list[CompositeSpec]:
    """Apply FDR control, the eps_mi_gain gate, top-k truncation, the alpha-drift and
    linear-residual/diff collapse gates, and the structural-fragility gate, in that order.
    """
    if bool(getattr(self.config, "mi_gain_fdr_control", True)):
        apply_fdr_control_to_candidates(
            candidates,
            alpha=float(getattr(self.config, "mi_gain_fdr_alpha", 0.10)),
        )

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
        extract_column_array=extract_column_array,
    )

    # Collapse redundant linear_residual -> diff when alpha ~ 1 and beta ~ 0 (linear_residual
    # has zero information advantage over diff but carries 2 fitted params); lifted to
    # ``_eval_stats`` to keep this file under the monolith threshold.
    kept_specs = apply_linear_residual_diff_collapse(
        self, kept_specs, df=df, train_idx=train_idx, y_train=y_train,
        extract_column_array=extract_column_array,
    )

    # Structural-fragility gate FIRST (cheap, train-only): drop base-additive specs whose inverse re-injects a
    # per-group-level base (fragile on unseen groups). Its inputs (group_ids/train_idx/y_full + per-spec
    # base_column/transform_name/fitted_params) are all ready here and it reads NO rerank output, so running it before
    # the expensive tiny-rerank means the doomed specs never pay for a tiny-model fit (prod TVT: 11/11 base-additive
    # specs dropped here -> the ~25-30 min rerank is skipped entirely when no survivor remains). Pruning a spec the gate
    # would drop anyway can only let a LESS-fragile spec into the rerank's top_m; the downstream val/honest gates still
    # apply to whatever survives, so no fragile spec slips through.
    if kept_specs and getattr(self.config, "structural_fragility_gate_enabled", True):
        from ._yscale_holdout_gate import apply_structural_fragility_gate

        kept_specs = apply_structural_fragility_gate(self, df, kept_specs, train_idx, y_full)

    return kept_specs
