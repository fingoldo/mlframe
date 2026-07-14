"""Per-spec rejection ledger for composite-target discovery.

Every discovery gate drops specs, but until now the DOWNSTREAM gates (alpha-drift, linres-collapse, tiny-rerank
threshold/Wilcoxon/per-bin, raw-dominance skip, structural-fragility, y-scale holdout, auto-base null/near-copy/dedup)
recorded their per-spec verdicts only in LOCAL lists that were logged and discarded -- so ``report_`` carried a single
generic "dropped by a downstream filter" string and "why was MY spec rejected by the y-scale gate?" was answerable only
by grepping logs. This module is a single structured ledger on the discovery instance that every gate appends to,
surfaced via the ``rejection_ledger`` property. Additive: the gates keep their local lists for the summary logs.
"""
from __future__ import annotations

from typing import Any, Optional


class RejectStage:
    """Canonical ledger ``stage`` labels (the public, queryable rejection-stage vocabulary)."""

    EPS_MI_GAIN = "eps_mi_gain"
    FDR = "fdr"
    ALPHA_DRIFT = "alpha_drift"
    ALPHA_DRIFT_BASE = "alpha_drift_base"
    LINRES_DIFF_COLLAPSE = "linres_diff_collapse"
    TINY_RERANK_THRESHOLD = "tiny_rerank_threshold"
    HONEST_OOF_FLOOR = "honest_oof_floor"
    TINY_RERANK_WILCOXON = "tiny_rerank_wilcoxon"
    TINY_RERANK_PER_BIN = "tiny_rerank_per_bin"
    RAW_DOMINATES_SKIP = "raw_dominates_skip"
    STRUCTURAL_FRAGILITY = "structural_fragility"
    YSCALE_HOLDOUT = "yscale_holdout"
    HONEST_RMSE = "honest_rmse"
    AUTO_BASE_NULL = "auto_base_null"
    AUTO_BASE_NEAR_COPY_Y = "auto_base_near_copy_y"
    AUTO_BASE_DEDUP = "auto_base_dedup"
    AUTO_BASE_DEMOTE = "auto_base_demote"


def ledger_init(self) -> None:
    """(Re)initialise the per-fit ledger. Called at fit entry + every degenerate early-return so the attribute exists."""
    self.rejection_ledger_ = []


def ledger_append(
    self,
    *,
    spec_name: Any,
    stage: str,
    reason: str = "",
    base_column: Any = "",
    transform_name: Any = "",
    numbers: Optional[dict] = None,
) -> None:
    """Append one ``(spec, rejecting-stage)`` row, reusing values already in scope at the call site.

    Defensive by design: a ledger bookkeeping failure must NEVER abort discovery, so all errors are swallowed.
    """
    try:
        led = getattr(self, "rejection_ledger_", None)
        if led is None:
            led = []
            self.rejection_ledger_ = led
        led.append({
            "spec_name": str(spec_name),
            "base_column": str(base_column or ""),
            "transform_name": str(transform_name or ""),
            "stage": str(stage),
            "reason": str(reason),
            "numbers": dict(numbers or {}),
        })
    except Exception:  # -- ledger is observability-only; never let it break a fit  # nosec B110 - best-effort/optional path, no module logger
        pass
