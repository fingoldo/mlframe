"""A wall-clock budget for the post-fit diagnostics block as a WHOLE.

One diagnostic used to carry a 20-second cap of its own while the block around it had none, so a production run
skipped the interaction-strength surface for being projected at 20s and then spent six and a half minutes on the
diagnostics that had no budget -- longer than the model fit itself (4m53s). A cap that binds on one member of a
group and on nothing else does not limit anything; it just picks an arbitrary victim.

The budget is advisory and checked BETWEEN diagnostics, never inside one: a half-drawn figure is worse than a
missing one, and no diagnostic here is interruptible. Everything skipped is named in a single line at the end,
so a shortened report never looks like a complete one.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# Diagnostics whose cost scales with rows or with the model's own structure, as opposed to the per-split metric
# panels. These are what get restricted to a single model when several near-identical ones are being reported.
HEAVY_DIAGNOSTICS: frozenset = frozenset({"pdp_ice", "pdp_2d", "interaction_strength", "slice_finder", "shap", "shap_interactions", "shap_per_instance"})


class HeavyDiagnosticsPolicy:
    """Decides whether the expensive diagnostics run for THIS model.

    A production run rendered the full set for five ensemble aggregations of two members correlated at 0.996 --
    every flavour returned the same test AUC and Brier to two decimals, so the SHAP and PDP surfaces were
    redrawn five times to describe the same model. Restricting them to one model is a change of scope, not of
    budget: the metrics and calibration panels, which exist precisely to compare members, still run for all.
    """

    def __init__(self, mode: str = "best", is_primary: bool = True) -> None:
        """``mode`` is "best" (heavy diagnostics on the primary model only) or "all" (the previous behaviour)."""
        # A caller passing "" -- or "ALL " with a stray space, or any typo -- is asking for something this
        # cannot honour, and silently turning it into "best" hides the mistake. The comment above this line used
        # to claim that was surfaced; nothing was, so an unrecognised mode quietly got the RESTRICTIVE behaviour.
        self.mode = "best" if mode is None else str(mode).strip().lower()
        if self.mode not in ("best", "all"):
            logger.warning(
                "DiagnosticsBudget: heavy_diagnostics_for=%r is not one of 'best' / 'all'; falling back to 'best' "
                "(heavy diagnostics on the primary model only).",
                mode,
            )
            self.mode = "best"
        self.is_primary = bool(is_primary)

    def allows(self, name: str) -> bool:
        """True when diagnostic ``name`` may run for this model."""
        if self.mode == "all" or name not in HEAVY_DIAGNOSTICS:
            return True
        return self.is_primary


class DiagnosticsBudget:
    """Elapsed-time gate for a sequence of independent diagnostics.

    ``max_seconds <= 0`` disables the gate entirely, which is the escape hatch for a caller who wants the full
    report regardless of cost.
    """

    def __init__(self, max_seconds: float, *, verbose: bool = True, policy: "HeavyDiagnosticsPolicy | None" = None) -> None:
        """Start the clock. ``policy`` decides scope (which diagnostics apply here); the budget decides time."""
        self.max_seconds = float(max_seconds or 0.0)
        self.verbose = verbose
        self.policy = HeavyDiagnosticsPolicy(mode="all") if policy is None else policy
        self.out_of_scope: List[str] = []
        self._t0 = time.perf_counter()
        self.skipped: List[str] = []

    @property
    def elapsed(self) -> float:
        """Seconds since the block started."""
        return time.perf_counter() - self._t0

    def exhausted(self) -> bool:
        """True once the block has spent its budget; always False when the budget is disabled."""
        return self.max_seconds > 0 and self.elapsed >= self.max_seconds

    def run(self, name: str, fn: Callable[[], T]) -> Optional[T]:
        """Run one diagnostic unless it is out of scope for this model or the budget is already spent."""
        if not self.policy.allows(name):
            self.out_of_scope.append(name)
            return None
        if self.exhausted():
            self.skipped.append(name)
            return None
        return fn()

    def report(self) -> None:
        """Say what was dropped and why, once. Silent when nothing was."""
        if self.out_of_scope:
            logger.info(
                "  [diagnostics] %d model-explanation diagnostic(s) not rendered for this model (%s): they are "
                "restricted to the primary model, because redrawing them per ensemble variant describes the same "
                "model repeatedly. Set ReportingConfig.heavy_diagnostics_for='all' to render them for every model.",
                len(self.out_of_scope), ", ".join(sorted(set(self.out_of_scope))),
            )
        if not self.skipped:
            return
        logger.warning(
            "  [diagnostics] budget of %.0fs exhausted after %.0fs; %d diagnostic(s) skipped: %s. This report is "
            "INCOMPLETE -- raise ReportingConfig.diagnostics_max_seconds (or set it to 0 to disable the budget) "
            "to render them.",
            self.max_seconds, self.elapsed, len(self.skipped), ", ".join(self.skipped),
        )


__all__ = ["HEAVY_DIAGNOSTICS", "DiagnosticsBudget", "HeavyDiagnosticsPolicy"]
