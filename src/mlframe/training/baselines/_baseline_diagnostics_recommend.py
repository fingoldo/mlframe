"""Recommendation builder for BaselineDiagnostics.

Carved out of ``baseline_diagnostics`` via method-rebinding (W10E pattern).
"""
from __future__ import annotations

import math


def _build_recommendation(
    self,
    ablation: list,
    init_score_baseline,
) -> tuple:
    """Three-way classifier:

    * **high_potential** - max(ablation delta%) >= high_potential_min_dominance_pct
      AND init_score baseline did NOT already extract that signal
      (delta_vs_raw_pct stayed > init_score_optimal_threshold_pct,
      i.e. residual still has structure).
    * **marginal** - max ablation delta% in [marginal_threshold_pct,
      high_potential_min_dominance_pct).
    * **unlikely_to_help** - max ablation delta% < marginal_threshold_pct
      OR init_score baseline already matches raw within
      init_score_optimal_threshold_pct (residual is mostly noise).
    """
    if not ablation:
        return "unlikely_to_help", "no ablation entries (FI all-zero or no features)"

    # default=-inf (not a bare max() over a possibly-empty filtered generator): every ablation entry's
    # delta_pct can be non-finite (e.g. a refit that degenerates to constant/NaN predictions after
    # dropping a specific feature on a tiny/degenerate sample) even though ablation itself is non-empty --
    # the raw-fit metric checked earlier is finite, but that says nothing about the per-drop refits. A bare
    # max() previously raised ValueError there, which the caller's broad except then surfaced as an opaque
    # "internal_error" skip instead of the informative unlikely_to_help verdict this situation actually
    # warrants.
    max_dom = max((e.delta_pct for e in ablation if math.isfinite(e.delta_pct)), default=float("-inf"))
    cfg = self.config
    if not math.isfinite(max_dom):
        return "unlikely_to_help", "every ablation entry's delta_pct was non-finite (no dominant feature could be assessed)"

    # Note on sign: ``init_score delta%`` uses the same convention as ablation
    # delta% - positive means init_score baseline performed WORSE than raw
    # (residual still has structure). Negative or near-zero means init_score
    # baseline already matches raw, so composite-mode unlikely to extract more signal.
    init_score_sufficient = init_score_baseline is not None and abs(init_score_baseline.delta_vs_raw_pct) <= cfg.init_score_optimal_threshold_pct

    if max_dom >= cfg.high_potential_min_dominance_pct and not init_score_sufficient:
        reason = f"top ablation delta%={max_dom:.2f} >= {cfg.high_potential_min_dominance_pct:.2f} " "(strong dominant feature)"
        if init_score_baseline is not None:
            reason += f"; init_score baseline still off raw by " f"{init_score_baseline.delta_vs_raw_pct:+.2f}% (residual has structure)"
        return "high_potential", reason

    if init_score_sufficient:
        return (
            "unlikely_to_help",
            f"init_score baseline matches raw within "
            f"{cfg.init_score_optimal_threshold_pct:.2f}pct "
            f"(delta={init_score_baseline.delta_vs_raw_pct:+.2f}%); "
            "native residual learning already captures the dominant signal",
        )

    if max_dom >= cfg.marginal_threshold_pct:
        return (
            "marginal",
            f"top ablation delta%={max_dom:.2f} in " f"[{cfg.marginal_threshold_pct:.2f}, {cfg.high_potential_min_dominance_pct:.2f})",
        )
    return (
        "unlikely_to_help",
        f"top ablation delta%={max_dom:.2f} < {cfg.marginal_threshold_pct:.2f} " "(no dominant features)",
    )
