"""``BaselineReport`` NamedTuple + ``SCHEMA_VERSION`` for ``dummy_baselines``.

Split out of ``dummy_baselines.py`` so the report-type definition lives in
a leaf module the rest of the dummy_baselines siblings can import without
re-entering the parent. The parent re-exports both names so historical
``from mlframe.training.dummy_baselines import BaselineReport`` imports
continue to resolve identity-equal.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------


SCHEMA_VERSION = "1.0"


class BaselineReport(NamedTuple):
    """Result of ``compute_dummy_baselines`` for one target.

    Attributes
    ----------
    target_type
        String name of the TargetTypes value (e.g. ``"regression"``).
    target_name
        Specific target column / output name.
    table
        DataFrame indexed by baseline-name with columns for the
        per-split, per-metric values. ``failed`` boolean column flags
        rows whose metrics computation raised.
    strongest
        Name of the strongest baseline by primary metric on the
        reference split (val with test fallback). ``None`` when both
        splits are degenerate or fewer than 2 baselines produced
        finite metrics.
    primary_metric
        The metric name used for strongest-pick (e.g. ``"val_RMSE"``).
    ts_period_used
        Inferred TS period for the strongest TS baseline (None for
        non-TS targets or when no TS baseline picked).
    plot_path
        Path to the strongest baseline's overlay PNG (None when not
        rendered -- short-circuit, no consumer, or suppressed).
    elapsed_s
        Wall time of the entire baseline computation.
    n_train, n_val, n_test
        Row counts of the splits.
    n_train_finite, n_val_finite, n_test_finite
        Finite-target row counts (surfaces all-NaN target columns).
    extras
        Free-form dict for target-type-specific diagnostics
        (per-output strongest-pick block for multi-output regression,
        ts_period_candidates, etc.).
    """

    target_type: str
    target_name: str
    table: pd.DataFrame
    strongest: str | None
    primary_metric: str | None
    ts_period_used: int | None
    plot_path: str | None
    elapsed_s: float
    n_train: int
    n_val: int
    n_test: int
    n_train_finite: int
    n_val_finite: int
    n_test_finite: int
    extras: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (schema_version + NaN->None)."""
        # Replace NaN with None so json.dumps() succeeds.
        # Handle both Python float AND numpy.float* - orjson does NOT
        # accept numpy scalars (TypeError: numpy.float64 not serializable);
        # pd.DataFrame iter rows yields numpy scalars on numeric columns.
        # Cast every numpy floating to native float after the finite-check.
        def _scrub(v: Any) -> Any:
            if isinstance(v, np.floating):
                if not np.isfinite(v):
                    return None
                return float(v)
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, float) and not np.isfinite(v):
                return None
            return v

        # Convert table to {baseline_name: {col: value}} with NaN -> None
        table_dict: dict[str, dict[str, Any]] = {}
        for idx, row in self.table.iterrows():
            table_dict[str(idx)] = {col: _scrub(row[col]) for col in self.table.columns}

        return {
            "schema_version": SCHEMA_VERSION,
            "target_type": self.target_type,
            "target_name": self.target_name,
            "data": table_dict,
            "strongest": self.strongest,
            "primary_metric": self.primary_metric,
            "ts_period_used": self.ts_period_used,
            "plot_path": self.plot_path,
            "elapsed_s": self.elapsed_s,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "n_train_finite": self.n_train_finite,
            "n_val_finite": self.n_val_finite,
            "n_test_finite": self.n_test_finite,
            # Scrub raw prediction arrays from extras
            # before serialization (they bloat metadata.pkl and are
            # not useful at load time -- they're consumed
            # synchronously by the pre-training overlay plotter).
            "extras": {
                k: v for k, v in self.extras.items()
                if k not in ("strongest_val_preds", "strongest_test_preds")
            },
        }

    def format_text(self, default_level: str = "INFO") -> str:
        """Render report for log emission.

        At ``default_level='INFO'`` (default per Operator Contract
        guarantee 1), emit only the verdict line(s) + plot path.
        Promote to ``'DEBUG'`` to get the full table.
        """
        lines: list[str] = []
        # Header with finite-n summary
        ts_tag = ""
        if self.ts_period_used is not None:
            ts_tag = f" ts_period={self.ts_period_used}"
        lines.append(
            f"[DUMMY_BASELINES] target='{self.target_name}' {self.target_type}"
            f"{ts_tag} n_train={self.n_train} (finite={self.n_train_finite})"
            f" n_val={self.n_val} (finite={self.n_val_finite})"
            f" n_test={self.n_test} (finite={self.n_test_finite})"
        )

        if self.strongest is None:
            lines.append(
                f"[DUMMY_BASELINES] target='{self.target_name}' strongest=None"
                f" (both splits degenerate; review table manually)"
            )
            return "\n".join(lines)

        # Verdict line with strongest baseline metric.
        try:
            strongest_row = self.table.loc[self.strongest]
            primary_val = strongest_row.get(self.primary_metric, float("nan"))
            # Lift vs mean / prior trivial baseline (whichever is in table).
            trivial_name = "mean" if "mean" in self.table.index else (
                "prior" if "prior" in self.table.index else None
            )
            lift_str = ""
            if trivial_name is not None and trivial_name != self.strongest:
                trivial_val = self.table.loc[trivial_name].get(self.primary_metric, float("nan"))
                if np.isfinite(primary_val) and np.isfinite(trivial_val) and trivial_val != 0:
                    # Wave 20 fix: registry dispatcher. The previous tuple
                    # whitelist missed val_F1, val_accuracy, val_R2, val_AP,
                    # val_precision, val_recall and reported inverted lift_pct
                    # in the operator-facing summary (P1: model still trained
                    # correctly upstream, but the user-visible verdict was
                    # silently flipped).
                    from .metrics_registry import metric_name_higher_is_better as _mhb
                    _direction = _mhb(self.primary_metric)
                    if _direction is True:
                        lift_pct = (primary_val - trivial_val) / abs(trivial_val) * 100
                    else:
                        # False (lower-is-better) or None (unknown) -> use
                        # the minimize convention; unknowns default toward
                        # not-flipping silently in the operator's face.
                        lift_pct = (trivial_val - primary_val) / abs(trivial_val) * 100
                    lift_str = f" lift_vs_{trivial_name}={lift_pct:+.1f}%"
            tie_suffix = ""
            paired = self.extras.get("paired_bootstrap") if isinstance(self.extras, dict) else None
            if paired and paired.get("p_strongest_beats") is not None:
                pct = int(round(paired["p_strongest_beats"] * 100))
                if self.extras.get("tie"):
                    tie_suffix = f" (beats runner-up in {pct}% of resamples - TIE, treat as noise)"
            lines.append(
                f"[DUMMY_BASELINES] target='{self.target_name}'"
                f" strongest={self.strongest}"
                f" {self.primary_metric}={primary_val:.4f}"
                f"{lift_str}"
                f" (n_baselines={len(self.table)}, full table at DEBUG){tie_suffix}"
            )
            # Paired-bootstrap delta vs runner-up with 95% CI.
            if paired:
                ru = paired.get("runner_up", "?")
                delta = paired.get("delta")
                ci = paired.get("delta_ci")
                p = paired.get("p_strongest_beats")
                if delta is not None and ci is not None and p is not None:
                    metric_short = self.primary_metric.replace("val_", "")
                    lines.append(
                        f"[DUMMY_BASELINES] target='{self.target_name}'"
                        f" Delta_{metric_short} vs runner-up ({ru}) = {delta:+.4f}"
                        f" [95% bootstrap CI: {ci[0]:+.4f}, {ci[1]:+.4f}];"
                        f" beats runner-up in {int(round(p * 100))}% of resamples"
                    )
            # Bootstrap CI line when present (small-n grounding).
            ci = self.extras.get("bootstrap_ci") if isinstance(self.extras, dict) else None
            if ci and "val" in ci:
                lo, point, hi = ci["val"]
                lines.append(
                    f"[DUMMY_BASELINES] target='{self.target_name}'"
                    f" strongest val 95% bootstrap CI: [{lo:.4f}, {hi:.4f}]"
                    f" (n_resamples={self.extras.get('bootstrap_ci_n_resamples', 1000)})"
                )
        except Exception as e:
            lines.append(
                f"[DUMMY_BASELINES] target='{self.target_name}'"
                f" strongest={self.strongest} (verdict format failed: {e})"
            )

        # Plot path line (when present).
        if self.plot_path:
            lines.append(
                f"[DUMMY_BASELINES] target='{self.target_name}'"
                f" overlay plot saved: {self.plot_path}"
            )

        # Extras: per-output multi-output strongest-pick.
        if "per_output_strongest" in self.extras:
            for out_idx, info in enumerate(self.extras["per_output_strongest"]):
                lines.append(
                    f"[DUMMY_BASELINES] target='{self.target_name}' Y[{out_idx}]:"
                    f" strongest={info['name']} ({info['primary_metric']}="
                    f"{info['primary_value']:.4f}, normalized={info.get('normalized', float('nan')):.3f})"
                )
            if "cross_output_strongest" in self.extras:
                xo = self.extras["cross_output_strongest"]
                lines.append(
                    f"[DUMMY_BASELINES] target='{self.target_name}'"
                    f" cross-output normalized strongest={xo['name']}"
                    f" (mean_normalized_RMSE={xo['mean_normalized']:.4f})"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------
# Hash recipe for sweep-orchestrator memoization
# ---------------------------------------------------------------------


