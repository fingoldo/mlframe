"""Where does a binned estimator overtake a t-statistic? The crossing point, not a point estimate.

The atlas carries one claim about sample size: on ``linear_gaussian_lowdim_n200`` a t-statistic beats a
binned mutual-information estimate, because binning throws away most of what little data there is. That
claim rests on ONE sample size, and a ranking at one hand-picked point is a choice made by whoever picked
the point -- the same objection this suite raises against choosing a signal-to-noise ratio instead of
calibrating to a ceiling.

A curve fixes it. Run the same bed, the same arms and the same seeds at a grid of row counts and report
where the two families cross. The crossing point is a finding that transfers: "below roughly this many
rows per informative column, prefer the parametric statistic" is advice a reader can act on, while "MI
lost at n=200" is a fact about one cell.

Everything except ``n`` is held fixed. Same structure, same coefficients, same calibrated ceiling, same
seeds, same arms -- so a difference along the curve is a difference in sample size and in nothing else.
The bed is rebuilt at each size rather than subsampled, because subsampling a calibrated bed changes its
achievable ceiling and the curve would then confound difficulty with size.

    python -m mlframe.feature_selection._benchmarks.fs_hybrid.sweep_recovery_vs_n --seeds 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["ROW_GRID", "BINNED_ARMS", "PARAMETRIC_ARMS", "SWEEP_ARMS", "run_point", "crossing_points", "summarize", "main"]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_results")

#: Geometric, spanning the range where the answer plausibly changes. The bottom is smaller than any bed in
#: the suite currently runs at; the top is where every method has enough data and the comparison stops
#: being informative, which is exactly why a single high-n point would have shown nothing.
ROW_GRID: Tuple[int, ...] = (150, 300, 600, 1200, 2500, 5000, 10000)

#: Arms that discretise before scoring. These are the ones the small-sample claim is ABOUT.
BINNED_ARMS: Tuple[str, ...] = ("univariate-mi", "skb-mi", "mrmr")

#: Arms built on a parametric statistic over the raw values, which is what binning is being compared to.
PARAMETRIC_ARMS: Tuple[str, ...] = ("skb-f", "select-fdr", "lars-order")

#: The controls travel with them: without `variance-sort` there is no way to tell a curve that is rising
#: because methods improve from one that is rising because the answer key got easier to hit by luck.
SWEEP_ARMS: Tuple[str, ...] = BINNED_ARMS + PARAMETRIC_ARMS + ("variance-sort",)

DEFAULT_BED = "linear_gaussian_lowdim_n200"


def run_point(bed: str, arm_name: str, n_samples: int, seed: int) -> Dict[str, Any]:
    """Fit one arm on one size of one bed and score what it recovered.

    Returns:
        A record with the recovered share of the answer key, the precision, and the selection size. A cell
        that fails records its error rather than disappearing, for the same reason the main runner does:
        the arms that fail are the ones whose curve would otherwise look best.
    """
    from ._arms import build_arm_roster
    from ._scm_beds import build_scm_bed

    frame, labels, truth = build_scm_bed(bed, seed=seed, n_samples=n_samples)
    answer = {str(column) for column in truth["base"]}
    roster = build_arm_roster(int(frame.shape[1]), k=len(answer), random_state=seed)
    if arm_name not in roster:
        raise KeyError(f"unknown arm {arm_name!r}; the roster has {sorted(roster)}")

    result = roster[arm_name]().run(frame, np.asarray(labels))
    names = [str(column) for column in frame.columns]
    selected = {names[index] for index, keep in enumerate(np.asarray(result.support, dtype=bool)) if keep}
    return {
        "bed": bed,
        "arm": arm_name,
        "n_samples": int(n_samples),
        "seed": int(seed),
        "n_selected": len(selected),
        "recall": len(selected & answer) / len(answer) if answer else float("nan"),
        "precision": len(selected & answer) / len(selected) if selected else 0.0,
    }


def _family_curve(rows: Sequence[Dict[str, Any]], arms: Sequence[str]) -> Dict[int, float]:
    """Return ``{n_samples: mean recall}`` pooled over a family's arms and seeds."""
    out: Dict[int, List[float]] = {}
    for row in rows:
        if str(row.get("arm")) in set(arms) and np.isfinite(float(row.get("recall", float("nan")))):
            out.setdefault(int(row["n_samples"]), []).append(float(row["recall"]))
    return {size: float(np.mean(values)) for size, values in sorted(out.items()) if values}


def crossing_points(rows: Sequence[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """Return the row-count bracket where the binned family overtakes the parametric one, if it does.

    Reported as a BRACKET rather than an interpolated point. The curve is measured on a coarse geometric
    grid and interpolating between two measured sizes would invent a precision the design does not have --
    the same objection this suite raises against every other point estimate it declines to make.
    """
    binned = _family_curve(rows, BINNED_ARMS)
    parametric = _family_curve(rows, PARAMETRIC_ARMS)
    shared = sorted(set(binned) & set(parametric))
    if len(shared) < 2:
        return None

    previous = binned[shared[0]] - parametric[shared[0]]
    for earlier, later in zip(shared, shared[1:]):
        current = binned[later] - parametric[later]
        if previous < 0.0 <= current:
            return earlier, later
        previous = current
    return None


def summarize(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Render the curve, the crossing bracket, and what a missing crossing would mean."""
    lines = ["", "=" * 100, "RECOVERY VERSUS SAMPLE SIZE: where a binned estimator overtakes a parametric one", "=" * 100, ""]
    binned = _family_curve(rows, BINNED_ARMS)
    parametric = _family_curve(rows, PARAMETRIC_ARMS)
    control = _family_curve(rows, ("variance-sort",))
    sizes = sorted(set(binned) | set(parametric))
    if not sizes:
        lines.append("no usable cells: every point failed, which is itself the result")
        return lines

    lines.append(f"{'n':>8}{'binned':>10}{'parametric':>13}{'gap':>9}{'control':>10}")
    for size in sizes:
        left, right = binned.get(size), parametric.get(size)
        gap = (left - right) if left is not None and right is not None else float("nan")
        lines.append(f"{size:>8}{left if left is not None else float('nan'):>10.3f}{right if right is not None else float('nan'):>13.3f}{gap:>9.3f}{control.get(size, float('nan')):>10.3f}")

    bracket = crossing_points(rows)
    lines.append("")
    if bracket is None:
        lines.append("No crossing inside the measured range. One family leads throughout, so the small-sample claim as")
        lines.append("stated does not hold on this bed: the difference is not about sample size at all.")
    else:
        lines.append(f"Crossing between n={bracket[0]} and n={bracket[1]}: below that bracket the parametric statistic leads,")
        lines.append("above it the binned one does. This bracket is the transferable finding; the ranking at any single n is not.")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run the sweep and print the curve."""
    parser = argparse.ArgumentParser(description="Recovery as a function of sample size, for binned versus parametric statistics")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bed", default=DEFAULT_BED)
    parser.add_argument("--sizes", default=",".join(str(size) for size in ROW_GRID))
    parser.add_argument("--out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    sizes = [int(token) for token in str(args.sizes).split(",") if token.strip()]
    rows: List[Dict[str, Any]] = []
    for n_samples in sizes:
        for arm_name in SWEEP_ARMS:
            for seed in range(int(args.seeds)):
                try:
                    rows.append(run_point(str(args.bed), arm_name, n_samples, seed))
                except Exception as exc:
                    logger.warning("n=%s arm=%s seed=%s failed: %s", n_samples, arm_name, seed, exc)
                    rows.append({"bed": str(args.bed), "arm": arm_name, "n_samples": n_samples, "seed": seed, "recall": float("nan"), "precision": float("nan"), "n_selected": 0, "error": f"{type(exc).__name__}: {exc}"})
            done = [r for r in rows if r["n_samples"] == n_samples and r["arm"] == arm_name]
            recalls = [float(r["recall"]) for r in done if np.isfinite(float(r["recall"]))]
            print(f"n={n_samples:<7} {arm_name:<16} recall={np.mean(recalls) if recalls else float('nan'):.3f} over {len(recalls)} seeds")

    out = args.out or os.path.join(RESULTS_DIR, f"sweep_recovery_vs_n_{args.bed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)

    for line in summarize(rows):
        print(line)
    print(f"\nwritten: {out}")
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if main() >= 0 else 1)
