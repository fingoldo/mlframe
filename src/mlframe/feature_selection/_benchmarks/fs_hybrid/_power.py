"""Power analysis for the paired design, estimated from cells that already exist rather than from a guess.

The pre-registration fixes ``R = 20`` dataset seeds per (arm, scenario) and owes a justification. The honest
way to produce one is the quantity the design actually depends on: ``tau``, the standard deviation of the
per-``dataset_seed`` paired difference ``arm - all-features`` inside one scenario. Everything else follows,
because the headline test is a paired ``t`` on ``m`` such differences with ``SE = tau / sqrt(m)``.

``tau`` is not assumed here. Phase 0 wrote thousands of cells over 20 seeds, so every (scenario, arm, model,
K) combination supplies a direct estimate, and the planning number is a quantile over those estimates rather
than a single pilot cell that happened to be quiet. This matters because ``tau`` is strongly heterogeneous
across beds -- a wide bed with an unstable selector moves by an order of magnitude more per seed than a
narrow one -- so a design sized on the median is underpowered on exactly the cells where the question is
hardest.

Two directions are reported, and they answer different questions:

* ``required_replicates(tau, delta)`` -- how many seeds detect a difference of ``delta`` at 80% power. This
  is the planning direction, used before spending compute.
* ``detectable_effect(tau, m)`` -- the smallest difference ``m`` seeds can resolve. This is the interpretive
  direction, and it is the one that makes a null result readable: "no arm beat the null" means something
  different when the design could only have seen a 0.05 gap than when it could have seen 0.005.

The estimate is conditional on the beds and arms that produced it and does not transfer to a different grid;
``tau`` is a property of the scenario population, which is hand-picked here by construction.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._leaderboard import NULL_ARM, extract_long_rows
from ._paired_stats import average_over_cv_seed, paired_differences

logger = logging.getLogger(__name__)

__all__ = [
    "TauEstimate",
    "required_replicates",
    "detectable_effect",
    "achieved_power",
    "tau_estimates",
    "power_report",
]

# Planning defaults. Two-sided, conventional 80% power, and the effect grid spans the range the
# pre-registration cares about: 0.002 is below any practical relevance, 0.02 is a difference a practitioner
# would act on without statistics.
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
DEFAULT_EFFECT_GRID: Tuple[float, ...] = (0.002, 0.005, 0.010, 0.020, 0.050)
MAX_REPLICATES = 100000


@dataclass(frozen=True)
class TauEstimate:
    """The paired-difference spread for one (scenario, arm) contrast against the null hypothesis."""

    scenario: str
    arm: str
    m: int
    mean_delta: float
    tau: float


def _z(probability: float) -> float:
    """Standard-normal quantile, imported locally so the module stays cheap to import."""
    from scipy import stats as _stats

    return float(_stats.norm.ppf(probability))


def required_replicates(tau: float, delta: float, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER) -> Optional[int]:
    """Return the paired seeds needed to detect ``delta`` at ``power``, or ``None`` when ``tau`` is unusable.

    The normal formula ``m = (z_{1-a/2} + z_power)^2 (tau/delta)^2`` understates the requirement at small
    ``m``, where the critical value comes from a ``t`` with ``m-1`` degrees of freedom rather than a normal.
    The fixed-point loop below re-solves with the ``t`` quantile until ``m`` stops moving, which typically
    adds one to three seeds in the range this design lives in.
    """
    if not np.isfinite(tau) or tau < 0.0 or not np.isfinite(delta) or delta <= 0.0:
        return None
    if tau == 0.0:
        # Every seed moved by the same amount: two seeds establish the direction and no more are useful.
        return 2

    from scipy import stats as _stats

    ratio = float(tau) / float(delta)
    m = max(2, math.ceil((_z(1.0 - alpha / 2.0) + _z(power)) ** 2 * ratio**2))
    for _ in range(50):
        crit = float(_stats.t.ppf(1.0 - alpha / 2.0, max(1, m - 1)))
        nxt = max(2, math.ceil((crit + _z(power)) ** 2 * ratio**2))
        if nxt == m:
            return m
        if nxt > MAX_REPLICATES:
            return None
        m = nxt
    return m


def detectable_effect(tau: float, m: int, alpha: float = DEFAULT_ALPHA, power: float = DEFAULT_POWER) -> Optional[float]:
    """Return the smallest paired difference ``m`` seeds resolve at ``power``, or ``None`` when undefined."""
    if not np.isfinite(tau) or tau < 0.0 or m < 2:
        return None
    from scipy import stats as _stats

    crit = float(_stats.t.ppf(1.0 - alpha / 2.0, m - 1))
    return float((crit + _z(power)) * tau / math.sqrt(m))


def achieved_power(tau: float, delta: float, m: int, alpha: float = DEFAULT_ALPHA) -> Optional[float]:
    """Return the power ``m`` seeds have against a true difference of ``delta``, or ``None`` when undefined."""
    if not np.isfinite(tau) or tau <= 0.0 or m < 2 or not np.isfinite(delta) or delta <= 0.0:
        return None
    from scipy import stats as _stats

    df = m - 1
    crit = float(_stats.t.ppf(1.0 - alpha / 2.0, df))
    ncp = float(delta) * math.sqrt(m) / float(tau)
    # Non-central t survival at the two-sided critical value; the lower tail is negligible for ncp > 0 but is
    # kept so the number stays correct for effects small enough that both tails contribute.
    upper = float(_stats.nct.sf(crit, df, ncp))
    lower = float(_stats.nct.cdf(-crit, df, ncp))
    return upper + lower


def tau_estimates(
    records: Iterable[Dict[str, Any]],
    model: str,
    k_label: str,
    metric: str = "roc_auc",
    null_arm: str = NULL_ARM,
    min_seeds: int = 3,
) -> List[TauEstimate]:
    """Estimate ``tau`` for every (scenario, arm) contrast available in ``records`` at one (model, K, metric)."""
    rows = average_over_cv_seed(extract_long_rows(records, model=model, k_label=k_label, metric=metric))
    scenarios = sorted({str(r["scenario"]) for r in rows})
    arms = sorted({str(r["arm"]) for r in rows if str(r["arm"]) != null_arm})

    out: List[TauEstimate] = []
    for scenario in scenarios:
        for arm in arms:
            deltas = paired_differences(rows, arm=arm, null_arm=null_arm, scenario=scenario)
            if len(deltas) < min_seeds:
                continue
            arr = np.asarray(deltas, dtype=np.float64)
            out.append(
                TauEstimate(
                    scenario=scenario,
                    arm=arm,
                    m=int(arr.size),
                    mean_delta=float(arr.mean()),
                    tau=float(arr.std(ddof=1)),
                )
            )
    return out


def _quantile_block(taus: Sequence[float], declared_r: int, effects: Sequence[float]) -> List[str]:
    """Render the planning table for the three planning quantiles of the observed ``tau`` distribution."""
    arr = np.asarray([t for t in taus if np.isfinite(t)], dtype=np.float64)
    lines = [
        "",
        (f"Observed tau over {arr.size} (scenario, arm) contrasts: "
        f"median {np.median(arr):.5f}, p75 {np.quantile(arr, 0.75):.5f}, p90 {np.quantile(arr, 0.90):.5f}, "
        f"max {arr.max():.5f}"),
        "",
        "| planning tau | " + " | ".join(f"R for d={d:g}" for d in effects) + f" | detectable at R={declared_r} |",
        "|---|" + "---|" * (len(effects) + 1),
    ]
    quantiles = (("median", float(np.median(arr))), ("p75", float(np.quantile(arr, 0.75))), ("p90", float(np.quantile(arr, 0.90))))
    for label, tau in quantiles:
        cells = []
        for effect in effects:
            need = required_replicates(tau, effect)
            cells.append("infeasible" if need is None else str(need))
        mde = detectable_effect(tau, declared_r)
        lines.append(f"| {label} = {tau:.5f} | " + " | ".join(cells) + " | " + ("n/a" if mde is None else f"{mde:.5f}") + " |")
    return lines


def power_report(
    records: Sequence[Dict[str, Any]],
    models: Sequence[str],
    k_labels: Sequence[str],
    metric: str = "roc_auc",
    declared_r: int = 20,
    effects: Sequence[float] = DEFAULT_EFFECT_GRID,
) -> str:
    """Return a markdown power report: observed ``tau``, seeds each effect needs, and the MDE at ``declared_r``."""
    lines = [
        "# Power analysis (estimated from executed cells, not from a pilot assumption)",
        "",
        f"Paired design, two-sided alpha={DEFAULT_ALPHA}, power={DEFAULT_POWER:.0%}, metric={metric}.",
        f"`tau` is the sd of the per-dataset_seed difference `arm - {NULL_ARM}` within one scenario.",
        "Quantiles are taken over (scenario, arm) contrasts, so p90 sizes the design for its harder cells",
        "rather than its quiet ones.",
    ]
    for model in models:
        for k_label in k_labels:
            ests = tau_estimates(records, model=model, k_label=k_label, metric=metric)
            if not ests:
                lines += ["", f"## {model} @ {k_label}", "", "no contrast reached the minimum seed count"]
                continue
            lines += ["", f"## {model} @ {k_label}"]
            lines += _quantile_block([e.tau for e in ests], declared_r=declared_r, effects=effects)
            worst = sorted(ests, key=lambda e: -e.tau)[:5]
            lines += ["", "Noisiest contrasts (these set the p90, and are where a null result is least informative):", ""]
            lines += [f"- `{e.scenario}` / `{e.arm}`: tau={e.tau:.5f}, mean delta={e.mean_delta:+.5f}, m={e.m}" for e in worst]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Read a results JSONL and print the power report."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Power analysis over executed benchmark cells")
    parser.add_argument("results", help="path to a results JSONL")
    parser.add_argument("--models", default="lightgbm,logistic")
    parser.add_argument("--k-labels", default="", help="comma-separated K labels; empty means every label present")
    parser.add_argument("--metric", default="roc_auc")
    parser.add_argument("--declared-r", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with open(args.results, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    k_labels = [k.strip() for k in args.k_labels.split(",") if k.strip()]
    if not k_labels:
        present = sorted({str(k) for rec in records for k in (rec.get("scores") or {})})
        k_labels = [k for k in present if k != "self"] or present
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(power_report(records, models=models, k_labels=k_labels, metric=args.metric, declared_r=args.declared_r))


if __name__ == "__main__":
    main()
