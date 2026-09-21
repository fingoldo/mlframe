"""Is "RFECV" one method, or a family of twenty-four? Pre-registration hypothesis 7.

RFECV does not observe feature importance once. It observes it per fold and then AGGREGATES, and both
halves of that aggregation are configurable: eight voting rules (``VotesAggregation``) decide how per-fold
rankings combine, and three policies (``fi_missing_policy``) decide what a feature absent from one fold's
importance table counts as. Every cell in this benchmark ran on one arbitrary point of that 24-point grid,
the library default, and reported the result under the name of the method.

That is only legitimate if the grid barely matters. This ablation measures whether it does, and the two
outcomes carry opposite consequences:

* **the spread is inside the noise band** -- the aggregation is a detail, the default is as good a
  representative as any, and every RFECV row in the atlas stands as written.
* **the spread exceeds the noise band** -- then "RFECV recovered 0.567 of the blanket" is not a property of
  RFECV. It is a property of one configuration, and the atlas owes the reader the range. A method whose
  internal knob moves the result more than the difference between methods is a family being reported as a
  point.

The comparison is paired by ``(bed, seed)``: every configuration sees the identical data, so the
difference between two configurations carries no data-draw variance at all. The noise band is taken from
the same run rather than assumed -- the spread across seeds WITHIN one configuration is what a difference
has to beat.

No new selector code exists for this: all twenty-four configurations are already implemented and reachable
today. The only thing that was missing was somebody running them.

    python -m mlframe.feature_selection._benchmarks.fs_hybrid.ablation_rfecv_aggregation --seeds 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["VOTING_RULES", "MISSING_POLICIES", "configurations", "run_one", "summarize", "main"]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_results")

#: Every rule the enum offers, not a chosen subset: choosing a subset after seeing which ones behave well
#: is the exact move this ablation exists to make impossible.
VOTING_RULES: Tuple[str, ...] = ("Minimax", "OG", "Borda", "Plurality", "Dowdall", "Copeland", "AM", "GM")

#: What a feature missing from one fold's importance table is worth. `worst` is the library default.
MISSING_POLICIES: Tuple[str, ...] = ("worst", "median", "skip")

#: Beds chosen for what they can DISCRIMINATE, before any of them was run: one where RFECV already scores
#: below every marginal filter, one where the answer key is a single column, one flat control where the
#: aggregation should not matter at all. A control that also moves means the spread is not about structure.
DEFAULT_BEDS: Tuple[str, ...] = ("mb_spouse_collider", "mediator_chain_with_proxy", "linear_k5_p50")


def configurations() -> List[Tuple[str, str]]:
    """Return the full 24-point grid as ``(voting_rule, missing_policy)`` pairs."""
    return [(rule, policy) for rule, policy in itertools.product(VOTING_RULES, MISSING_POLICIES)]


def run_one(bed: str, seed: int, rule: str, policy: str, rows: int) -> Dict[str, Any]:
    """Fit RFECV under one configuration on one seed of one bed and score what it recovered.

    Returns:
        A record carrying the configuration, the recovered set's precision and recall against the bed's
        declared answer key, and the size of the selection. Selection size is kept because a configuration
        that recovers the same fraction while selecting twice as many columns has not done the same thing.
    """
    import lightgbm as lgb
    import pandas as pd

    from mlframe.feature_selection import extract_selected
    from mlframe.feature_selection.wrappers import RFECV, FIConfig, SearchConfig

    from ._scm_beds import build_scm_bed

    frame, labels, truth = build_scm_bed(bed, seed=seed, n_samples=rows)
    model = RFECV(
        estimator=lgb.LGBMClassifier(n_estimators=80, verbose=-1, n_jobs=-1, random_state=seed),
        cv=3,
        scoring=None,
        verbose=0,
        fi_config=FIConfig(importance_getter="auto", n_features_selection_rule="one_se_min", votes_aggregation_method=rule, fi_missing_policy=policy),
        search_config=SearchConfig(max_refits=12, max_runtime_mins=2.0),
        random_state=seed,
    )
    model.fit(frame, pd.Series(np.asarray(labels)))

    names = [str(column) for column in frame.columns]
    selected = {str(column) for column in extract_selected(model, names) if str(column) in set(names)}
    answer = {str(column) for column in truth["base"]}
    return {
        "bed": bed,
        "seed": seed,
        "voting_rule": rule,
        "missing_policy": policy,
        "n_selected": len(selected),
        "recall": len(selected & answer) / len(answer) if answer else float("nan"),
        "precision": len(selected & answer) / len(selected) if selected else 0.0,
        "selected": sorted(selected),
    }


def summarize(rows: Sequence[Dict[str, Any]]) -> List[str]:
    """Render the verdict: the spread across configurations against the spread across seeds.

    The comparison that matters is not "do the configurations differ" -- with enough seeds everything
    differs -- but whether they differ by MORE than the same bed differs from itself across data draws. A
    configuration grid whose spread sits inside the seed-to-seed spread is a detail; one that exceeds it is
    a hidden degree of freedom in every result the method appears in.
    """
    lines = ["", "=" * 100, "RFECV AGGREGATION GRID: 8 voting rules x 3 missing-value policies", "=" * 100]
    beds = sorted({str(row["bed"]) for row in rows})
    for bed in beds:
        here = [row for row in rows if row["bed"] == bed]
        if not here:
            continue
        per_config: Dict[Tuple[str, str], List[float]] = {}
        for row in here:
            per_config.setdefault((str(row["voting_rule"]), str(row["missing_policy"])), []).append(float(row["recall"]))
        means = {config: float(np.mean(values)) for config, values in per_config.items() if values}
        if not means:
            continue
        # The within-configuration seed spread, pooled: what a difference between configurations must beat
        # before it is worth calling a difference at all.
        within = [float(np.std(values, ddof=1)) for values in per_config.values() if len(values) > 1]
        noise = float(np.mean(within)) if within else float("nan")
        best = max(means.items(), key=lambda item: item[1])
        worst = min(means.items(), key=lambda item: item[1])
        spread = best[1] - worst[1]
        verdict = "INSIDE the seed-to-seed noise: the default is representative" if np.isfinite(noise) and spread <= noise else "EXCEEDS the seed-to-seed noise: RFECV is a FAMILY here, not a method"
        lines.append("")
        lines.append(f"{bed}: recall spans {worst[1]:.3f} ({worst[0][0]}/{worst[0][1]}) to {best[1]:.3f} ({best[0][0]}/{best[0][1]})")
        lines.append(f"  spread {spread:.3f} vs per-seed noise {noise:.3f} -- {verdict}")
        default = means.get(("Borda", "worst"))
        if default is not None:
            lines.append(f"  the library default (Borda/worst), which every headline cell used: {default:.3f}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Run the grid over the chosen beds and seeds, write the records, print the verdict."""
    parser = argparse.ArgumentParser(description="Does RFECV's per-fold aggregation choice move its result?")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--beds", default=",".join(DEFAULT_BEDS))
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--out", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    beds = [name.strip() for name in str(args.beds).split(",") if name.strip()]
    rows: List[Dict[str, Any]] = []
    for bed in beds:
        for rule, policy in configurations():
            for seed in range(int(args.seeds)):
                try:
                    record = run_one(bed, seed, rule, policy, int(args.rows))
                except Exception as exc:  # a configuration that cannot run is a finding, not a gap in the table
                    logger.warning("bed=%s rule=%s policy=%s seed=%s failed: %s", bed, rule, policy, seed, exc)
                    record = {"bed": bed, "seed": seed, "voting_rule": rule, "missing_policy": policy, "error": f"{type(exc).__name__}: {exc}", "recall": float("nan"), "precision": float("nan"), "n_selected": 0}
                rows.append(record)
            done = [r for r in rows if r["bed"] == bed and r["voting_rule"] == rule and r["missing_policy"] == policy]
            recalls = [float(r["recall"]) for r in done if np.isfinite(float(r["recall"]))]
            print(f"{bed:<28} {rule:<10} {policy:<7} recall={np.mean(recalls) if recalls else float('nan'):.3f} n={len(done)}")

    out = args.out or os.path.join(RESULTS_DIR, "ablation_rfecv_aggregation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)

    for line in summarize(rows):
        print(line)
    print(f"\nwritten: {out}")
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(0 if main() >= 0 else 1)
