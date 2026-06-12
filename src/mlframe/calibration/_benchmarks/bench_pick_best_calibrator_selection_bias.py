"""Quantify the in-sample selection bias of ``pick_best_calibrator``.

pick_best_calibrator (selection="same_oof", legacy) fits each candidate calibrator on the OOF
probabilities AND scores its ECE on the SAME OOF rows, then picks the lowest-ECE candidate. A flexible
calibrator (Isotonic) can drive its same-OOF ECE to ~0 by construction (isotonic regression
interpolates the training calibration), so the ``lowest_ece_ci_separated`` rule systematically prefers
it and reports an OPTIMISTIC ECE that does not hold on fresh data.

The shipped default ``selection="inner_cv"`` ranks each candidate by HELD-OUT ECE (fit on inner-train
folds, score on the held-out fold, averaged), then refits the chosen calibrator on the full OOF -- so
the reported ECE is honest (no longer ~0) and the chosen calibrator generalises. This bench reports the
fresh-holdout ECE of the OLD (same_oof) vs NEW (inner_cv) selections per scenario + seed, plus the
optimism gap, as the default-flip evidence. Run: ``python -m
mlframe.calibration._benchmarks.bench_pick_best_calibrator_selection_bias``.
"""
from __future__ import annotations

import json
import numpy as np

from mlframe.calibration.policy import pick_best_calibrator, _fit_calibrator, _ece_score


def _gen(rng, n, slope, noise):
    y = (rng.random(n) < 0.5).astype(int)
    p = np.clip(0.5 + (y - 0.5) * slope + rng.normal(0, noise, n), 0.005, 0.995)
    return p, y


def _holdout_ece(selection, oof_p, oof_y, ho_p, ho_y, seed):
    """Pick a calibrator with ``selection``, refit it on full OOF, return (chosen, reported, fresh-holdout ECE)."""
    res = pick_best_calibrator(None, None, oof_p, oof_y, n_bootstrap=200, random_state=seed, selection=selection)
    chosen = res["chosen"]
    cal = _fit_calibrator(chosen, oof_p, oof_y)
    holdout = float(_ece_score(ho_y, cal(ho_p))) if cal is not None else float("nan")
    return chosen, float(res["ece_mean"]), holdout


def run(seeds=range(8)):
    # ``small`` (n=300) is where flexible Isotonic over-fits the OOF: same_oof picks it on a ~0 in-sample
    # ECE that does not survive fresh data, while inner_cv's held-out ranking sees the gap and prefers a
    # smoother calibrator -- the generalisation win. Larger-n scenarios have enough rows that Isotonic
    # generalises fine, so the two selections agree there (no regression).
    scenarios = {
        "small_overfit": (300, 1.6, 0.25),
        "overconfident": (4000, 1.6, 0.15),
        "mild": (4000, 0.9, 0.2),
        "noisy": (4000, 0.7, 0.3),
    }
    rows = []
    for sc, (n, slope, noise) in scenarios.items():
        for seed in seeds:
            rng = np.random.default_rng(seed)
            oof_p, oof_y = _gen(rng, n, slope, noise)
            ho_p, ho_y = _gen(rng, n, slope, noise)
            old_chosen, old_rep, old_ho = _holdout_ece("same_oof", oof_p, oof_y, ho_p, ho_y, seed)
            new_chosen, new_rep, new_ho = _holdout_ece("inner_cv", oof_p, oof_y, ho_p, ho_y, seed)
            rows.append({"scenario": sc, "seed": int(seed),
                         "old_chosen": old_chosen, "old_reported_ece": old_rep, "old_holdout_ece": old_ho,
                         "new_chosen": new_chosen, "new_reported_ece": new_rep, "new_holdout_ece": new_ho,
                         "holdout_delta": new_ho - old_ho, "old_optimism": old_ho - old_rep})
    per_scenario = {}
    for sc in scenarios:
        srows = [r for r in rows if r["scenario"] == sc]
        per_scenario[sc] = {
            "old_mean_holdout": round(float(np.mean([r["old_holdout_ece"] for r in srows])), 6),
            "new_mean_holdout": round(float(np.mean([r["new_holdout_ece"] for r in srows])), 6),
            "new_wins_or_ties": int(sum(r["new_holdout_ece"] <= r["old_holdout_ece"] + 1e-9 for r in srows)),
            "n_seeds": len(srows),
        }
    scenarios_new_no_worse = sum(v["new_mean_holdout"] <= v["old_mean_holdout"] + 1e-9 for v in per_scenario.values())
    summary = {
        "per_scenario": per_scenario,
        "scenarios_new_no_worse": f"{scenarios_new_no_worse}/{len(per_scenario)}",
        "old_mean_holdout_all": round(float(np.mean([r["old_holdout_ece"] for r in rows])), 6),
        "new_mean_holdout_all": round(float(np.mean([r["new_holdout_ece"] for r in rows])), 6),
        "old_mean_reported_all": round(float(np.mean([r["old_reported_ece"] for r in rows])), 6),
        "new_mean_reported_all": round(float(np.mean([r["new_reported_ece"] for r in rows])), 6),
        "old_mean_optimism_all": round(float(np.mean([r["old_optimism"] for r in rows])), 6),
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    out = run()
    print(json.dumps(out["summary"], indent=2))
