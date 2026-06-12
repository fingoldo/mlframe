"""Quantify the in-sample selection bias of ``pick_best_calibrator``.

pick_best_calibrator fits each candidate calibrator on the OOF probabilities AND scores its ECE on
the SAME OOF rows, then picks the lowest-ECE candidate. A flexible calibrator (Isotonic) can drive
its same-OOF ECE to ~0 by construction (isotonic regression interpolates the training calibration),
so the ``lowest_ece_ci_separated`` rule systematically prefers it and reports an OPTIMISTIC ECE that
does not hold on fresh data. This bench measures the optimism gap (fresh-holdout ECE minus the
reported same-OOF ECE) and the selection distribution across scenarios + seeds, as the evidence for a
FUTURE inner-CV (or train/score-split) selection fix. Run: ``python -m
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


def run(n=4000, seeds=range(8)):
    scenarios = {"overconfident": (1.6, 0.15), "mild": (0.9, 0.2), "noisy": (0.7, 0.3)}
    rows = []
    chosen_counts: dict[str, int] = {}
    for sc, (slope, noise) in scenarios.items():
        for seed in seeds:
            rng = np.random.default_rng(seed)
            oof_p, oof_y = _gen(rng, n, slope, noise)
            ho_p, ho_y = _gen(rng, n, slope, noise)
            res = pick_best_calibrator(None, None, oof_p, oof_y, n_bootstrap=200, random_state=seed)
            chosen = res["chosen"]
            reported = float(res["ece_mean"])
            cal = _fit_calibrator(chosen, oof_p, oof_y)
            holdout = float(_ece_score(ho_y, cal(ho_p))) if cal is not None else float("nan")
            chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1
            rows.append({"scenario": sc, "seed": int(seed), "chosen": chosen,
                         "reported_oof_ece": reported, "holdout_ece": holdout,
                         "optimism": holdout - reported})
    iso = [r for r in rows if r["chosen"] == "Isotonic"]
    summary = {
        "chosen_counts": chosen_counts,
        "isotonic_share": round(len(iso) / max(len(rows), 1), 3),
        "mean_optimism_all": round(float(np.mean([r["optimism"] for r in rows])), 5),
        "mean_optimism_isotonic": round(float(np.mean([r["optimism"] for r in iso])), 5) if iso else None,
        "mean_reported_isotonic": round(float(np.mean([r["reported_oof_ece"] for r in iso])), 6) if iso else None,
        "mean_holdout_isotonic": round(float(np.mean([r["holdout_ece"] for r in iso])), 6) if iso else None,
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    out = run()
    print(json.dumps(out["summary"], indent=2))
