"""Committed re-verification bench for BorutaShap ``early_stop_tentative`` (margin-gated adaptive trial-stop).

Claim under test: the margin-gated stop reclaims large wall-time at DECISION-EQUIVALENCE -- the CONFIRMED (accepted)
set is identical to running the full ``n_trials`` cap, while skipping the trailing trials that only re-test a residual
tentative tail that never resolves.

Design: 5 synthetic scenarios x 3 seeds = 15 cells. Each cell fits BorutaShap twice (cap-run vs early-stop-run) with
an identical RandomForest surrogate + identical random_state, and records:
  - trials run by each (the per-trial model fit + importance pass is the dominant cost, so trials-saved ~ wall-saved),
  - wall time of each,
  - Jaccard of the ACCEPTED set (the load-bearing "confirmed" output) -- the flip criterion,
  - Jaccard of the rejected set and of selected_features_.

Flip rule (mlframe CLAUDE.md "variant defaults: accurate-first, speed breaks ties"): flip the default to ON only if
the accepted-set Jaccard is >= ~0.95 across the MAJORITY of cells (here we require the mean and the per-scenario means
to clear the bar). The accepted set is the decision that matters; a near-identical rejected set is the documented slack
(a feature that would reject between the stop trial and the cap).

Run (host-safe, GPU off):
  CUDA_VISIBLE_DEVICES="" MLFRAME_NO_CUDA_AUTOCONFIG=1 MLFRAME_KEEP_BROKEN_CUPY=1 \
      python src/mlframe/feature_selection/_benchmarks/bench_boruta_early_stop_tentative.py

Output: prints a per-cell + per-scenario table and writes _results/boruta_early_stop_tentative.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[3]))  # repo/src so ``import mlframe`` resolves when run as a script

N_TRIALS_CAP = 80
PATIENCE = 20
MARGIN = 0.15
SEEDS = (0, 1, 2)


def _base(n, seed, *, n_inf, n_red, n_noise, interaction=True, quadratic=True):
    """Strong linear signal + redundant correlated copies + (optional) interaction/quadratic operands with ~zero
    marginal signal + pure noise. The marginal-less operands park near hit-rate 0.5 -> a residual tentative tail the
    all-decided stop never clears, which is exactly what the margin-gated stop is meant to reclaim."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, n_inf))
    logit = np.zeros(n)
    for i in range(n_inf):
        logit += (1.4 - 0.1 * i) * z[:, i]
    cols = {f"inf_{i}": z[:, i] for i in range(n_inf)}
    if interaction and n_inf >= 2:
        logit += 1.6 * z[:, 0] * z[:, 1]
    if quadratic and n_inf >= 3:
        logit += 1.2 * (z[:, 2] ** 2 - 1.0)
    logit /= 1.6
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    for parent in range(min(n_red, n_inf)):
        for j in range(3):
            cols[f"red_{parent}_{j}"] = z[:, parent] + 0.30 * rng.standard_normal(n)
    for i in range(n_noise):
        cols[f"noise_{i}"] = rng.standard_normal(n)
    X = pd.DataFrame(cols)
    order = list(X.columns)
    rng.shuffle(order)
    return X[order], pd.Series(y, name="target")


SCENARIOS = {
    "linear_redundant": dict(n=3000, n_inf=6, n_red=3, n_noise=24, interaction=False, quadratic=False),
    "interaction_tail": dict(n=3000, n_inf=8, n_red=3, n_noise=28, interaction=True, quadratic=True),
    "wide_noise": dict(n=2500, n_inf=5, n_red=2, n_noise=45, interaction=True, quadratic=False),
    "few_relevant": dict(n=3500, n_inf=4, n_red=1, n_noise=20, interaction=True, quadratic=True),
    "dense_signal": dict(n=2500, n_inf=10, n_red=4, n_noise=18, interaction=True, quadratic=True),
}


def _jac(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def _selector(**kw):
    from sklearn.ensemble import RandomForestClassifier
    from mlframe.feature_selection.boruta_shap import BorutaShap

    base = dict(
        model=RandomForestClassifier(n_estimators=60, n_jobs=4, random_state=0),
        importance_measure="gini", classification=True, n_trials=N_TRIALS_CAP, percentile=95,
        pvalue=0.05, verbose=False, random_state=0,
    )
    base.update(kw)
    return BorutaShap(**base)


def main():
    cells = []
    for scen, kw in SCENARIOS.items():
        for seed in SEEDS:
            X, y = _base(seed=seed, **kw)

            off = _selector(early_stop_tentative=False)
            t0 = time.perf_counter()
            off.fit(X, y)
            off_wall = time.perf_counter() - t0

            on = _selector(early_stop_tentative=True, early_stop_patience=PATIENCE, early_stop_margin=MARGIN)
            t0 = time.perf_counter()
            on.fit(X, y)
            on_wall = time.perf_counter() - t0

            cell = dict(
                scenario=scen, seed=seed,
                off_trials=int(off.n_trials_run_), on_trials=int(on.n_trials_run_),
                off_wall=round(off_wall, 3), on_wall=round(on_wall, 3),
                wall_saved_pct=round(100.0 * (off_wall - on_wall) / off_wall, 1) if off_wall else 0.0,
                jaccard_accepted=round(_jac(on.accepted, off.accepted), 4),
                jaccard_rejected=round(_jac(on.rejected, off.rejected), 4),
                jaccard_selected=round(_jac(on.selected_features_, off.selected_features_), 4),
                off_n_tentative=len(off.tentative),
            )
            cells.append(cell)
            print(
                f"{scen:<18} s{seed} | trials {cell['off_trials']:>3}->{cell['on_trials']:>3} "
                f"wall {cell['off_wall']:>6}->{cell['on_wall']:>6}s ({cell['wall_saved_pct']:>5}% saved) | "
                f"J(acc)={cell['jaccard_accepted']:.3f} J(rej)={cell['jaccard_rejected']:.3f} "
                f"tail={cell['off_n_tentative']}"
            )

    df = pd.DataFrame(cells)
    print("\n=== per-scenario means ===")
    per_scen = df.groupby("scenario").agg(
        wall_saved_pct=("wall_saved_pct", "mean"),
        jaccard_accepted=("jaccard_accepted", "mean"),
        jaccard_rejected=("jaccard_rejected", "mean"),
        off_n_tentative=("off_n_tentative", "mean"),
    ).round(3)
    print(per_scen.to_string())

    mean_wall = float(df["wall_saved_pct"].mean())
    mean_jacc_acc = float(df["jaccard_accepted"].mean())
    min_scen_jacc_acc = float(per_scen["jaccard_accepted"].min())
    n_cells_acc_equiv = int((df["jaccard_accepted"] >= 0.95).sum())
    verdict_flip = (mean_jacc_acc >= 0.95) and (min_scen_jacc_acc >= 0.95) and (n_cells_acc_equiv >= len(df) * 0.5)

    summary = dict(
        n_cells=len(df), n_trials_cap=N_TRIALS_CAP, patience=PATIENCE, margin=MARGIN,
        mean_wall_saved_pct=round(mean_wall, 2),
        mean_jaccard_accepted=round(mean_jacc_acc, 4),
        min_scenario_jaccard_accepted=round(min_scen_jacc_acc, 4),
        n_cells_accepted_equiv_ge_095=n_cells_acc_equiv,
        verdict_flip_default_on=bool(verdict_flip),
    )
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))

    out = _HERE / "_results" / "boruta_early_stop_tentative.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(summary=summary, cells=cells), indent=2, sort_keys=True))
    print(f"\nwrote {out}")
    return summary


if __name__ == "__main__":
    main()
