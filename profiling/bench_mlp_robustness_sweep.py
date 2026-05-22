"""Empirical search for MLP HPT overrides that close the Ridge-vs-MLP gap under feature drift.

**SUPERSEDED by** ``bench_mlp_robustness_sweep_nonlinear.py`` (the multi-
DGP / multi-metric variant). This script only sweeps a LINEAR DGP and
only ranks by R^2; the production override in
``ROBUST_MLP_OVERRIDES_UNDER_DRIFT`` is now drawn from the nonlinear
variant, which agrees across R^2 + RMSE/y_std + MAE/y_std on a
different alpha (1e-4 vs this script's 0.1). This file is kept for
historical reproducibility of the original grounding and as the
fastest way to re-bench the linear-DGP behaviour in isolation.

Companion to ``bench_drift_fi_vs_model_harm.py``. That study showed
``weighted_drift_score`` predicts MLP catastrophic harm (Pearson r=+0.834,
zero false positives at threshold=3.0). The 2026-05-22 redirect from
"skip neural" to "fix neural HPT":

So instead of dropping MLP, we search the HPT space for overrides that
narrow ``MLP_excess_harm = Ridge_R^2_test - MLP_R^2_test`` on the same
drift scenarios. Phase 1 (this script) sweeps the Cartesian
{alpha} x {hidden_layer_sizes} x {activation}; Phase 2 validates the
top winners on low-drift / no-drift cells to confirm they don't degrade
the no-drift baseline.

Mechanism hypothesis
--------------------
MLP failure mode under drift: the network overfits the train-input
distribution; for a test input shifted beyond train range, the activations
fire in regions never optimised, producing arbitrary outputs. Ridge wins
trivially via linear extrapolation. The candidate fixes are anything that
pushes MLP toward linear behaviour: higher L2 (alpha), smaller capacity
(hidden), bounded activation (tanh) or a literal linear head (identity).

Experimental design
-------------------
- Same synthetic regression as bench_drift_fi_vs_model_harm.py: K=5,
  alphas=[10, 0.1, 0.1, 0.1, 0.1], y = alphas . x + noise.
- Phase 1: drift_z=10 on the dominant feature (where harm is largest),
  48 configs (4 alpha x 4 hidden x 3 activation), 20 seeds per config.
- Phase 2: top-5 from phase 1 + baseline, validated at drift_z in
  {0.0, 2.0, 10.0, 20.0}, both dominant + noise drift targets, 10 seeds
  per cell. Confirms the winner doesn't sacrifice the no-drift baseline.

Outputs
-------
- ``profiling/_results/bench_mlp_robustness_sweep_<stamp>.csv``
- Phase-1 leaderboard sorted by mean MLP_excess_harm (lower = better).
- Phase-2 validation table for the top configs across drift_z grid.
- Top winning override dict for direct paste into
  ``ROBUST_MLP_OVERRIDES_UNDER_DRIFT`` in
  ``src/mlframe/training/feature_drift_report.py``.

Run::

    python -m mlframe.profiling.bench_mlp_robustness_sweep
"""
from __future__ import annotations

import csv
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


N_FEATURES: int = 5
ALPHAS_DOMINANT = np.array([10.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
N_TRAIN: int = 2000
N_TEST: int = 500
NOISE_STD: float = 1.0
MLP_MAX_ITER: int = 200

ALPHA_GRID: tuple[float, ...] = (1e-4, 1e-1, 1.0, 10.0)
HIDDEN_GRID: tuple[tuple[int, ...], ...] = ((100,), (32, 16), (16,), (8,))
ACTIVATION_GRID: tuple[str, ...] = ("relu", "tanh", "identity")

# Phase 1: where the harm is. drift_z=10 on the dominant feature is the
# most-different-from-train regime the prior bench surfaced. Plenty of
# headroom to discriminate good vs bad overrides without flooring R^2.
PHASE1_DRIFT_Z: float = 10.0
PHASE1_DRIFT_TARGET: str = "dominant"
PHASE1_N_SEEDS: int = 20

# Phase 2: validate that the winner doesn't degrade the no-drift baseline
# or break the noise-drift case (where weighted score stays below threshold).
PHASE2_DRIFT_Z_LEVELS: Sequence[float] = (0.0, 2.0, 10.0, 20.0)
PHASE2_DRIFT_TARGETS = ("dominant", "noise")
PHASE2_N_SEEDS: int = 10
PHASE2_TOP_K: int = 5


def _build_trial(rng: np.random.Generator, drift_target: str, drift_z: float):
    alphas = ALPHAS_DOMINANT.copy()
    X_train = rng.normal(0.0, 1.0, (N_TRAIN, N_FEATURES))
    y_train = X_train @ alphas + rng.normal(0.0, NOISE_STD, N_TRAIN)
    X_test = rng.normal(0.0, 1.0, (N_TEST, N_FEATURES))
    if drift_target == "dominant":
        X_test[:, 0] += drift_z
    elif drift_target == "noise":
        X_test[:, 1] += drift_z
    y_test = X_test @ alphas + rng.normal(0.0, NOISE_STD, N_TEST)
    return X_train, y_train, X_test, y_test


def _run_one_trial(
    rng: np.random.Generator,
    drift_target: str,
    drift_z: float,
    alpha: float,
    hidden: tuple[int, ...],
    activation: str,
) -> dict:
    X_train, y_train, X_test, y_test = _build_trial(rng, drift_target, drift_z)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
    ridge_r2 = float(r2_score(y_test, ridge.predict(X_test_s)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        mlp = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation=activation,
            alpha=alpha,
            max_iter=MLP_MAX_ITER,
            random_state=int(rng.integers(0, 1_000_000)),
            early_stopping=True,
            n_iter_no_change=20,
        ).fit(X_train_s, y_train)
    mlp_r2 = float(r2_score(y_test, mlp.predict(X_test_s)))

    return {
        "drift_target": drift_target,
        "drift_z": float(drift_z),
        "alpha": alpha,
        "hidden": str(hidden),
        "activation": activation,
        "ridge_test_r2": ridge_r2,
        "mlp_test_r2": mlp_r2,
        "mlp_excess_harm": ridge_r2 - mlp_r2,
    }


def _config_key(alpha: float, hidden: tuple[int, ...], activation: str) -> str:
    return f"alpha={alpha:g} hidden={hidden} activation={activation}"


def _run_phase1() -> tuple[list[dict], list[tuple[str, float, dict]]]:
    """Phase 1: full sweep at the harm regime. Returns (rows, leaderboard).

    Leaderboard entries: (config_key, mean_mlp_excess_harm, config_dict).
    Sorted ascending (lower harm = better)."""
    print()
    print("# Phase 1: full MLP HPT sweep at drift_z=%.1f on '%s'" % (
        PHASE1_DRIFT_Z, PHASE1_DRIFT_TARGET))
    n_configs = len(ALPHA_GRID) * len(HIDDEN_GRID) * len(ACTIVATION_GRID)
    print("#   %d configs x %d seeds = %d trials" % (
        n_configs, PHASE1_N_SEEDS, n_configs * PHASE1_N_SEEDS))
    print()

    rows: list[dict] = []
    summaries: list[tuple[str, float, dict]] = []
    t0 = time.perf_counter()
    for i, (alpha, hidden, activation) in enumerate(
            product(ALPHA_GRID, HIDDEN_GRID, ACTIVATION_GRID), start=1):
        harms = []
        for seed in range(PHASE1_N_SEEDS):
            rng = np.random.default_rng(seed * 31 + i)
            row = _run_one_trial(
                rng, PHASE1_DRIFT_TARGET, PHASE1_DRIFT_Z,
                alpha, hidden, activation,
            )
            row["seed"] = seed
            row["phase"] = "1"
            rows.append(row)
            harms.append(row["mlp_excess_harm"])
        mean_harm = float(np.mean(harms))
        cfg = {"alpha": alpha, "hidden_layer_sizes": hidden, "activation": activation}
        summaries.append((_config_key(alpha, hidden, activation), mean_harm, cfg))
    elapsed = time.perf_counter() - t0
    print(f"# Phase 1 done in {elapsed:.1f}s")
    summaries.sort(key=lambda s: s[1])
    return rows, summaries


def _run_phase2(winners: list[tuple[str, float, dict]]) -> list[dict]:
    """Phase 2: top-K + baseline, validated across drift_z grid + both
    drift targets. Confirms the winner doesn't sacrifice no-drift R^2."""
    baseline_cfg = {
        "alpha": ALPHA_GRID[0],
        "hidden_layer_sizes": HIDDEN_GRID[0],
        "activation": ACTIVATION_GRID[0],
    }
    baseline_key = _config_key(
        baseline_cfg["alpha"], baseline_cfg["hidden_layer_sizes"], baseline_cfg["activation"],
    )
    keep_configs = list(winners[:PHASE2_TOP_K])
    if all(k != baseline_key for k, _h, _c in keep_configs):
        keep_configs.append((baseline_key, float("nan"), baseline_cfg))
    n_cells = len(PHASE2_DRIFT_Z_LEVELS) * len(PHASE2_DRIFT_TARGETS)
    n_trials = len(keep_configs) * n_cells * PHASE2_N_SEEDS
    print()
    print("# Phase 2: validate %d configs across %d cells x %d seeds = %d trials" % (
        len(keep_configs), n_cells, PHASE2_N_SEEDS, n_trials))
    print()

    rows: list[dict] = []
    t0 = time.perf_counter()
    for ci, (_key, _harm, cfg) in enumerate(keep_configs, start=1):
        for tgt in PHASE2_DRIFT_TARGETS:
            for z in PHASE2_DRIFT_Z_LEVELS:
                for seed in range(PHASE2_N_SEEDS):
                    rng = np.random.default_rng(seed * 97 + ci * 7)
                    row = _run_one_trial(
                        rng, tgt, z,
                        cfg["alpha"], cfg["hidden_layer_sizes"], cfg["activation"],
                    )
                    row["seed"] = seed
                    row["phase"] = "2"
                    rows.append(row)
    elapsed = time.perf_counter() - t0
    print(f"# Phase 2 done in {elapsed:.1f}s")
    return rows


def _print_leaderboard(summaries: list[tuple[str, float, dict]], top: int = 12):
    print()
    print(f"# Phase 1 leaderboard (lower MLP_excess_harm = better; top {top})")
    print(f"{'rank':>4} {'config':<55} {'mean_excess_harm':>18}")
    print("-" * 80)
    for i, (key, harm, _cfg) in enumerate(summaries[:top], start=1):
        print(f"{i:>4} {key:<55} {harm:>18.4f}")
    print()


def _print_phase2_grid(rows: list[dict]):
    """Pivot: rows = config, columns = (drift_target, drift_z), cell = mean harm."""
    by_cfg: dict[tuple, dict[tuple, list[float]]] = {}
    for r in rows:
        cfg_key = (r["alpha"], r["hidden"], r["activation"])
        cell_key = (r["drift_target"], r["drift_z"])
        by_cfg.setdefault(cfg_key, {}).setdefault(cell_key, []).append(r["mlp_excess_harm"])
    print()
    print("# Phase 2 validation grid (mean MLP_excess_harm; lower = better)")
    cell_keys = sorted({(r["drift_target"], r["drift_z"]) for r in rows})
    header = f"{'config':<55} " + " ".join(
        f"{tgt[:3]}_z{z:g}".rjust(10) for tgt, z in cell_keys)
    print(header)
    print("-" * len(header))
    for cfg_key, cells in by_cfg.items():
        alpha, hidden, activation = cfg_key
        label = f"alpha={alpha:g} hidden={hidden} activation={activation}"
        vals = " ".join(
            f"{np.mean(cells.get(ck, [float('nan')])):>10.3f}" for ck in cell_keys
        )
        print(f"{label:<55} {vals}")
    print()


def main():
    print()
    print("# bench_mlp_robustness_sweep")
    print("#   N_TRAIN=%d N_TEST=%d K=%d alphas=%s" % (
        N_TRAIN, N_TEST, N_FEATURES, ALPHAS_DOMINANT.tolist()))
    print("#   alpha_grid=%s" % (ALPHA_GRID,))
    print("#   hidden_grid=%s" % (HIDDEN_GRID,))
    print("#   activation_grid=%s" % (ACTIVATION_GRID,))

    phase1_rows, leaderboard = _run_phase1()
    _print_leaderboard(leaderboard, top=12)

    phase2_rows = _run_phase2(leaderboard)
    _print_phase2_grid(phase2_rows)

    all_rows = phase1_rows + phase2_rows
    out_dir = _HERE / "_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_mlp_robustness_sweep_{stamp}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"# wrote {out_path}")
    print()

    # Final pick: best phase-1 config that also dominates baseline in phase-2's
    # no-drift cell (drift_z=0). We don't want to ship an override that hurts
    # the no-drift case.
    baseline_cfg = (ALPHA_GRID[0], str(HIDDEN_GRID[0]), ACTIVATION_GRID[0])
    no_drift_baseline_harm = float(np.mean([
        r["mlp_excess_harm"]
        for r in phase2_rows
        if r["drift_z"] == 0.0
        and (r["alpha"], r["hidden"], r["activation"]) == baseline_cfg
    ] or [float("nan")]))
    print("# Final-pick filter: candidate must not degrade no-drift baseline")
    print(f"#   baseline (alpha={baseline_cfg[0]:g}, hidden={baseline_cfg[1]}, activation={baseline_cfg[2]})")
    print(f"#   no-drift mean excess harm: {no_drift_baseline_harm:.4f}")

    final_pick: dict[str, Any] | None = None
    for key, harm, cfg in leaderboard:
        cfg_tuple = (cfg["alpha"], str(cfg["hidden_layer_sizes"]), cfg["activation"])
        if cfg_tuple == baseline_cfg:
            continue
        no_drift_cells = [
            r["mlp_excess_harm"]
            for r in phase2_rows
            if r["drift_z"] == 0.0
            and (r["alpha"], r["hidden"], r["activation"]) == cfg_tuple
        ]
        if not no_drift_cells:
            continue
        cand_no_drift = float(np.mean(no_drift_cells))
        if cand_no_drift <= no_drift_baseline_harm + 0.02:  # within 0.02 R^2 of baseline
            final_pick = cfg
            print(f"#   PICK -> {key}: phase1_harm={harm:.4f}, no-drift_harm={cand_no_drift:.4f}")
            break
        else:
            print(f"#   skip  {key}: phase1_harm={harm:.4f}, no-drift_harm={cand_no_drift:.4f} (degrades baseline)")
    print()
    print("# RECOMMENDED ROBUST_MLP_OVERRIDES_UNDER_DRIFT for sklearn MLPRegressor:")
    print("# (paste into feature_drift_report.py)")
    if final_pick:
        print(repr(final_pick))
    else:
        print("# (no config passed the no-drift-stability filter; keep dict empty)")
    print()


if __name__ == "__main__":
    main()
