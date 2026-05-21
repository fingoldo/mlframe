"""Nonlinear-DGP follow-up to ``bench_mlp_robustness_sweep.py``.

The first sweep (2026-05-22) ran on a strictly linear DGP
(``y = alphas . x + noise``) and the empirical winner was an identity-
activation linear head -- unsurprising because the underlying signal had
zero nonlinearity for the MLP to exploit. The open question that scoped-
out of that result: does the same override hold on NONLINEAR DGPs, or
does identity activation sacrifice real nonlinear-capture capacity?

This bench answers it. We define three nonlinear DGPs that span the
common shapes mlframe targets actually take:

  - ``quadratic_dominant``: y = 10*x_dom + 0.5*x_dom^2 + 0.1*sum(x_others)
    + noise. Additive nonlinearity in the dominant feature -- this is
    where identity activation might LOSE because it cannot represent the
    curvature.

  - ``interaction``: y = 10*x_dom + 3*x_dom*x_2 + noise. Bilinear
    interaction term -- only a model that mixes inputs (any hidden layer)
    can capture it; identity is structurally incapable.

  - ``sinusoidal``: y = 5*sin(x_dom) + 3*x_dom + noise. Smooth periodic
    nonlinearity -- the toughest test for MLP-vs-Ridge because Ridge
    can still get the linear-trend term and the residual is the entire
    nonlinear signal.

For each DGP we run the SAME 48-config Cartesian sweep at drift_z=10 on
the dominant feature (phase 1) and a 4-cell validation (phase 2) -- same
methodology as the linear-DGP bench so results are directly comparable.

Three possible outcomes:

  (A) identity wins on every DGP -> ship as-is (the linear-DGP override
      generalises; nonlinear capture is less valuable than drift
      robustness in this regime).

  (B) A nonlinear-aware config wins on the nonlinear DGPs while still
      not degrading the linear-DGP case -> ship THAT as the universal
      override.

  (C) Different DGPs need different overrides -> the production
      recommendation needs a DGP-shape selector. Baseline diagnostics
      already gives us Ridge_R^2 on train; high Ridge_R^2 (>0.95)
      => linear-shaped => use identity. Lower Ridge_R^2 => nonlinear
      shape => use the nonlinear winner.

Run::

    python -m mlframe.profiling.bench_mlp_robustness_sweep_nonlinear
"""
from __future__ import annotations

import csv
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

# mlframe's fast metrics are drop-in for sklearn but 15-17x faster on the
# 500-row test slice we hit per trial: 3520 trials x 6 metric evals each
# = 21k calls, where sklearn's per-call constant-factor dominates.
from mlframe.metrics.core import (
    fast_mean_absolute_error,
    fast_r2_score,
    fast_root_mean_squared_error,
)


N_FEATURES: int = 5
ALPHAS_DOMINANT = np.array([10.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)
N_TRAIN: int = 2000
N_TEST: int = 500
NOISE_STD: float = 1.0
MLP_MAX_ITER: int = 200

ALPHA_GRID: tuple[float, ...] = (1e-4, 1e-1, 1.0, 10.0)
HIDDEN_GRID: tuple[tuple[int, ...], ...] = ((100,), (32, 16), (16,), (8,))
ACTIVATION_GRID: tuple[str, ...] = ("relu", "tanh", "identity")

PHASE1_DRIFT_Z: float = 10.0
PHASE1_DRIFT_TARGET: str = "dominant"
PHASE1_N_SEEDS: int = 15  # 48 cfg x 3 DGP x 15 seeds = 2160 trials/phase1

PHASE2_DRIFT_Z_LEVELS: Sequence[float] = (0.0, 2.0, 10.0, 20.0)
PHASE2_DRIFT_TARGETS = ("dominant",)  # narrow to dominant; noise stays trivial here
PHASE2_N_SEEDS: int = 8
PHASE2_TOP_K: int = 4


def _dgp_linear(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    """Control case (matches the original sweep)."""
    return X @ ALPHAS_DOMINANT + rng.normal(0.0, NOISE_STD, X.shape[0])


def _dgp_quadratic_dominant(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    """Additive curvature in the dominant feature."""
    return (
        10.0 * X[:, 0]
        + 0.5 * X[:, 0] ** 2
        + 0.1 * X[:, 1:].sum(axis=1)
        + rng.normal(0.0, NOISE_STD, X.shape[0])
    )


def _dgp_interaction(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    """Bilinear interaction -- linear models cannot capture by construction."""
    return (
        10.0 * X[:, 0]
        + 3.0 * X[:, 0] * X[:, 1]
        + rng.normal(0.0, NOISE_STD, X.shape[0])
    )


def _dgp_sinusoidal(rng: np.random.Generator, X: np.ndarray) -> np.ndarray:
    """Smooth periodic nonlinearity in the dominant feature."""
    return (
        5.0 * np.sin(X[:, 0])
        + 3.0 * X[:, 0]
        + rng.normal(0.0, NOISE_STD, X.shape[0])
    )


DGPs: dict[str, Callable[[np.random.Generator, np.ndarray], np.ndarray]] = {
    "linear": _dgp_linear,
    "quadratic_dominant": _dgp_quadratic_dominant,
    "interaction": _dgp_interaction,
    "sinusoidal": _dgp_sinusoidal,
}


def _build_trial(
    rng: np.random.Generator, drift_target: str, drift_z: float, dgp_name: str,
):
    X_train = rng.normal(0.0, 1.0, (N_TRAIN, N_FEATURES))
    X_test = rng.normal(0.0, 1.0, (N_TEST, N_FEATURES))
    if drift_target == "dominant":
        X_test[:, 0] += drift_z
    elif drift_target == "noise":
        X_test[:, 1] += drift_z
    dgp = DGPs[dgp_name]
    y_train = dgp(rng, X_train)
    y_test = dgp(rng, X_test)
    return X_train, y_train, X_test, y_test


def _run_one_trial(
    rng: np.random.Generator, drift_target: str, drift_z: float, dgp_name: str,
    alpha: float, hidden: tuple[int, ...], activation: str,
) -> dict:
    X_train, y_train, X_test, y_test = _build_trial(rng, drift_target, drift_z, dgp_name)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
    ridge_pred = ridge.predict(X_test_s)
    ridge_r2 = float(fast_r2_score(y_test, ridge_pred))
    ridge_rmse = float(fast_root_mean_squared_error(y_test, ridge_pred))
    ridge_mae = float(fast_mean_absolute_error(y_test, ridge_pred))

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
    mlp_pred = mlp.predict(X_test_s)
    mlp_r2 = float(fast_r2_score(y_test, mlp_pred))
    mlp_rmse = float(fast_root_mean_squared_error(y_test, mlp_pred))
    mlp_mae = float(fast_mean_absolute_error(y_test, mlp_pred))
    y_test_std = float(np.std(y_test))

    return {
        "dgp": dgp_name,
        "drift_target": drift_target,
        "drift_z": float(drift_z),
        "alpha": alpha,
        "hidden": str(hidden),
        "activation": activation,
        "ridge_test_r2": ridge_r2,
        "ridge_test_rmse": ridge_rmse,
        "ridge_test_mae": ridge_mae,
        "mlp_test_r2": mlp_r2,
        "mlp_test_rmse": mlp_rmse,
        "mlp_test_mae": mlp_mae,
        "y_test_std": y_test_std,
        # excess_harm positive = MLP underperforms Ridge under this metric
        "mlp_excess_harm": ridge_r2 - mlp_r2,            # R^2 gap (lower=better)
        "mlp_excess_rmse": mlp_rmse - ridge_rmse,        # RMSE gap (lower=better)
        "mlp_excess_mae": mlp_mae - ridge_mae,           # MAE gap (lower=better)
        # Normalised gaps: |MLP_err - Ridge_err| / y_test_std. Lets cross-DGP
        # comparison be in std-units; absolute RMSE / MAE are scale-dependent
        # (sinusoidal+drift y_test_std is huge so absolute RMSE numbers are huge).
        "mlp_excess_rmse_norm": (mlp_rmse - ridge_rmse) / max(y_test_std, 1e-9),
        "mlp_excess_mae_norm": (mlp_mae - ridge_mae) / max(y_test_std, 1e-9),
    }


def _config_key(alpha: float, hidden: tuple[int, ...], activation: str) -> str:
    return f"alpha={alpha:g} hidden={hidden} activation={activation}"


METRICS = ("mlp_excess_harm", "mlp_excess_rmse_norm", "mlp_excess_mae_norm")
METRIC_LABELS = {
    "mlp_excess_harm": "R^2 gap (Ridge_R^2 - MLP_R^2)",
    "mlp_excess_rmse_norm": "RMSE gap / y_std",
    "mlp_excess_mae_norm": "MAE gap / y_std",
}


def _phase1_per_dgp(
    dgp_name: str,
) -> tuple[list[dict], dict[str, list[tuple[str, float, dict]]]]:
    """Return (raw rows, per-metric leaderboards). The leaderboard dict maps
    metric_name -> sorted list of (config_key, mean_value, cfg)."""
    rows: list[dict] = []
    by_cfg: dict[tuple, dict] = {}
    t0 = time.perf_counter()
    for i, (alpha, hidden, activation) in enumerate(
            product(ALPHA_GRID, HIDDEN_GRID, ACTIVATION_GRID), start=1):
        per_metric: dict[str, list[float]] = {m: [] for m in METRICS}
        for seed in range(PHASE1_N_SEEDS):
            rng = np.random.default_rng(seed * 31 + i + abs(hash(dgp_name)) % 7919)
            row = _run_one_trial(
                rng, PHASE1_DRIFT_TARGET, PHASE1_DRIFT_Z, dgp_name,
                alpha, hidden, activation,
            )
            row["seed"] = seed
            row["phase"] = "1"
            rows.append(row)
            for m in METRICS:
                per_metric[m].append(row[m])
        cfg = {"alpha": alpha, "hidden_layer_sizes": hidden, "activation": activation}
        by_cfg[(alpha, hidden, activation)] = {
            "cfg": cfg,
            "key": _config_key(alpha, hidden, activation),
            "means": {m: float(np.mean(vs)) for m, vs in per_metric.items()},
        }
    elapsed = time.perf_counter() - t0
    print(f"# Phase 1 [{dgp_name}] done in {elapsed:.1f}s "
          f"({len(ALPHA_GRID) * len(HIDDEN_GRID) * len(ACTIVATION_GRID)} configs x "
          f"{PHASE1_N_SEEDS} seeds = {len(rows)} trials)")
    leaderboards: dict[str, list[tuple[str, float, dict]]] = {}
    for m in METRICS:
        ranked = [(entry["key"], entry["means"][m], entry["cfg"]) for entry in by_cfg.values()]
        ranked.sort(key=lambda r: r[1])
        leaderboards[m] = ranked
    return rows, leaderboards


def _phase2_per_dgp(
    dgp_name: str, winners: dict[str, list[tuple[str, float, dict]]],
) -> list[dict]:
    baseline_cfg = {
        "alpha": ALPHA_GRID[0], "hidden_layer_sizes": HIDDEN_GRID[0],
        "activation": ACTIVATION_GRID[0],
    }
    baseline_key = _config_key(
        baseline_cfg["alpha"], baseline_cfg["hidden_layer_sizes"], baseline_cfg["activation"],
    )
    # Union the top-K from EACH metric's leaderboard so the phase-2 grid
    # validates whichever metric ends up being the production criterion.
    seen: set[str] = set()
    keep: list[tuple[str, float, dict]] = []
    for metric_board in winners.values():
        for entry in metric_board[:PHASE2_TOP_K]:
            if entry[0] not in seen:
                seen.add(entry[0])
                keep.append(entry)
    if baseline_key not in seen:
        keep.append((baseline_key, float("nan"), baseline_cfg))

    rows: list[dict] = []
    t0 = time.perf_counter()
    for ci, (_k, _h, cfg) in enumerate(keep, start=1):
        for tgt in PHASE2_DRIFT_TARGETS:
            for z in PHASE2_DRIFT_Z_LEVELS:
                for seed in range(PHASE2_N_SEEDS):
                    rng = np.random.default_rng(seed * 97 + ci * 7 + abs(hash(dgp_name)) % 7919)
                    row = _run_one_trial(
                        rng, tgt, z, dgp_name,
                        cfg["alpha"], cfg["hidden_layer_sizes"], cfg["activation"],
                    )
                    row["seed"] = seed
                    row["phase"] = "2"
                    rows.append(row)
    elapsed = time.perf_counter() - t0
    print(f"# Phase 2 [{dgp_name}] done in {elapsed:.1f}s ({len(rows)} trials)")
    return rows


def _print_leaderboard(
    dgp_name: str,
    leaderboards: dict[str, list[tuple[str, float, dict]]],
    top: int = 8,
):
    for m, board in leaderboards.items():
        print()
        print(f"# Phase 1 [{dgp_name}] -- ranked by {METRIC_LABELS[m]} (lower = better; top {top})")
        print(f"{'rank':>4} {'config':<55} {'mean':>14}")
        print("-" * 80)
        for i, (k, val, _c) in enumerate(board[:top], start=1):
            print(f"{i:>4} {k:<55} {val:>14.4f}")
    print()


def _print_phase2_grid(dgp_name: str, rows: list[dict]):
    cell_keys = sorted({(r["drift_target"], r["drift_z"]) for r in rows})
    for metric in METRICS:
        by_cfg: dict[tuple, dict[tuple, list[float]]] = {}
        for r in rows:
            ck = (r["alpha"], r["hidden"], r["activation"])
            ek = (r["drift_target"], r["drift_z"])
            by_cfg.setdefault(ck, {}).setdefault(ek, []).append(r[metric])
        print(f"# Phase 2 grid [{dgp_name}] -- {METRIC_LABELS[metric]} (lower = better)")
        header = f"{'config':<55} " + " ".join(
            f"{t[:3]}_z{z:g}".rjust(10) for t, z in cell_keys)
        print(header)
        print("-" * len(header))
        for ck, cells in by_cfg.items():
            alpha, hidden, activation = ck
            label = f"alpha={alpha:g} hidden={hidden} activation={activation}"
            vals = " ".join(
                f"{np.mean(cells.get(c, [float('nan')])):>10.3f}" for c in cell_keys
            )
            print(f"{label:<55} {vals}")
        print()


def _pick_cross_dgp_winner_for_metric(
    metric: str,
    per_dgp_leaderboards: dict[str, dict[str, list[tuple[str, float, dict]]]],
) -> tuple[str | None, dict | None]:
    """Min-max across DGPs for one metric. A config is kept only if it
    appears in every DGP's leaderboard; ranked by worst-case (max across
    DGPs), tiebroken by mean."""
    if not per_dgp_leaderboards:
        return None, None
    boards_for_metric = {
        dgp: lbs[metric] for dgp, lbs in per_dgp_leaderboards.items() if metric in lbs
    }
    if not boards_for_metric:
        return None, None
    all_keys = set(k for lb in boards_for_metric.values() for k, _h, _c in lb)
    rows = []
    for key in all_keys:
        per_dgp: dict[str, float] = {}
        cfg = None
        for dgp_name, lb in boards_for_metric.items():
            for k, h, c in lb:
                if k == key:
                    per_dgp[dgp_name] = h
                    cfg = c
                    break
        if len(per_dgp) == len(boards_for_metric):
            worst = max(per_dgp.values())
            mean = float(np.mean(list(per_dgp.values())))
            rows.append((key, worst, mean, per_dgp, cfg))
    rows.sort(key=lambda r: (r[1], r[2]))

    print()
    print(f"# Cross-DGP min-max leaderboard -- {METRIC_LABELS[metric]} (top 10 by worst-case)")
    print(f"{'rank':>4} {'config':<55} {'worst':>10} {'avg':>10} per_dgp")
    print("-" * 100)
    for i, (k, worst, mean, per_dgp, _c) in enumerate(rows[:10], start=1):
        per_dgp_str = " ".join(f"{n[:5]}={v:.3f}" for n, v in per_dgp.items())
        print(f"{i:>4} {k:<55} {worst:>10.4f} {mean:>10.4f} {per_dgp_str}")
    print()
    if not rows:
        return None, None
    best_key, _, _, _, best_cfg = rows[0]
    return best_key, best_cfg


def main():
    print()
    print("# bench_mlp_robustness_sweep_nonlinear")
    print(f"#   DGPs={list(DGPs.keys())}")
    print(f"#   alpha_grid={ALPHA_GRID}")
    print(f"#   hidden_grid={HIDDEN_GRID}")
    print(f"#   activation_grid={ACTIVATION_GRID}")
    print(f"#   phase1: drift_z={PHASE1_DRIFT_Z} on '{PHASE1_DRIFT_TARGET}', "
          f"{PHASE1_N_SEEDS} seeds per cfg")

    all_rows: list[dict] = []
    per_dgp_leaderboards: dict[str, dict[str, list[tuple[str, float, dict]]]] = {}
    for dgp_name in DGPs:
        p1_rows, lbs = _phase1_per_dgp(dgp_name)
        all_rows.extend(p1_rows)
        per_dgp_leaderboards[dgp_name] = lbs
        _print_leaderboard(dgp_name, lbs)

    for dgp_name, lbs in per_dgp_leaderboards.items():
        p2_rows = _phase2_per_dgp(dgp_name, lbs)
        all_rows.extend(p2_rows)
        _print_phase2_grid(dgp_name, p2_rows)

    out_dir = _HERE / "_results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"bench_mlp_robustness_sweep_nonlinear_{stamp}.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"# wrote {out_path}")
    print()

    per_metric_winners: dict[str, tuple[str | None, dict | None]] = {}
    for metric in METRICS:
        per_metric_winners[metric] = _pick_cross_dgp_winner_for_metric(
            metric, per_dgp_leaderboards,
        )

    print("# CROSS-METRIC SUMMARY (cross-DGP min-max winners under each metric)")
    print(f"{'metric':<28} {'winner':<60}")
    print("-" * 90)
    for metric, (k, _c) in per_metric_winners.items():
        print(f"{METRIC_LABELS[metric]:<28} {k or '(none)':<60}")
    print()

    keys = {k for k, _c in per_metric_winners.values() if k}
    if len(keys) == 1:
        print(f"# AGREEMENT: all 3 metrics agree on winner -> {keys.pop()}")
    else:
        print("# DIVERGENCE: metrics disagree on the winner. Need a metric-of-record")
        print("# decision before shipping the override.")
    print()


if __name__ == "__main__":
    main()
