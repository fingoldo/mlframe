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
        "dgp": dgp_name,
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


def _phase1_per_dgp(dgp_name: str) -> tuple[list[dict], list[tuple[str, float, dict]]]:
    rows: list[dict] = []
    summaries: list[tuple[str, float, dict]] = []
    t0 = time.perf_counter()
    for i, (alpha, hidden, activation) in enumerate(
            product(ALPHA_GRID, HIDDEN_GRID, ACTIVATION_GRID), start=1):
        harms = []
        for seed in range(PHASE1_N_SEEDS):
            rng = np.random.default_rng(seed * 31 + i + abs(hash(dgp_name)) % 7919)
            row = _run_one_trial(
                rng, PHASE1_DRIFT_TARGET, PHASE1_DRIFT_Z, dgp_name,
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
    print(f"# Phase 1 [{dgp_name}] done in {elapsed:.1f}s "
          f"({len(ALPHA_GRID) * len(HIDDEN_GRID) * len(ACTIVATION_GRID)} configs x "
          f"{PHASE1_N_SEEDS} seeds = {len(rows)} trials)")
    summaries.sort(key=lambda s: s[1])
    return rows, summaries


def _phase2_per_dgp(
    dgp_name: str, winners: list[tuple[str, float, dict]],
) -> list[dict]:
    baseline_cfg = {
        "alpha": ALPHA_GRID[0], "hidden_layer_sizes": HIDDEN_GRID[0],
        "activation": ACTIVATION_GRID[0],
    }
    baseline_key = _config_key(
        baseline_cfg["alpha"], baseline_cfg["hidden_layer_sizes"], baseline_cfg["activation"],
    )
    keep = list(winners[:PHASE2_TOP_K])
    if all(k != baseline_key for k, _h, _c in keep):
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


def _print_leaderboard(dgp_name: str, summaries: list[tuple[str, float, dict]], top: int = 10):
    print()
    print(f"# Phase 1 leaderboard [{dgp_name}] (lower MLP_excess_harm = better; top {top})")
    print(f"{'rank':>4} {'config':<55} {'mean_excess_harm':>18}")
    print("-" * 80)
    for i, (k, harm, _c) in enumerate(summaries[:top], start=1):
        print(f"{i:>4} {k:<55} {harm:>18.4f}")
    print()


def _print_phase2_grid(dgp_name: str, rows: list[dict]):
    by_cfg: dict[tuple, dict[tuple, list[float]]] = {}
    for r in rows:
        ck = (r["alpha"], r["hidden"], r["activation"])
        ek = (r["drift_target"], r["drift_z"])
        by_cfg.setdefault(ck, {}).setdefault(ek, []).append(r["mlp_excess_harm"])
    print(f"# Phase 2 grid [{dgp_name}] (mean MLP_excess_harm; lower = better)")
    cell_keys = sorted({(r["drift_target"], r["drift_z"]) for r in rows})
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


def _pick_cross_dgp_winner(
    per_dgp_leaderboards: dict[str, list[tuple[str, float, dict]]],
) -> tuple[str | None, dict | None]:
    """Identify the config with the BEST WORST-CASE mean excess harm across
    all DGPs. A config that wins on linear but loses on interaction is
    rejected; we want max_dgp(mean_harm) minimised so the override is safe
    across signal shapes."""
    if not per_dgp_leaderboards:
        return None, None
    all_keys = set(k for lb in per_dgp_leaderboards.values() for k, _h, _c in lb)
    rows = []
    for key in all_keys:
        per_dgp: dict[str, float] = {}
        cfg = None
        for dgp_name, lb in per_dgp_leaderboards.items():
            for k, h, c in lb:
                if k == key:
                    per_dgp[dgp_name] = h
                    cfg = c
                    break
        if len(per_dgp) == len(per_dgp_leaderboards):
            worst = max(per_dgp.values())
            mean = float(np.mean(list(per_dgp.values())))
            rows.append((key, worst, mean, per_dgp, cfg))
    rows.sort(key=lambda r: (r[1], r[2]))

    print()
    print("# Cross-DGP min-max leaderboard (top 10 by worst-case mean excess harm)")
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
    per_dgp_leaderboards: dict[str, list[tuple[str, float, dict]]] = {}
    for dgp_name in DGPs:
        p1_rows, lb = _phase1_per_dgp(dgp_name)
        all_rows.extend(p1_rows)
        per_dgp_leaderboards[dgp_name] = lb
        _print_leaderboard(dgp_name, lb)

    for dgp_name, lb in per_dgp_leaderboards.items():
        p2_rows = _phase2_per_dgp(dgp_name, lb)
        all_rows.extend(p2_rows)
        _print_phase2_grid(dgp_name, p2_rows)

    best_key, best_cfg = _pick_cross_dgp_winner(per_dgp_leaderboards)

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

    print("# RECOMMENDED CROSS-DGP MLP override (min-max winner):")
    if best_cfg:
        print(f"#   key={best_key}")
        print(f"#   cfg={best_cfg}")
    else:
        print("# (no cross-DGP winner identified)")
    print()


if __name__ == "__main__":
    main()
