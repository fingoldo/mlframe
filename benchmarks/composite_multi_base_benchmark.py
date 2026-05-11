"""Measure-first benchmark for OPEN-1 (multi-base forward-stepwise auto-selection).

Decision under test: should Discovery auto-promote single-base linear_residual specs to multi-base via ``forward_stepwise_multi_base``?

Scenarios stress-tested:
- ``S1_single_dominant``: y = 0.95 * b1 + small_eps. Multi-base should pick NO additional bases (gate at min_marginal_gain=0.02 rejects spurious noise). Result locks the "do no harm" property.
- ``S2_two_orthogonal``: y = 0.6 * b1 + 0.3 * b2 + eps. Multi-base should add b2 and reduce hold-out RMSE strictly.
- ``S3_three_orthogonal``: y = 0.5*b1 + 0.3*b2 - 0.2*b3 + eps. All three should be picked under max_k=3.
- ``S4_collinear_pool``: y = 0.5*b1 + eps but pool also contains b1_dup ~= b1 (corr > 0.99). Multi-base should NOT add b1_dup (condition-number gate inside _linear_residual_multi_fit + marginal_gain ~ 0).
- ``S5_partial_pool``: 5 noise candidates + 1 weak signal. Multi-base picks the weak signal if gain >= threshold, otherwise rejects all.

For each scenario, we measure:
- RMSE_single (single-base linear_residual baseline; seed only).
- RMSE_multi (after forward_stepwise_multi_base).
- gain_pct = (RMSE_single - RMSE_multi) / RMSE_single * 100.
- final_bases (list of column names kept).
- search_time_seconds.

Run via: ``D:/ProgramData/anaconda3/python.exe mlframe/benchmarks/composite_multi_base_benchmark.py``

Decision rule:
- If geometric mean of gain_pct across S2, S3 is > 5% AND S1, S4 produce 0 added bases (do-no-harm), AUTO-INTEGRATE is recommended.
- Otherwise standalone-helper-only is the safer default (per R10c "measure-first" rule).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from mlframe.training.composite import (
    _linear_residual_fit,
    _linear_residual_forward,
    _linear_residual_inverse,
    forward_stepwise_multi_base,
)

_OUT_PATH = Path(__file__).parent / "composite_multi_base_benchmark_results.json"


def _single_base_holdout_rmse(y: np.ndarray, base: np.ndarray) -> float:
    """Honest hold-out RMSE for single-base linear_residual: 3-fold CV-RMSE on the OLS prediction (NOT the forward+inverse roundtrip which is a tautology)."""
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(np.arange(y.size)):
        params = _linear_residual_fit(y[train_idx], base[train_idx])
        alpha, beta = float(params["alpha"]), float(params["beta"])
        y_pred = alpha * base[val_idx] + beta
        rmses.append(float(np.sqrt(np.mean((y_pred - y[val_idx]) ** 2))))
    return float(np.mean(rmses))


def _multi_base_holdout_rmse(y: np.ndarray, base_matrix: np.ndarray) -> float:
    """Same metric as single-base but with the (n, K) base matrix."""
    from sklearn.model_selection import KFold
    from mlframe.training.composite import _linear_residual_multi_fit
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rmses = []
    for train_idx, val_idx in kf.split(np.arange(y.size)):
        params = _linear_residual_multi_fit(y[train_idx], base_matrix[train_idx])
        alphas = np.asarray(params["alphas"], dtype=np.float64)
        beta = float(params["beta"])
        y_pred = base_matrix[val_idx].astype(np.float64) @ alphas + beta
        rmses.append(float(np.sqrt(np.mean((y_pred - y[val_idx]) ** 2))))
    return float(np.mean(rmses))


def _make_scenario(name: str, n: int = 2000, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    if name == "S1_single_dominant":
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b_noise = {f"noise_{i}": rng.normal(size=n) for i in range(4)}
        y = 0.95 * b1 + rng.normal(scale=0.1, size=n)
        seeds = ["b1"]
        bases = {"b1": b1, **b_noise}
        return {"y": y, "candidates": bases, "seeds": seeds, "expected_added": 0}
    if name == "S2_two_orthogonal":
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b2 = rng.normal(loc=0.0, scale=3.0, size=n)
        b_noise = {f"noise_{i}": rng.normal(size=n) for i in range(3)}
        y = 0.6 * b1 + 0.3 * b2 + rng.normal(scale=0.2, size=n)
        return {"y": y, "candidates": {"b1": b1, "b2": b2, **b_noise}, "seeds": ["b1"], "expected_added": 1}
    if name == "S3_three_orthogonal":
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b2 = rng.normal(loc=0.0, scale=3.0, size=n)
        b3 = rng.uniform(-1.0, 1.0, size=n)
        b_noise = {f"noise_{i}": rng.normal(size=n) for i in range(3)}
        y = 0.5 * b1 + 0.3 * b2 - 0.2 * b3 + rng.normal(scale=0.1, size=n)
        return {"y": y, "candidates": {"b1": b1, "b2": b2, "b3": b3, **b_noise}, "seeds": ["b1"], "expected_added": 2}
    if name == "S4_collinear_pool":
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b1_dup = b1 + 0.001 * rng.normal(size=n)   # near-clone of b1
        b_noise = {f"noise_{i}": rng.normal(size=n) for i in range(3)}
        y = 0.5 * b1 + rng.normal(scale=0.3, size=n)
        return {"y": y, "candidates": {"b1": b1, "b1_dup": b1_dup, **b_noise}, "seeds": ["b1"], "expected_added": 0}
    if name == "S5_partial_pool":
        b1 = rng.normal(loc=10.0, scale=2.0, size=n)
        b_weak = rng.normal(loc=0.0, scale=1.0, size=n)
        b_noise = {f"noise_{i}": rng.normal(size=n) for i in range(5)}
        y = 0.5 * b1 + 0.05 * b_weak + rng.normal(scale=0.3, size=n)   # weak signal in b_weak
        return {"y": y, "candidates": {"b1": b1, "b_weak": b_weak, **b_noise}, "seeds": ["b1"], "expected_added": 0}   # weak gain may or may not clear 2%
    raise ValueError(f"Unknown scenario {name}")


def run_benchmark() -> Dict[str, Any]:
    scenarios = ["S1_single_dominant", "S2_two_orthogonal", "S3_three_orthogonal",
                 "S4_collinear_pool", "S5_partial_pool"]
    results: List[Dict[str, Any]] = []
    for sc_name in scenarios:
        sc = _make_scenario(sc_name)
        y = sc["y"]
        candidates = sc["candidates"]
        seeds = sc["seeds"]
        # Baseline: single-base RMSE.
        rmse_single = _single_base_holdout_rmse(y, candidates[seeds[0]])
        # Multi-base forward-stepwise.
        t0 = time.perf_counter()
        kept, diag = forward_stepwise_multi_base(
            y, candidates, seed_bases=seeds, max_k=3, min_marginal_rmse_gain=0.02,
        )
        elapsed = time.perf_counter() - t0
        # Multi RMSE on the final kept set.
        if kept:
            base_matrix = np.column_stack([candidates[n] for n in kept])
            rmse_multi = _multi_base_holdout_rmse(y, base_matrix)
        else:
            rmse_multi = rmse_single
        gain_pct = (rmse_single - rmse_multi) / max(rmse_single, 1e-12) * 100.0
        n_added = len(kept) - len(seeds)
        results.append({
            "scenario": sc_name,
            "expected_added": sc["expected_added"],
            "actual_added": n_added,
            "kept_bases": kept,
            "rmse_single": rmse_single,
            "rmse_multi": rmse_multi,
            "gain_pct": gain_pct,
            "search_time_seconds": elapsed,
            "n_diagnostics_steps": len(diag),
        })
    # Decision rule.
    pos_scenarios = [r for r in results if r["scenario"] in ("S2_two_orthogonal", "S3_three_orthogonal")]
    neg_scenarios = [r for r in results if r["scenario"] in ("S1_single_dominant", "S4_collinear_pool")]
    geo_mean_gain = float(np.exp(np.mean(np.log([max(r["gain_pct"], 0.01) for r in pos_scenarios]))))
    no_harm = all(r["actual_added"] == 0 for r in neg_scenarios)
    auto_promote = (geo_mean_gain > 5.0) and no_harm
    summary = {
        "scenarios": results,
        "geo_mean_gain_on_positive_scenarios_pct": geo_mean_gain,
        "no_harm_on_negative_scenarios": no_harm,
        "decision_auto_promote": auto_promote,
        "decision_rule": (
            "auto-promote iff (geo_mean_gain on S2/S3 > 5%) AND (S1/S4 added 0 extra bases)."
        ),
    }
    return summary


if __name__ == "__main__":
    summary = run_benchmark()
    _OUT_PATH.write_text(json.dumps(summary, indent=2, default=float))
    print(f"Wrote benchmark results to {_OUT_PATH}")
    print(json.dumps(summary, indent=2, default=float))
