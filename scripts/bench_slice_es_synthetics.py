"""Exploratory bench: 4 synthetic constructions for slice-stable ES value-prop.

Run from mlframe root:
    D:/ProgramData/anaconda3/python.exe scripts/bench_slice_es_synthetics.py

For each construction we run baseline (single-val ES) vs slice-stable (K random and K temporal)
across N seeds, paired Wilcoxon one-sided test. Picks the construction with the strongest
signal that slice-stable improves test RMSE.

The constructions target known failure modes of single-val ES:

  S1. SMALL+NOISY val: high val noise -> noisy ES signal -> single-val picks lucky/unlucky iter.
      Slice-stable t-LCB penalty stops earlier on average. Expected: source="random" wins.
  S2. TEMPORAL DRIFT in val: val rows are time-ordered and noise sigma ramps up across time.
      Single-val averages out the heteroscedasticity. source="temporal" should expose it.
  S3. CONCEPT SHIFT mid-val: late portion of val has shifted target. Single-val averages;
      temporal shards expose the late-region degradation as variance.
  S4. LONG OVERFIT TAIL: small train (1k), heavy LGB depth, many iters with low learning rate.
      LGB overfits train noise late; high val noise hides the regression in single-val.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
from scipy.stats import wilcoxon

from mlframe.training._callbacks import LightGBMCallback
from mlframe.training._data_helpers import _setup_eval_set
from mlframe.training._slice_helpers import build_slice_eval_sets


# ---------------------------------------------------------------------------
# Synthetic constructions
# ---------------------------------------------------------------------------

def _gen_S1(seed: int) -> tuple:
    """Small + noisy val, signal everywhere."""
    rng = np.random.default_rng(seed)
    d = 4
    def gen(n, sigma):
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    return gen(2000, 0.3), gen(80, 1.2), gen(3000, 0.3), None


def _gen_S2(seed: int) -> tuple:
    """Temporal drift: val ordered by t, sigma ramps 0.1 -> 1.5 across rows."""
    rng = np.random.default_rng(seed)
    d = 5
    def gen_uniform(n, sigma):
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y

    X_tr, y_tr = gen_uniform(2500, 0.4)
    # val: ordered by t with sigma ramp
    n_val = 250
    X_val = rng.uniform(0, 1, (n_val, d))
    t_val = np.linspace(0, 1, n_val)
    sigma_val = 0.1 + 1.4 * t_val
    y_val = np.sum(np.sin(2 * np.pi * X_val), axis=1) + rng.normal(0, sigma_val)
    X_val_df = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(d)])
    X_te, y_te = gen_uniform(2500, 0.4)
    return (X_tr, y_tr), (X_val_df, y_val), (X_te, y_te), t_val


def _gen_S3(seed: int) -> tuple:
    """Concept shift in val second half: y -> y + 0.6."""
    rng = np.random.default_rng(seed)
    d = 5
    def gen_uniform(n, sigma):
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y

    X_tr, y_tr = gen_uniform(2500, 0.3)
    n_val = 240
    X_val = rng.uniform(0, 1, (n_val, d))
    y_val_base = np.sum(np.sin(2 * np.pi * X_val), axis=1) + rng.normal(0, 0.3, n_val)
    shift = np.where(np.arange(n_val) < n_val // 2, 0.0, 0.6)
    y_val = y_val_base + shift
    X_val_df = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(d)])
    t_val = np.arange(n_val) / n_val
    X_te, y_te = gen_uniform(2500, 0.3)
    return (X_tr, y_tr), (X_val_df, y_val), (X_te, y_te), t_val


def _gen_S4(seed: int) -> tuple:
    """Long overfit tail: small train + heavy LGB, high val noise."""
    rng = np.random.default_rng(seed)
    d = 6
    def gen(n, sigma):
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    return gen(800, 0.4), gen(120, 1.0), gen(3000, 0.4), None


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------

def _fit_one(
    seed: int,
    construction: str,
    slice_k: int,
    source: str = "random",
    aggregate: str = "t_lcb",
    n_estimators: int = 800,
    learning_rate: float = 0.03,
    num_leaves: int = 15,
    patience: int = 40,
    deep_overfit: bool = False,
) -> float:
    if construction == "S1":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te), t_val = _gen_S1(seed)
    elif construction == "S2":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te), t_val = _gen_S2(seed)
    elif construction == "S3":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te), t_val = _gen_S3(seed)
    elif construction == "S4":
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te), t_val = _gen_S4(seed)
    else:
        raise ValueError(construction)

    if deep_overfit:
        n_estimators = 2000
        learning_rate = 0.02
        num_leaves = 31

    fit_params: dict = {}
    if slice_k > 0:
        shards = build_slice_eval_sets(
            X_val, y_val, source=source, k=slice_k, min_rows_per_shard=12,
            random_state=seed,
            time_values=t_val if (source == "temporal") else None,
        )
        if not shards:
            # too few rows per shard; fall through to single-val mode silently
            slice_k = 0

    cb = LightGBMCallback(
        patience=patience, min_delta=0.0,
        monitor_dataset="valid_0", monitor_metric="l2", mode="min",
        slice_k=slice_k if slice_k > 0 else 0,
        slice_aggregate_mode=aggregate,
        slice_aggregate_confidence=0.9,
        slice_correlation_inflation=1.5,
        slice_persist_history=False,
        verbose=0,
    )
    fit_params["callbacks"] = [cb]
    if slice_k > 0:
        _setup_eval_set("LGBMRegressor", fit_params, X_val, y_val,
                        model_category="lgb", extra_eval_sets=shards)
    else:
        _setup_eval_set("LGBMRegressor", fit_params, X_val, y_val, model_category="lgb")

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators, learning_rate=learning_rate, verbose=-1,
        num_leaves=num_leaves, random_state=seed,
    )
    model.fit(X_tr, y_tr, **fit_params)
    best_it = cb.best_iter or model.n_estimators
    preds = model.predict(X_te, num_iteration=best_it)
    return float(np.sqrt(np.mean((preds - y_te) ** 2)))


def _wilcoxon_one_sided(baseline: list[float], slice_es: list[float]) -> float:
    diffs = np.array(baseline) - np.array(slice_es)
    if np.all(diffs == 0):
        return 1.0
    return float(wilcoxon(diffs, alternative="greater", zero_method="zsplit").pvalue)


def _bench_one(name: str, construction: str, slice_kwargs: dict, n_seeds: int = 30,
               deep_overfit: bool = False) -> dict:
    t0 = time.perf_counter()
    baseline = [_fit_one(seed, construction, slice_k=0, deep_overfit=deep_overfit)
                for seed in range(n_seeds)]
    t1 = time.perf_counter()
    slice_es = [_fit_one(seed, construction, deep_overfit=deep_overfit, **slice_kwargs)
                for seed in range(n_seeds)]
    t2 = time.perf_counter()
    p_value = _wilcoxon_one_sided(baseline, slice_es)
    med_b, med_s = float(np.median(baseline)), float(np.median(slice_es))
    gap_pct = (med_b - med_s) / med_b * 100.0
    wins = int(np.sum(np.array(slice_es) < np.array(baseline)))
    return {
        "name": name,
        "construction": construction,
        "n_seeds": n_seeds,
        "slice_kwargs": slice_kwargs,
        "median_baseline_rmse": round(med_b, 5),
        "median_slice_rmse": round(med_s, 5),
        "gap_pct": round(gap_pct, 3),
        "p_value": round(p_value, 4),
        "wins": wins,
        "baseline_time_s": round(t1 - t0, 1),
        "slice_time_s": round(t2 - t1, 1),
    }


def main():
    n_seeds = 30
    results = []

    print("Benching 4 constructions x {random, temporal} shards, 30 seeds each.")
    print("(~5-15 min total)\n")

    cfgs = [
        ("S1_random_k5_tlcb",     "S1", dict(slice_k=5, source="random",   aggregate="t_lcb"),     False),
        ("S2_temporal_k5_tlcb",   "S2", dict(slice_k=5, source="temporal", aggregate="t_lcb"),     False),
        ("S2_temporal_k5_quant",  "S2", dict(slice_k=5, source="temporal", aggregate="quantile"),  False),
        ("S3_temporal_k4_tlcb",   "S3", dict(slice_k=4, source="temporal", aggregate="t_lcb"),     False),
        ("S4_random_k5_tlcb_deep","S4", dict(slice_k=5, source="random",   aggregate="t_lcb"),     True),
        ("S4_random_k5_mms_deep", "S4", dict(slice_k=5, source="random",   aggregate="mean_minus_std"), True),
    ]
    for name, c, kwargs, deep in cfgs:
        print(f"  >>> {name} ... ", end="", flush=True)
        r = _bench_one(name, c, kwargs, n_seeds=n_seeds, deep_overfit=deep)
        results.append(r)
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else ("*" if r["p_value"] < 0.1 else ""))
        print(f"gap={r['gap_pct']:+.2f}%  p={r['p_value']:.3f} {sig}  wins={r['wins']}/{n_seeds}  ({r['baseline_time_s']+r['slice_time_s']:.0f}s)")

    print("\n=== Summary (sorted by p_value asc) ===")
    for r in sorted(results, key=lambda x: x["p_value"]):
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else ("*" if r["p_value"] < 0.1 else ""))
        print(f"  {r['name']:30s} gap={r['gap_pct']:+6.2f}%  p={r['p_value']:.4f}  {sig}")

    Path("benchmarks").mkdir(exist_ok=True)
    out_path = Path("benchmarks/slice_es_synthetic_bench.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
