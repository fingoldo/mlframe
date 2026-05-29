"""Bench v2: aggressive overfit constructions + non-conservative aggregator combos.

Wave-1 bench had slice-stable consistently LOSING (gap -0.05% to -0.45%, p>0.5 always). The
likely causes are:

  - The t-LCB penalty at K=5 + correlation_inflation=1.5 stops too EARLY relative to single-val,
    cutting off useful late-iter signal.
  - The patience auto-bump (x1.5 at K=5) tries to compensate but doesn't recover the loss.
  - LGB at moderate scale is already well-regularized -- ES isn't the bottleneck.

This wave tests the OPPOSITE end: hard-to-overfit constructions where ES decisions matter most,
and aggregator settings that are minimally conservative.
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


def _gen_overfit(seed: int, n_train: int, n_val: int, n_test: int, d: int,
                  noise_train: float, noise_val: float, noise_test: float) -> tuple:
    rng = np.random.default_rng(seed)
    def gen(n, sigma):
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    return gen(n_train, noise_train), gen(n_val, noise_val), gen(n_test, noise_test)


def _gen_overfit_temporal(seed: int, n_train: int, n_val: int, n_test: int, d: int) -> tuple:
    """Train + test from clean distribution; val time-ordered with noise ramp."""
    rng = np.random.default_rng(seed)
    def gen_clean(n, sigma):
        X = rng.uniform(0, 1, (n, d))
        y = np.sum(np.sin(2 * np.pi * X), axis=1) + rng.normal(0, sigma, n)
        return pd.DataFrame(X, columns=[f"f{i}" for i in range(d)]), y
    X_tr, y_tr = gen_clean(n_train, 0.3)
    # val: noise grows steeply from 0.2 to 2.0
    X_val = rng.uniform(0, 1, (n_val, d))
    t_val = np.linspace(0, 1, n_val)
    sigma_val = 0.2 + 1.8 * t_val
    y_val = np.sum(np.sin(2 * np.pi * X_val), axis=1) + rng.normal(0, sigma_val)
    X_val_df = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(d)])
    X_te, y_te = gen_clean(n_test, 0.3)
    return (X_tr, y_tr), (X_val_df, y_val), (X_te, y_te), t_val


def _fit_one(seed: int, scenario: str, slice_k: int, source: str = "random",
              aggregate: str = "mean", confidence: float = 0.6, quantile_level: float = 0.5,
              correlation_inflation: float = 1.0, min_rows_per_shard: int = 8,
              patience: int = 25, n_estimators: int = 1000, learning_rate: float = 0.04,
              num_leaves: int = 31, alpha: float = 0.5) -> float:
    if scenario == "overfit_small":
        # Small train (overfittable) + tiny noisy val (noisy ES signal) + clean test
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _gen_overfit(
            seed, n_train=400, n_val=60, n_test=2500, d=4,
            noise_train=0.3, noise_val=1.5, noise_test=0.3)
        t_val = None
    elif scenario == "overfit_temporal":
        # Val time-ordered with noise ramp; clean train and test
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te), t_val = _gen_overfit_temporal(
            seed, n_train=800, n_val=120, n_test=2500, d=5)
    elif scenario == "overfit_v_small_train":
        # Very small train (n=200) so LGB MUST overfit; small noisy val; clean test
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = _gen_overfit(
            seed, n_train=200, n_val=80, n_test=2500, d=4,
            noise_train=0.3, noise_val=1.2, noise_test=0.3)
        t_val = None
    else:
        raise ValueError(scenario)

    fit_params: dict = {}
    if slice_k > 0:
        shards = build_slice_eval_sets(
            X_val, y_val, source=source, k=slice_k,
            min_rows_per_shard=min_rows_per_shard, random_state=seed,
            time_values=t_val if source == "temporal" else None,
        )
        if not shards:
            slice_k = 0

    cb = LightGBMCallback(
        patience=patience, min_delta=0.0,
        monitor_dataset="valid_0", monitor_metric="l2", mode="min",
        slice_k=slice_k if slice_k > 0 else 0,
        slice_aggregate_mode=aggregate,
        slice_aggregate_confidence=confidence,
        slice_aggregate_quantile_level=quantile_level,
        slice_aggregate_alpha=alpha,
        slice_correlation_inflation=correlation_inflation,
        slice_persist_history=False, verbose=0,
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


def _wilcoxon(baseline, slice_es):
    diffs = np.array(baseline) - np.array(slice_es)
    if np.all(diffs == 0):
        return 1.0
    return float(wilcoxon(diffs, alternative="greater", zero_method="zsplit").pvalue)


def _bench(name: str, scenario: str, slice_kwargs: dict, n_seeds: int = 30):
    t0 = time.perf_counter()
    baseline = [_fit_one(seed, scenario, slice_k=0) for seed in range(n_seeds)]
    t1 = time.perf_counter()
    slice_es = [_fit_one(seed, scenario, **slice_kwargs) for seed in range(n_seeds)]
    t2 = time.perf_counter()
    p_value = _wilcoxon(baseline, slice_es)
    med_b, med_s = float(np.median(baseline)), float(np.median(slice_es))
    gap_pct = (med_b - med_s) / med_b * 100.0
    wins = int(np.sum(np.array(slice_es) < np.array(baseline)))
    return dict(name=name, scenario=scenario, n_seeds=n_seeds, slice_kwargs=slice_kwargs,
                median_baseline=round(med_b, 5), median_slice=round(med_s, 5),
                gap_pct=round(gap_pct, 3), p_value=round(p_value, 4), wins=wins,
                wall_s=round(t2 - t0, 1))


def main():
    n_seeds = 30
    results = []

    # Wave-2: minimal-penalty aggregators on overfit-prone setups
    cfgs = [
        # Aggregator: mean (zero penalty) -- if this loses, slice ES has structural overhead bug
        ("OS_mean_random_k5",      "overfit_small",  dict(slice_k=5, source="random",
                                                            aggregate="mean",
                                                            correlation_inflation=1.0)),
        # quantile_level=0.5 (median) -- minimal penalty, robust to outlier shards
        ("OS_q50_random_k5",       "overfit_small",  dict(slice_k=5, source="random",
                                                            aggregate="quantile", quantile_level=0.5,
                                                            correlation_inflation=1.0)),
        # quantile_level=0.7 (mild upper) -- gentler than 0.9
        ("OS_q70_random_k5",       "overfit_small",  dict(slice_k=5, source="random",
                                                            aggregate="quantile", quantile_level=0.7,
                                                            correlation_inflation=1.0)),
        # t-LCB conf=0.5 -- almost-no LCB penalty
        ("OS_tlcb05_random_k5",    "overfit_small",  dict(slice_k=5, source="random",
                                                            aggregate="t_lcb", confidence=0.5,
                                                            correlation_inflation=1.0)),
        # t-LCB conf=0.6 -- gentle LCB
        ("OS_tlcb06_random_k5",    "overfit_small",  dict(slice_k=5, source="random",
                                                            aggregate="t_lcb", confidence=0.6,
                                                            correlation_inflation=1.0)),
        # Temporal scenario: do shards help when there's actually a time signal?
        ("OT_mean_temporal_k5",    "overfit_temporal", dict(slice_k=5, source="temporal",
                                                              aggregate="mean",
                                                              correlation_inflation=1.0)),
        ("OT_tlcb05_temporal_k5",  "overfit_temporal", dict(slice_k=5, source="temporal",
                                                              aggregate="t_lcb", confidence=0.5,
                                                              correlation_inflation=1.0)),
        ("OT_q70_temporal_k5",     "overfit_temporal", dict(slice_k=5, source="temporal",
                                                              aggregate="quantile", quantile_level=0.7,
                                                              correlation_inflation=1.0)),
        # Very small train -> heavy overfit potential
        ("VS_tlcb05_random_k5",    "overfit_v_small_train", dict(slice_k=5, source="random",
                                                                    aggregate="t_lcb", confidence=0.5,
                                                                    correlation_inflation=1.0)),
        ("VS_q50_random_k5",       "overfit_v_small_train", dict(slice_k=5, source="random",
                                                                    aggregate="quantile", quantile_level=0.5,
                                                                    correlation_inflation=1.0)),
    ]
    print(f"Wave-2 bench: {len(cfgs)} configs x {n_seeds} seeds\n")
    for name, scenario, kwargs in cfgs:
        print(f"  >>> {name} ... ", end="", flush=True)
        r = _bench(name, scenario, kwargs, n_seeds=n_seeds)
        results.append(r)
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else ("*" if r["p_value"] < 0.1 else ""))
        print(f"gap={r['gap_pct']:+.2f}%  p={r['p_value']:.3f} {sig}  wins={r['wins']}/{n_seeds}  ({r['wall_s']:.0f}s)")

    print("\n=== Summary (sorted by p_value asc) ===")
    for r in sorted(results, key=lambda x: x["p_value"]):
        sig = "***" if r["p_value"] < 0.01 else ("**" if r["p_value"] < 0.05 else ("*" if r["p_value"] < 0.1 else ""))
        print(f"  {r['name']:30s} gap={r['gap_pct']:+6.2f}%  p={r['p_value']:.4f}  wins={r['wins']}/{n_seeds}  {sig}")

    Path("benchmarks").mkdir(exist_ok=True)
    Path("benchmarks/slice_es_wave2_bench.json").write_text(json.dumps(results, indent=2))
    print("\nWritten: benchmarks/slice_es_wave2_bench.json")


if __name__ == "__main__":
    main()
