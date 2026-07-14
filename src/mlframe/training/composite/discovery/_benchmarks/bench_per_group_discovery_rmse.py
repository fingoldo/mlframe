"""Bench: opt-in per-group composite discovery -- honest-holdout RMSE, per-group spec vs single global-forced spec.

Lever: ``CompositeTargetDiscoveryConfig.per_group_discovery_enabled`` (see ``discovery/_per_group.py`` /
``discovery/_fit.py``'s trailing gated block). The core value proposition: when different groups/clusters genuinely
have different (base, transform) relationships, forcing ONE globally-discovered spec onto every group is a
compromise that under-fits every group; per-group discovery (opt-in, >= ``per_group_min_rows`` rows per group,
default 500) should let each large group keep its own best spec while small/unseen groups still get the safe
global fallback.

Honest method (leakage-safe): each group gets its own TRAIN/HOLDOUT split. Discovery (both the global run and each
qualifying group's own run) only ever sees TRAIN rows. For each group we then train a LightGBM inner model on the
group's TRAIN rows to predict T = transform(y, base) under (a) the GLOBAL spec forced onto every group and (b) that
group's OWN per-group-discovered spec, invert to y-scale, and score RMSE on the group's DISJOINT HOLDOUT rows
(never touched by discovery or by the inner model's fit). No group's holdout ever informs another group's spec or
inner model.

Scenario: 3 groups with genuinely different DGPs (A: y = base_1 + noise; B: y = 3*base_2 + noise; C: y = pure
noise, unrelated to either base) plus a small group D (150 rows, below the 500-row per-group floor) to confirm the
fallback path is exercised. Run:
    python -m mlframe.training.composite.discovery._benchmarks.bench_per_group_discovery_rmse
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.composite.estimator import CompositeTargetEstimator
from mlframe.training.configs import CompositeTargetDiscoveryConfig


def _make_group(rng: np.random.Generator, label: str, n: int, y_fn) -> pd.DataFrame:
    """One group's rows: two base columns plus ``y = y_fn(base_1, base_2)`` plus noise."""
    base_1 = rng.normal(size=n)
    base_2 = rng.normal(size=n)
    y = y_fn(base_1, base_2) + rng.normal(scale=0.05, size=n)
    return pd.DataFrame({"base_1": base_1, "base_2": base_2, "y": y, "grp": label})


def _fast_config(**overrides: Any) -> CompositeTargetDiscoveryConfig:
    """Fast per-group discovery config for the bench, with ``**overrides`` applied."""
    base: dict[str, Any] = dict(
        enabled=True,
        per_group_discovery_enabled=True,
        per_group_column="grp",
        per_group_min_rows=500,
        tiny_screening_models="single_lgbm",
        tiny_model_n_seed_repeats=1,
        interaction_base_discovery_enabled=False,
        auto_chain_discovery_enabled=False,
        transform_waic_validation_enabled=False,
        honest_holdout_frac=0.2,
        multi_base_enabled=False,
        base_candidates=["base_1", "base_2"],
    )
    base.update(overrides)
    return CompositeTargetDiscoveryConfig(**base)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root-mean-squared error between ``y_true`` and ``y_pred``."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _fit_inner_and_score(spec, train_df: pd.DataFrame, holdout_df: pd.DataFrame, seed: int) -> float:
    """Train a LightGBM CompositeTargetEstimator under ``spec`` on ``train_df``, score y-scale RMSE on ``holdout_df``."""
    from lightgbm import LGBMRegressor

    est = CompositeTargetEstimator(
        base_estimator=LGBMRegressor(n_estimators=60, num_leaves=15, learning_rate=0.1, random_state=seed, verbosity=-1),
        transform_name=spec.transform_name,
        base_column=spec.base_column,
        moe_gate_enabled=False,  # isolate the raw spec's reconstruction quality, no {composite,raw,lag} MoE routing
    )
    X_cols = ["base_1", "base_2"]
    est.fit(train_df[X_cols], train_df["y"].to_numpy())
    y_hat = est.predict(holdout_df[X_cols])
    return _rmse(holdout_df["y"].to_numpy(), y_hat)


def run_bench(seed: int = 0) -> dict:
    """Run one honest-holdout global-forced-vs-per-group RMSE comparison across the 4 scenario groups."""
    rng = np.random.default_rng(seed)
    # (label, n_train, n_holdout, DGP)
    specs_wanted = [
        ("A", 900, 300, lambda b1, b2: b1),
        ("B", 900, 300, lambda b1, b2: 3.0 * b2),
        ("C", 900, 300, lambda b1, b2: np.zeros_like(b1)),
        ("D", 150, 150, lambda b1, b2: b1),  # below per_group_min_rows=500 -> exercises the fallback path
    ]
    train_parts, holdout_parts = [], []
    for label, n_tr, n_ho, fn in specs_wanted:
        train_parts.append(_make_group(rng, label, n_tr, fn))
        holdout_parts.append(_make_group(rng, label, n_ho, fn))
    train_df = pd.concat(train_parts, ignore_index=True)
    holdout_by_group = {label: df for (label, *_r), df in zip(specs_wanted, holdout_parts)}

    cfg = _fast_config()
    disc = CompositeTargetDiscovery(cfg)
    train_idx = np.arange(len(train_df))
    disc.fit(train_df, "y", ["base_1", "base_2"], train_idx)

    if not disc.specs_:
        raise RuntimeError("global discovery found no specs -- bench cannot proceed")
    global_spec = disc.specs_[0]  # top-ranked global spec, the one that WOULD be forced onto every group

    results: dict[str, dict] = {}
    for label, n_tr, n_ho, _fn in specs_wanted:
        group_train = train_df[train_df["grp"] == label].reset_index(drop=True)
        group_holdout = holdout_by_group[label]

        group_specs = disc.specs_by_group_.get(label)
        own_spec = group_specs[0] if group_specs else global_spec
        used_own_discovery = group_specs is not None

        rmse_global_forced = _fit_inner_and_score(global_spec, group_train, group_holdout, seed=seed)
        rmse_per_group = _fit_inner_and_score(own_spec, group_train, group_holdout, seed=seed)

        results[label] = {
            "n_train": n_tr,
            "n_holdout": n_ho,
            "used_own_discovery": used_own_discovery,
            "global_spec": global_spec.name,
            "own_spec": own_spec.name,
            "rmse_global_forced": rmse_global_forced,
            "rmse_per_group": rmse_per_group,
            "improvement_pct": 100.0 * (rmse_global_forced - rmse_per_group) / rmse_global_forced,
        }
    return results


def main() -> None:
    """Run the bench across 5 seeds and print a per-group, per-seed RMSE comparison table."""
    all_results = []
    for seed in range(5):
        all_results.append(run_bench(seed))

    print(f"{'group':<6}{'n_tr':>6}{'n_ho':>6}{'own_disc':>9}  {'global_spec':<22}{'own_spec':<22}"
          f"{'rmse_global':>13}{'rmse_pergrp':>13}{'improve%':>10}")
    agg: dict[str, list[float]] = {"A": [], "B": [], "C": [], "D": []}
    for seed, res in enumerate(all_results):
        print(f"-- seed {seed} --")
        for label, r in res.items():
            print(
                f"{label:<6}{r['n_train']:>6}{r['n_holdout']:>6}{str(r['used_own_discovery']):>9}  "
                f"{r['global_spec']:<22}{r['own_spec']:<22}"
                f"{r['rmse_global_forced']:>13.4f}{r['rmse_per_group']:>13.4f}{r['improvement_pct']:>9.1f}%"
            )
            agg[label].append(r["improvement_pct"])

    print("\nMean improvement %% (rmse_per_group vs rmse_global_forced), 5 seeds:")
    for label, vals in agg.items():
        print(f"  group {label}: mean={np.mean(vals):+.1f}%  min={np.min(vals):+.1f}%  max={np.max(vals):+.1f}%")


if __name__ == "__main__":
    main()
