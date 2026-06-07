"""cProfile harness for ``mlframe.training.dummy_baselines``.

Runs realistic-shape inputs through ``compute_dummy_baselines`` and
dumps the top-N cumulative-time entries per target_type so hotspots
can be optimised.

Usage::

    python -m mlframe.training._profile_dummy_baselines
    python -m mlframe.training._profile_dummy_baselines --n-train 1_000_000
    python -m mlframe.training._profile_dummy_baselines --target binary

Findings (2026-05-10, n_train=1M wall-time WITHOUT cProfile overhead):

- regression_ts: 0.95s (linear_extrap + ACF + per_group + 4 constants + 2
  rolling)
- regression_no_ts: 0.75s (4 constants + per_group)
- binary: 1.18s (sklearn log_loss / roc_auc dominate; vectorisable but
  marginal at this scale)
- 5M-row × 5-target extrapolation: ~25s -- inside the 30-120s plan
  budget.

cProfile attribution inflates pandas-internal call timings by ~13x
(e.g. ``pd.Series.nunique`` shows 1.4s in cProfile vs 0.8ms in standalone
microbench). The apparent "hotspots" in the profile output are NOT real
hotspots — they are attribution artefacts. Standalone wall-time
measurement (without ``cProfile.enable()``) is the authoritative cost.

linear_extrap already caps the tail at 10_000 rows so its polyfit is
<1ms regardless of n_train; closed-form OLS replacement was measured
3x faster in microbench but saves <1ms in practice — skipped per
"measure perf BEFORE applying optimization" rule.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import logging
import pstats
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from mlframe.training.configs import DummyBaselinesConfig
from .dummy import compute_dummy_baselines

# Logger to INFO so we see verdict lines but not DEBUG full tables.
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _make_regression(n_train: int, n_val: int, n_test: int, *, ts: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = n_train + n_val + n_test
    # Synth: trend + weekly seasonality + noise — TS baselines should fire.
    t = np.arange(n)
    y = 0.001 * t + 5.0 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1.0, n)
    cat = rng.integers(0, 50, n).astype("int32")
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
        "cat_lo": cat,
    })
    train_X = X.iloc[:n_train].reset_index(drop=True)
    val_X = X.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_X = X.iloc[n_train + n_val:].reset_index(drop=True)
    if ts:
        # hourly cadence so n=O(1e6) fits inside int64 ns range from 2000.
        ts_all = pd.date_range("2000-01-01", periods=n, freq="h")
        return {
            "target_type": "regression", "target_name": "y",
            "train_y": y[:n_train], "val_y": y[n_train:n_train + n_val], "test_y": y[n_train + n_val:],
            "train_X": train_X, "val_X": val_X, "test_X": test_X,
            "timestamps_train": ts_all[:n_train],
            "timestamps_val": ts_all[n_train:n_train + n_val],
            "timestamps_test": ts_all[n_train + n_val:],
            "cat_features": ["cat_lo"],
        }
    return {
        "target_type": "regression", "target_name": "y",
        "train_y": y[:n_train], "val_y": y[n_train:n_train + n_val], "test_y": y[n_train + n_val:],
        "train_X": train_X, "val_X": val_X, "test_X": test_X,
        "cat_features": ["cat_lo"],
    }


def _make_binary(n_train: int, n_val: int, n_test: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_tr = rng.integers(0, 2, n_train)
    y_va = rng.integers(0, 2, n_val)
    y_te = rng.integers(0, 2, n_test)
    le = LabelEncoder().fit(np.concatenate([y_tr, y_va, y_te]))
    return {
        "target_type": "binary_classification", "target_name": "b",
        "train_y": y_tr, "val_y": y_va, "test_y": y_te,
        "train_X": pd.DataFrame({"x": rng.normal(size=n_train), "cat": rng.integers(0, 20, n_train)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=n_val), "cat": rng.integers(0, 20, n_val)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=n_test), "cat": rng.integers(0, 20, n_test)}),
        "cat_features": ["cat"],
        "target_label_encoder": le,
    }


def _make_multiclass(n_train: int, n_val: int, n_test: int, n_classes: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_tr = rng.integers(0, n_classes, n_train)
    y_va = rng.integers(0, n_classes, n_val)
    y_te = rng.integers(0, n_classes, n_test)
    le = LabelEncoder().fit(np.concatenate([y_tr, y_va, y_te]))
    return {
        "target_type": "multiclass_classification", "target_name": "m",
        "train_y": y_tr, "val_y": y_va, "test_y": y_te,
        "train_X": pd.DataFrame({"x": rng.normal(size=n_train)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=n_val)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=n_test)}),
        "target_label_encoder": le,
    }


def _make_multilabel(n_train: int, n_val: int, n_test: int, K: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "target_type": "multilabel_classification", "target_name": "ml",
        "train_y": rng.integers(0, 2, (n_train, K)),
        "val_y": rng.integers(0, 2, (n_val, K)),
        "test_y": rng.integers(0, 2, (n_test, K)),
        "train_X": pd.DataFrame({"x": rng.normal(size=n_train)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=n_val)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=n_test)}),
    }


def _make_ltr(n_train: int, n_val: int, n_test: int, group_size: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "target_type": "learning_to_rank", "target_name": "ltr",
        "train_y": rng.integers(0, 5, n_train),
        "val_y": rng.integers(0, 5, n_val),
        "test_y": rng.integers(0, 5, n_test),
        "train_X": pd.DataFrame({"x": rng.normal(size=n_train)}),
        "val_X": pd.DataFrame({"x": rng.normal(size=n_val)}),
        "test_X": pd.DataFrame({"x": rng.normal(size=n_test)}),
        "group_ids_train": np.repeat(np.arange(n_train // group_size + 1), group_size)[:n_train],
        "group_ids_val": np.repeat(np.arange(n_val // group_size + 1), group_size)[:n_val],
        "group_ids_test": np.repeat(np.arange(n_test // group_size + 1), group_size)[:n_test],
    }


def _profile_call(args_dict: dict, label: str, top_n: int = 30) -> tuple[float, str]:
    cfg = DummyBaselinesConfig()
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    rep = compute_dummy_baselines(config=cfg, **args_dict)
    profiler.disable()
    elapsed = time.perf_counter() - t0
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(top_n)
    print(f"\n=== {label} (wall {elapsed:.2f}s) ===")
    print(f"strongest={rep.strongest} primary={rep.primary_metric} "
          f"n_baselines={len(rep.table)}")
    return elapsed, s.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=500_000)
    p.add_argument("--n-val", type=int, default=50_000)
    p.add_argument("--n-test", type=int, default=50_000)
    p.add_argument("--target", default="all")
    p.add_argument("--top", type=int, default=30)
    args = p.parse_args()

    n_tr, n_va, n_te = args.n_train, args.n_val, args.n_test

    targets = {
        "regression_ts": lambda: _make_regression(n_tr, n_va, n_te, ts=True),
        "regression_no_ts": lambda: _make_regression(n_tr, n_va, n_te, ts=False),
        "binary": lambda: _make_binary(n_tr, n_va, n_te),
        "multiclass": lambda: _make_multiclass(n_tr, n_va, n_te),
        "multilabel": lambda: _make_multilabel(min(n_tr, 100_000), min(n_va, 10_000), min(n_te, 10_000)),
        "ltr": lambda: _make_ltr(min(n_tr, 100_000), min(n_va, 10_000), min(n_te, 10_000)),
    }

    if args.target != "all" and args.target in targets:
        targets = {args.target: targets[args.target]}

    print(f"# Profiling shape n_train={n_tr:_} n_val={n_va:_} n_test={n_te:_}")
    summary: list[tuple[str, float]] = []
    for label, factory in targets.items():
        d = factory()
        elapsed, prof = _profile_call(d, label, top_n=args.top)
        summary.append((label, elapsed))
        print(prof)

    print("\n# Wall-time summary:")
    total = 0.0
    for label, t in summary:
        print(f"  {label:<22} {t:>8.2f}s")
        total += t
    print(f"  {'TOTAL':<22} {total:>8.2f}s")


if __name__ == "__main__":
    main()
