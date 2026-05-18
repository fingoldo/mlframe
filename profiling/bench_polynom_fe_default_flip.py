"""HIGH #4 2026-05-18: ``fe_smart_polynom_iters`` default-flip evaluation.

Background
----------
``MRMR.fe_smart_polynom_iters`` is 0 by default. When enabled (>0), the
Hermite/Chebyshev/Laguerre polynomial-pair optimiser runs Optuna (or
CMA-ES) for ``fe_smart_polynom_iters`` independent seed-offset rounds,
each running ``fe_smart_polynom_optimization_steps`` trials (default 1000).
This finds non-trivial polynomial pair interactions (XOR-like, saddle,
periodic) that the linear / unary / fixed-binary FE block misses.

Cost: on n=4M production data Optuna at default (5 rounds x 1000 trials)
takes ~88 minutes per MRMR fit. On smaller datasets (n=10k) it's seconds.

Question
--------
Should the default flip to a non-zero value per the "Accuracy / perf over
legacy" rule? If yes, what value?

This benchmark answers it on three controlled problems where polynom-FE
SHOULD help:

1. ``xor_pair``: y = sign(x_a * x_b > 0). Pure degree-1 product.
   Linear FE blind, polynom-FE essential.

2. ``saddle``: y = sign(x_a^2 - x_b^2 > 0). Degree-2 hermite basis.
   Linear FE blind; polynom-FE finds He_2(z_a) - He_2(z_b).

3. ``mixed_linear_interaction``: y = 1.5*x_a + 2.0*x_a*x_b + noise.
   Linear FE captures x_a; polynom-FE adds the interaction lift.

Decision rule
-------------
Flip the default if at least 2 of 3 scenarios show:

* downstream model RMSE / 1-AUC improvement >= 20% vs polynom-FE-OFF
  baseline on the same MRMR support; AND
* MRMR wall time delta is <= 10x slower (so the cost is justified by
  the lift).

Findings 2026-05-18 (printed to stdout, not asserted):
- xor: see stdout
- saddle: see stdout
- mixed: see stdout
- final verdict at end

Run: python profiling/bench_polynom_fe_default_flip.py
"""
from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np
import pandas as pd


def _xor_problem(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n).astype(np.float64)
    x_b = rng.normal(size=n).astype(np.float64)
    # XOR-like discrete target plus 3 noise features.
    y = ((np.sign(x_a * x_b) > 0)).astype(np.int64)
    df = pd.DataFrame({
        "x_a": x_a, "x_b": x_b,
        "noise1": rng.normal(size=n).astype(np.float64),
        "noise2": rng.normal(size=n).astype(np.float64),
        "noise3": rng.normal(size=n).astype(np.float64),
    })
    return df, y, "binary"


def _saddle_problem(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n).astype(np.float64)
    x_b = rng.normal(size=n).astype(np.float64)
    y = ((x_a ** 2 - x_b ** 2) > 0).astype(np.int64)
    df = pd.DataFrame({
        "x_a": x_a, "x_b": x_b,
        "noise1": rng.normal(size=n).astype(np.float64),
        "noise2": rng.normal(size=n).astype(np.float64),
    })
    return df, y, "binary"


def _mixed_problem(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n).astype(np.float64)
    x_b = rng.normal(size=n).astype(np.float64)
    # Linear in x_a + multiplicative interaction.
    z = 1.5 * x_a + 2.0 * x_a * x_b + rng.normal(0, 0.3, n)
    y = (z > np.median(z)).astype(np.int64)
    df = pd.DataFrame({
        "x_a": x_a, "x_b": x_b,
        "noise1": rng.normal(size=n).astype(np.float64),
        "noise2": rng.normal(size=n).astype(np.float64),
    })
    return df, y, "binary"


def _eval_mrmr(*, df, y, target_kind, fe_smart_polynom_iters: int) -> dict:
    """Run MRMR with / without polynom-FE; return wall time + downstream
    metric measured by LightGBM trained on MRMR support."""
    from mlframe.feature_selection.filters.mrmr import MRMR
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold

    t0 = time.perf_counter()
    mrmr = MRMR(
        fe_smart_polynom_iters=fe_smart_polynom_iters,
        fe_smart_polynom_optimization_steps=40 if fe_smart_polynom_iters else 1000,
        fe_max_steps=1,
        verbose=0,
    )
    mrmr.fit(df, y)
    fit_time = time.perf_counter() - t0

    n_eng = len(mrmr._engineered_recipes_ or [])
    n_support = len(mrmr.support_)

    # Downstream: 3-fold LightGBM AUC on the MRMR-transformed frame.
    try:
        import lightgbm as lgb
        X_trans = mrmr.transform(df)
        if hasattr(X_trans, "values"):
            X_arr = X_trans.values
        else:
            X_arr = np.asarray(X_trans)
        kf = KFold(n_splits=3, shuffle=True, random_state=11)
        aucs = []
        for tr, va in kf.split(X_arr):
            model = lgb.LGBMClassifier(
                n_estimators=50, max_depth=4, num_leaves=15,
                random_state=11, verbose=-1,
            )
            model.fit(X_arr[tr], y[tr])
            pred = model.predict_proba(X_arr[va])[:, 1]
            aucs.append(roc_auc_score(y[va], pred))
        downstream_auc = float(np.mean(aucs))
    except Exception as e:
        downstream_auc = float("nan")
        print(f"  downstream eval failed: {e}")

    return {
        "fit_time": fit_time,
        "n_support": n_support,
        "n_engineered": n_eng,
        "downstream_auc": downstream_auc,
    }


def _bench_scenario(name: str, builder: Callable) -> dict:
    print(f"\n=== Scenario: {name} ===")
    df, y, target_kind = builder(n=2000, seed=42)
    print(f"  n={len(df)}, target balance: {np.bincount(y)}")

    print(f"  [polynom OFF] running...")
    off = _eval_mrmr(
        df=df, y=y, target_kind=target_kind,
        fe_smart_polynom_iters=0,
    )
    print(f"  [polynom OFF] fit_time={off['fit_time']:.2f}s, "
          f"support_size={off['n_support']}, "
          f"engineered={off['n_engineered']}, "
          f"downstream_AUC={off['downstream_auc']:.4f}")

    print(f"  [polynom  ON] running (n_iters=2, n_trials=40)...")
    on = _eval_mrmr(
        df=df, y=y, target_kind=target_kind,
        fe_smart_polynom_iters=2,
    )
    print(f"  [polynom  ON] fit_time={on['fit_time']:.2f}s, "
          f"support_size={on['n_support']}, "
          f"engineered={on['n_engineered']}, "
          f"downstream_AUC={on['downstream_auc']:.4f}")

    auc_delta_pct = (
        100.0 * (on['downstream_auc'] - off['downstream_auc']) / max(off['downstream_auc'], 1e-9)
        if np.isfinite(on['downstream_auc']) and np.isfinite(off['downstream_auc']) else float("nan")
    )
    time_ratio = on['fit_time'] / max(off['fit_time'], 1e-9)
    print(f"  delta: AUC {auc_delta_pct:+.2f}%, time {time_ratio:.1f}x slower")

    return {"name": name, "off": off, "on": on,
            "auc_delta_pct": auc_delta_pct, "time_ratio": time_ratio}


def main() -> int:
    print("=" * 70)
    print("HIGH #4 fe_smart_polynom_iters default-flip evaluation")
    print("=" * 70)
    print("Decision rule: flip 0 -> 2 if >= 2/3 scenarios show:")
    print("  * downstream AUC improvement >= 20% (vs polynom-OFF baseline)")
    print("  * fit_time ratio <= 10x slower")
    print()

    results = []
    for name, builder in [
        ("xor", _xor_problem),
        ("saddle", _saddle_problem),
        ("mixed_linear_interaction", _mixed_problem),
    ]:
        try:
            results.append(_bench_scenario(name, builder))
        except Exception as e:
            print(f"\n  [{name}] FAILED: {e}")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if not results:
        print("No scenarios completed; cannot decide.")
        return 1
    passed = [r for r in results
              if r['auc_delta_pct'] >= 20.0 and r['time_ratio'] <= 10.0]
    no_harm = [r for r in results
               if np.isfinite(r['auc_delta_pct']) and r['auc_delta_pct'] >= -1.0]
    print(f"  scenarios passing both criteria: {len(passed)} / {len(results)}")
    print(f"  scenarios with no-harm (>= -1% AUC): {len(no_harm)} / {len(results)}")
    print()
    if len(passed) >= 2:
        print("  RECOMMENDATION: flip default fe_smart_polynom_iters: 0 -> 2")
        print("  (>= 2 scenarios show >= 20% AUC lift within 10x time budget)")
        return 0
    if len(no_harm) == len(results) and len(no_harm) >= 2:
        print("  RECOMMENDATION: keep default 0; polynom-FE shows lift on")
        print("  some scenarios but the AUC delta is < 20% on majority.")
        print("  Re-run on production-shaped data before opting in.")
        return 0
    print("  RECOMMENDATION: keep default 0; mixed / inconclusive results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
