"""cProfile + wall-time bench for new dummy_baselines additions (2026-05-10).

Covers components added AFTER the initial dummy_baselines profile pass:
- `_paired_bootstrap_vs_runner_up` (D2)
- `_bootstrap_ci_for_strongest` (D16)
- `_compute_quantile_baselines` + per-α pinball metrics
- `_compute_multi_output_regression` (D4 recursive dispatch)
- `_augment_with_dropped_high_card_cols` (high-card re-attach)

Honest measurement protocol (per the mlframe CLAUDE.md profile rule):

1. Wall-time microbench (no cProfile attribution overhead) on a
   realistic shape FIRST.
2. cProfile only when wall-time is non-trivial (>50ms) AND the function
   is on a hot path (called per-target, per-baseline, etc.).
3. cProfile attribution inflates pandas/sklearn deep-stack call timings
   ~10-13x vs standalone wall-time (see ``_profile_dummy_baselines.py``
   docstring). Document as attribution noise when the standalone
   microbench shows <1ms despite a cProfile-flagged hotspot.

Usage::

    python -m mlframe.training._profile_dummy_baselines_recent
    python -m mlframe.training._profile_dummy_baselines_recent --component paired_bootstrap

== Findings (2026-05-10) ==

Pre-optimization (Python loop with sklearn metrics inside) at the
bootstrap_ci_threshold cutoff (n<2000):

  paired_bootstrap n=1500 1000 resamples:  1130ms
  bootstrap_ci    n=1500 1000 resamples:  1067ms

Per dummy_baselines call (one paired + one bootstrap_ci): ~2.2s.
On a typical 5-target suite run that's ~11s of bootstrap overhead.

Post-numba (parallel + fastmath + LCG-per-iteration index draws):

  paired_bootstrap n=1500 1000 resamples:    3.36ms (336× faster)
  bootstrap_ci    n=1500 1000 resamples:    4.49ms (238× faster)

Per call now ~8ms; per 5-target suite ~40ms. Saves ~10s of wall-time.

Numba kernels: ``_numba_paired_bootstrap_rmse`` /
``_numba_paired_bootstrap_mae`` / ``_numba_bootstrap_rmse_samples`` /
``_numba_bootstrap_mae_samples``. Each uses a per-iteration LCG seed
(Knuth multiplicative hash) so resamples are reproducible across
``n_workers`` (the sklearn ``rng.integers`` call would have been a
shared-state bottleneck under prange). Falls back to the original
sklearn-loop path for log_loss variants (no numba kernel — the per-
class label_binarize cost is sklearn-internal and not worth a custom
kernel at the n<2000 gate) and when numba is unavailable.

== Other findings — no actionable speedup applied ==

quantile per-α (1M rows, 5 alphas): 339ms — dominated by 5 × 2 × N_baselines
sklearn ``mean_pinball_loss`` calls. Numba kernel possible but
mean_pinball_loss is already vectorized; potential ~2-3× win at the
cost of duplicating the algorithm. Skipped — 339ms on 1M rows is
acceptable for a per-target diagnostic.

multi_output_regression (1K-10K rows K=3-5): 6-10s for the FIRST call,
47ms once numba JIT cache warms. The slow first call is dominated by
JIT cold-start of the regression-dispatcher's internal numba kernels
(numba_macro_log_loss, _numba_within_group_descending_rank). Subsequent
multi-output target calls reuse the cached compilations.

augment_with_dropped_high_card_cols (5M rows, 1 col): 277ms — boolean-
mask slice on full-train ndarray. Numpy already optimal for this
pattern; no actionable speedup. Acceptable — fires once per target.

== cProfile attribution noise calibration ==

cProfile of these helpers inflates pandas/sklearn deep-stack timings
~10-13× vs standalone wall-time (e.g. cProfile reports 1.4s on
``pd.Series.nunique`` while standalone microbench shows 0.8ms on the
same call). When cProfile flags an apparent hotspot, cross-check with
this wall-time bench BEFORE chasing the optimization — many "hotspots"
are attribution artifacts of the cProfile internal call-stack tracking.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from mlframe.training.configs import DummyBaselinesConfig
from .dummy import (
    _bootstrap_ci_for_strongest,
    _paired_bootstrap_vs_runner_up,
    compute_dummy_baselines,
)


def _bench(label, fn, n_iter=5):
    """Return (median_ms, result_of_last_call)."""
    times = []
    last = None
    for _ in range(n_iter):
        t0 = time.perf_counter()
        last = fn()
        times.append(time.perf_counter() - t0)
    median_ms = float(np.median(times) * 1000)
    print(f"  {label:<55} median={median_ms:>8.2f}ms")
    return median_ms, last


def bench_paired_bootstrap():
    """D2 paired-bootstrap on (strongest, runner-up) — n-gated to <2000.

    Representative n: 100-1500 (real-world val/test shapes hit
    bootstrap_ci_threshold cutoff at exactly 2000)."""
    print("\n=== _paired_bootstrap_vs_runner_up (D2, gated n<2000) ===")
    rng = np.random.default_rng(0)
    for n in (100, 500, 1500):
        y = rng.normal(size=n)
        # Two predictions of similar quality (close metric values)
        p_strongest = y + rng.normal(0, 0.5, n)  # better
        p_runner = y + rng.normal(0, 0.6, n)  # slightly worse
        # Build a minimal table for the helper
        from mlframe.metrics.core import fast_root_mean_squared_error
        rmse_s = float(fast_root_mean_squared_error(y, p_strongest))
        rmse_r = float(fast_root_mean_squared_error(y, p_runner))
        table = pd.DataFrame(
            {"val_RMSE": [rmse_s, rmse_r]},
            index=["strongest", "runner_up"],
        )
        val_preds = {"strongest": p_strongest, "runner_up": p_runner}
        test_preds = {"strongest": p_strongest, "runner_up": p_runner}
        _bench(
            f"n={n} 1000 resamples",
            lambda: _paired_bootstrap_vs_runner_up(
                "regression", "strongest", "val_RMSE", table,
                val_preds, test_preds, y, y,
                n_resamples=1000,
                seed=42,
            ),
        )


def bench_bootstrap_ci():
    """D16 bootstrap CI for strongest baseline — same n-gate."""
    print("\n=== _bootstrap_ci_for_strongest (D16, gated n<2000) ===")
    rng = np.random.default_rng(0)
    for n in (100, 500, 1500):
        y = rng.normal(size=n)
        p = y + rng.normal(0, 0.5, n)
        val_preds = {"baseline": p}
        test_preds = {"baseline": p}
        _bench(
            f"n={n} 1000 resamples",
            lambda: _bootstrap_ci_for_strongest(
                "regression", "baseline", "val_RMSE",
                val_preds, test_preds, y, y,
                n_resamples=1000, seed=42,
            ),
        )


def bench_quantile_per_alpha():
    """Per-α empirical quantile dispatch + pinball-loss metrics."""
    print("\n=== _compute_quantile_baselines + pinball metrics ===")
    rng = np.random.default_rng(0)
    cfg = DummyBaselinesConfig()
    for n in (10_000, 100_000, 1_000_000):
        n_va = max(n // 10, 100)
        y_tr = rng.normal(0, 1, n)
        y_va = rng.normal(0, 1, n_va)
        y_te = rng.normal(0, 1, n_va)
        X_tr = pd.DataFrame({"x": rng.normal(size=n)})
        X_va = pd.DataFrame({"x": rng.normal(size=n_va)})
        X_te = pd.DataFrame({"x": rng.normal(size=n_va)})
        _bench(
            f"n_train={n:_} K=5 alphas",
            lambda: compute_dummy_baselines(
                target_type="quantile_regression",
                target_name="q",
                train_y=y_tr, val_y=y_va, test_y=y_te,
                train_X=X_tr, val_X=X_va, test_X=X_te,
                quantile_alphas=[0.1, 0.25, 0.5, 0.75, 0.9],
                config=cfg,
            ),
        )


def bench_multi_output_regression():
    """D4 multi-output dispatcher (recursive K calls)."""
    print("\n=== _compute_multi_output_regression (D4) ===")
    rng = np.random.default_rng(0)
    cfg = DummyBaselinesConfig()
    for n, K in [(1000, 3), (10_000, 5), (100_000, 3)]:
        n_va = max(n // 10, 100)
        y_tr = rng.normal(0, 1, (n, K))
        y_va = rng.normal(0, 1, (n_va, K))
        y_te = rng.normal(0, 1, (n_va, K))
        X_tr = pd.DataFrame({"x": rng.normal(size=n)})
        X_va = pd.DataFrame({"x": rng.normal(size=n_va)})
        X_te = pd.DataFrame({"x": rng.normal(size=n_va)})
        _bench(
            f"n_train={n:_} K={K}",
            lambda: compute_dummy_baselines(
                target_type="regression", target_name="Y",
                train_y=y_tr, val_y=y_va, test_y=y_te,
                train_X=X_tr, val_X=X_va, test_X=X_te,
                config=cfg,
            ),
        )


def bench_augment_helper():
    """High-card re-attach helper on real-shape inputs."""
    print("\n=== _augment_with_dropped_high_card_cols ===")
    from mlframe.training.core import _augment_with_dropped_high_card_cols
    rng = np.random.default_rng(0)
    for n in (100_000, 1_000_000, 5_000_000):
        n_va = max(n // 10, 100)
        # Pre-OD train length = n; OD-filtered length = ~0.95*n (5% outliers)
        n_post_od = int(n * 0.95)
        train_od_idx = rng.choice(n, n_post_od, replace=False)
        train_od_mask = np.zeros(n, dtype=bool)
        train_od_mask[train_od_idx] = True

        train = pd.DataFrame({"x": rng.normal(size=n_post_od).astype("float32")})
        val = pd.DataFrame({"x": rng.normal(size=n_va).astype("float32")})
        test = pd.DataFrame({"x": rng.normal(size=n_va).astype("float32")})
        # 1 dropped col - group_id-like high-card string
        n_groups = 600
        group_id_train = np.array([f"g_{i % n_groups:04d}" for i in range(n)], dtype=object)
        group_id_val = np.array([f"g_{i % n_groups:04d}" for i in range(n_va)], dtype=object)
        group_id_test = np.array([f"g_{i % n_groups:04d}" for i in range(n_va)], dtype=object)
        dropped = {
            "group_id": {
                "train": group_id_train,
                "val": group_id_val,
                "test": group_id_test,
            }
        }
        _bench(
            f"n_train={n:_} (post-OD={n_post_od:_}) 1 high-card col",
            lambda: _augment_with_dropped_high_card_cols(
                dropped, train, val, test,
                train_od_idx=train_od_mask,
            ),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--component",
        default="all",
        choices=("all", "paired_bootstrap", "bootstrap_ci",
                 "quantile", "multi_output", "augment"),
    )
    args = p.parse_args()

    print("# Wall-time bench (NO cProfile attribution overhead)")
    print(f"# Numpy: {np.__version__}, Pandas: {pd.__version__}")

    if args.component in ("all", "paired_bootstrap"):
        bench_paired_bootstrap()
    if args.component in ("all", "bootstrap_ci"):
        bench_bootstrap_ci()
    if args.component in ("all", "quantile"):
        bench_quantile_per_alpha()
    if args.component in ("all", "multi_output"):
        bench_multi_output_regression()
    if args.component in ("all", "augment"):
        bench_augment_helper()


if __name__ == "__main__":
    main()
