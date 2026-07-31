"""A/B bench for ``_jackknife_metric``'s mask-vs-slice-concatenate LOO gather (2026-07-31).

Confirms bit-identity against the pre-optimization boolean-mask implementation and measures the
wall-time win at the shape that surfaced the hotspot (n=2M, max_n=2000, cProfile'd via
``profile_one_combo.py --combo c0016_cbe1b080 --rows 2000000 --save-charts``: 6.162s tottime / 2
calls in ``_jackknife_metric``).

Usage:
    python profiling/bench_jackknife_slice_concat.py
"""

from __future__ import annotations

import time

import numpy as np


def _jackknife_metric_mask_baseline(y_true, y_pred, metric_fn, max_n=2000):
    """Pre-optimization boolean-mask implementation, kept here only as the A/B baseline."""
    n = y_true.shape[0]
    if n < 3:
        return None
    if n <= max_n:
        sel = np.arange(n)
    else:
        sel = np.linspace(0, n - 1, max_n).astype(np.int64)
    keep_mask = np.ones(n, dtype=bool)
    out = np.empty(sel.shape[0], dtype=np.float64)
    w = 0
    for i in sel:
        keep_mask[i] = False
        v = float(metric_fn(y_true[keep_mask], y_pred[keep_mask]))
        keep_mask[i] = True
        out[w] = v
        w += 1
    return out[:w]


def _mean_metric(a, b):
    return float(np.mean(a) - np.mean(b))


def main():
    rng = np.random.default_rng(0)
    n = 2_000_000
    max_n = 2000
    y_true = rng.normal(size=n)
    y_pred = rng.normal(size=n)

    from mlframe.evaluation._bootstrap_jackknife import _jackknife_metric

    t0 = time.perf_counter()
    out_new = _jackknife_metric(y_true, y_pred, _mean_metric, max_n=max_n)
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    out_old = _jackknife_metric_mask_baseline(y_true, y_pred, _mean_metric, max_n=max_n)
    t_old = time.perf_counter() - t0

    identical = np.array_equal(out_new, out_old)
    print(f"n={n:,} max_n={max_n}")
    print(f"old (boolean mask):     {t_old:.4f}s")
    print(f"new (slice concatenate): {t_new:.4f}s")
    print(f"speedup: {t_old / t_new:.2f}x")
    print(f"bit-identical: {identical}")
    assert identical, "slice-concatenate LOO must be bit-identical to the boolean-mask baseline"


if __name__ == "__main__":
    main()
