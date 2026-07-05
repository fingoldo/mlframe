"""P-10 bench: measure the fused FE MI kernel's single-vs-split-N crossover, motivating the KTC lookup.

The prior code hardcoded ``K < 48 and n >= 262144`` (a 6-SM GTX 1050 Ti point) to choose the split-N launch.
This sweeps single vs split (forced via ``MLFRAME_FE_MI_SPLIT``) across a (n, k) grid and prints which leg
wins per cell -- showing the hardcoded predicate leaves wins on the table on other hardware, which is exactly
why ``lookup_fe_mi_split_backend`` now consults the per-host ``kernel_tuning_cache`` (constants = fallback).

Measured (RTX 500 Ada, cupy 14.1, warm median of >=3, n_iters=1):
  n=50k   k=8   -> split  1.36 vs single 1.99 ms   (fallback WRONGLY picks single: n<262144)
  n=200k  k=8   -> split  0.70 vs single 0.88 ms   (fallback WRONGLY picks single: n<262144)
  n=500k  k=200 -> split 14.88 vs single 15.36 ms  (fallback WRONGLY picks single: k>=48)
  n=1M    k=8   -> split  1.96 vs single 4.36 ms   (2.2x; fallback correctly picks split)
  n=1M    k=64  -> single  9.32 vs split 10.66 ms  (fallback correctly picks single)
Both legs are bit-identical (selection-equivalent); the sweep only reorders launches.

Run: PYTHONPATH=src python -m mlframe.feature_selection._benchmarks.bench_fe_mi_split_crossover_ktc
"""
from __future__ import annotations

from mlframe.feature_selection._benchmarks.kernel_tuning_cache._auto_tune_sweeps_b import _run_sweep_fe_mi_split


def main() -> None:
    regions = _run_sweep_fe_mi_split(n_iters=1)
    print(f"{len(regions)} regions:")
    for r in regions:
        if r["n_samples_max"] is None:
            print(f"  catch-all -> {r['backend_choice']}")
            continue
        print(f"  n<={r['n_samples_max']:>9} k<={r['k_max']:>4} -> {r['backend_choice']:>6} " f"(single={r['single_ms']} split={r['split_ms']} ms)")


if __name__ == "__main__":
    main()
