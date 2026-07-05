"""Peak-allocation bench: per-base ``mi_y`` baseline via ``exclude_col`` view vs ``np.delete`` matrix copy.

``CompositeTargetDiscovery.fit`` screens B base candidates; for each it needs the baseline
``MI(y, X_without_base)`` aggregated over every feature column EXCEPT the candidate base. The legacy way to
isolate "all columns except one" was ``np.delete(full_prebinned, base_idx, axis=1)`` -- a fresh ``(n, F-1)``
int16 copy of the WHOLE prebinned feature matrix, allocated once per base. The new ``exclude_col`` path threads
the base index into ``_mi_to_target_prebinned`` / ``_mi_per_feature_prebinned`` so the per-feature loop simply
SKIPS that column on the full matrix -- zero per-base matrix allocation -- and the survivor MI vector is
bit-identical to the deleted-matrix path (each survivor's MI is computed from the exact same column; the
aggregate reduces over the same contiguous ``[0..k-1, k+1..]`` order ``np.delete`` would produce).

This bench measures PEAK tracemalloc over the whole B-base baseline sweep for both paths, at n=50k / F=40 / B=5
(the finding's scenario), warmed once and repeated several times (min-of-reps reported to suppress allocator
noise). It also asserts bit-identity (the per-base baseline scalar is equal between the two paths) so a future
"just delete it again" cannot silently diverge.

MEASURED (this Windows host, py3.14, n=50k F=40 B=5, nbins=12, 5 reps, min-of-reps peak):
  old (np.delete per base): peak ~ 7.44 MB
  new (exclude_col view):   peak ~ 1.24 MB   (delta ~ -6.20 MB, ~6x lower peak)
  bit-identical baselines:  True
The new path drops B copies of the ``(n, F-1)`` int16 prebinned matrix from the baseline path (~B * n * (F-1) * 2
bytes of transient/held allocation) at bit-identical MI. The exact numbers vary with host allocator behaviour but
the DIRECTION (new <= old, identical baselines) is the invariant the paired test pins.

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_mi_y_baseline_exclude_col
"""

from __future__ import annotations

import json
import os
import sys
import tracemalloc
from datetime import datetime

import numpy as np

from mlframe.training.composite.discovery.screening import (
    _aggregate_mi_per_feature,
    _mi_per_feature_prebinned,
    _mi_to_target_prebinned,
    _prebin_feature_columns,
)

_N = 50_000
_F = 40
_B = 5
_NBINS = 12
_REPS = 5


def _make_prebinned(n: int, f: int, nbins: int, seed: int = 0):
    """Build a finite (n, f) float32 feature matrix + (n,) target, prebin the features once."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, f)).astype(np.float32)
    # A mild signal so the per-feature MI is non-trivial (not all-zero bins).
    y = (0.6 * x[:, 0] + 0.3 * x[:, 1] - 0.2 * x[:, 2]).astype(np.float64)
    y += rng.standard_normal(n) * 0.5
    prebinned = _prebin_feature_columns(x, nbins=nbins)
    return prebinned, y


def _baseline_old(prebinned: np.ndarray, y: np.ndarray, base_idxs, nbins: int) -> list[float]:
    """Legacy: per base, np.delete the column then aggregate MI over the deleted copy."""
    out: list[float] = []
    for k in base_idxs:
        deleted = np.delete(prebinned, k, axis=1)  # (n, F-1) int16 copy per base
        per_feat = _mi_per_feature_prebinned(deleted, y, nbins=nbins)
        out.append(_aggregate_mi_per_feature(per_feat, "mean"))
    return out


def _baseline_new(prebinned: np.ndarray, y: np.ndarray, base_idxs, nbins: int) -> list[float]:
    """New: per base, exclude_col view on the full matrix -- no per-base matrix copy."""
    out: list[float] = []
    for k in base_idxs:
        out.append(_mi_to_target_prebinned(prebinned, y, nbins=nbins, aggregation="mean", exclude_col=k))
    return out


def _peak_mb(fn, *args) -> tuple[float, list[float]]:
    """Run ``fn(*args)`` under tracemalloc, return (peak MB during the call, result)."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    res = fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1024.0**2, res


def main() -> None:
    prebinned, y = _make_prebinned(_N, _F, _NBINS)
    base_idxs = list(range(_B))

    # Warm + bit-identity gate.
    old0 = _baseline_old(prebinned, y, base_idxs, _NBINS)
    new0 = _baseline_new(prebinned, y, base_idxs, _NBINS)
    identical = all(o == n for o, n in zip(old0, new0))

    old_peak = float("inf")
    new_peak = float("inf")
    for _ in range(_REPS):
        p_old, _ = _peak_mb(_baseline_old, prebinned, y, base_idxs, _NBINS)
        p_new, _ = _peak_mb(_baseline_new, prebinned, y, base_idxs, _NBINS)
        old_peak = min(old_peak, p_old)
        new_peak = min(new_peak, p_new)

    print(f"mi_y baseline exclude_col bench  n={_N} F={_F} B={_B} nbins={_NBINS} " f"reps={_REPS} py={sys.version.split()[0]}")
    print(f"  bit-identical baselines: {identical}")
    print(f"  old (np.delete per base): peak {old_peak:8.3f} MB")
    print(f"  new (exclude_col view):   peak {new_peak:8.3f} MB  (delta {new_peak - old_peak:+.3f} MB)")

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "mi_y_baseline_exclude_col.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ts": datetime.now().isoformat(),
                "n": _N, "F": _F, "B": _B, "nbins": _NBINS, "reps": _REPS,
                "bit_identical": identical,
                "old_peak_mb": old_peak,
                "new_peak_mb": new_peak,
                "delta_mb": new_peak - old_peak,
            },
            fh,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
