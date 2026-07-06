"""Bench for greedy_cmi_fe per-step y/z hoist (CPX10).

The greedy loop scored every remaining candidate with _cmi_from_binned, which
re-renumbers yz and re-bins z each call though y_bin and z_joint are fixed
within a step. Hoisting precompute_cmi_yz_terms once per step and scoring via
cmi_from_binned_fixed_yz reuses the y/z block. Bit-identical CMI.
Run: python this.py
"""
from __future__ import annotations

import time

import numpy as np


def main():
    from mlframe.feature_selection.filters._mi_greedy_cmi_fe import (
        _cmi_from_binned,
        cmi_from_binned_fixed_yz,
        precompute_cmi_yz_terms,
    )

    rng = np.random.default_rng(0)
    n, ncand = 2000, 300
    y = rng.integers(0, 4, n).astype(np.int64)
    z = rng.integers(0, 40, n).astype(np.int64)
    cands = [rng.integers(0, 8, n).astype(np.int64) for _ in range(ncand)]

    def old():
        return [_cmi_from_binned(c, y, z) for c in cands]

    def new():
        yi, zi, h_yz, h_z, k_yz, k_z, nn = precompute_cmi_yz_terms(y, z)
        return [cmi_from_binned_fixed_yz(c, yi, zi, h_yz, h_z, k_yz, k_z, nn) for c in cands]

    old(); new()
    assert np.array_equal(np.asarray(old()), np.asarray(new())), "identity"  # nosec B101 - internal invariant check in src/mlframe/feature_selection/filters/_benchmarks, not reachable with untrusted input

    def _best(fn, reps=7):
        t = []
        for _ in range(reps):
            s = time.perf_counter(); fn(); t.append(time.perf_counter() - s)
        return min(t)

    t_old = _best(old)
    t_new = _best(new)
    print(f"per-step scan {ncand} cands n={n}: OLD {t_old*1e3:.2f}ms -> " f"NEW {t_new*1e3:.2f}ms ({t_old/t_new:.2f}x) identity OK")


if __name__ == "__main__":
    main()
