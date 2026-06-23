"""Bench: hoist the FIXED H(Z) / H(Y,Z) entropies out of the DCD swap
permutation-null loop (``evaluate_swap_candidate._run_member_null`` + the
aggregate null in ``_dcd_swap.py``).

WHY
---
Both swap permutation-null loops run B iterations that each call
``conditional_mi(x=permuted_col, y=target, z=S_minus_anchor, entropy_cache=None)``.
Across the B permutations ONLY the X column is shuffled -- y and z are
byte-for-byte FIXED. ``conditional_mi`` computes
``I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)``. H(Z) and H(Y,Z) depend
only on the (fixed) y and z columns, so they are IDENTICAL across all B
permutations -- yet OLD recomputes both every iteration (entropy_cache=None,
entropy_z=-1, entropy_yz=-1). That is 2 wasted merge_vars+entropy passes per
permutation * B permutations per swap-null.

NEW: compute entropy_z and entropy_yz ONCE before the B-loop (from any column
state -- the permuted column never participates in either), then pass them in
via the ``entropy_z=`` / ``entropy_yz=`` kwargs the function already exposes.
Bit-identical BY CONSTRUCTION: the same H(Z), H(Y,Z) values are reused; only
H(X,Z) and H(X,Y,Z) (which genuinely change with the permuted X) are recomputed.

Run: CUDA_VISIBLE_DEVICES="" python bench_dcd_swap_null_entropy_hoist.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from mlframe.feature_selection.filters.info_theory import conditional_mi


def _make_data(n, n_cols, n_bins, seed):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, n_bins, size=(n, n_cols)).astype(np.int32)
    nbins = np.full(n_cols, n_bins, dtype=np.int64)
    return data, nbins


def old_null(data, nbins, x_col, y_col, z_cols, B, seed):
    """Verbatim shape of the OLD swap-null inner loop."""
    rng = np.random.default_rng(seed)
    data_perm = data.copy()
    col_orig = data_perm[:, x_col].copy()
    x = np.array([x_col], dtype=np.int64)
    y = np.array([y_col], dtype=np.int64)
    z = np.array(z_cols, dtype=np.int64)
    n_exceed = 0
    obs = 0.5
    for _ in range(B):
        shuffled = col_orig.copy()
        rng.shuffle(shuffled)
        data_perm[:, x_col] = shuffled
        val = conditional_mi(
            factors_data=data_perm, x=x, y=y, z=z,
            var_is_nominal=None, factors_nbins=nbins,
            entropy_cache=None, can_use_x_cache=False, can_use_y_cache=False,
        )
        if val >= obs:
            n_exceed += 1
    return (n_exceed + 1) / (B + 1)


def new_null(data, nbins, x_col, y_col, z_cols, B, seed):
    """NEW: hoist the fixed H(Z) + H(Y,Z) out of the B-loop."""
    from mlframe.feature_selection.filters.info_theory import entropy, merge_vars
    rng = np.random.default_rng(seed)
    data_perm = data.copy()
    col_orig = data_perm[:, x_col].copy()
    x = np.array([x_col], dtype=np.int64)
    y = np.array([y_col], dtype=np.int64)
    z = np.array(z_cols, dtype=np.int64)
    # Hoisted, permutation-invariant entropies (depend only on fixed y/z cols).
    _, freqs_z, _ = merge_vars(data_perm, np.sort(z), None, nbins, dtype=np.int32)
    h_z = float(entropy(freqs_z))
    yz = np.sort(np.concatenate([y, z]))
    _, freqs_yz, _ = merge_vars(data_perm, yz, None, nbins, dtype=np.int32)
    h_yz = float(entropy(freqs_yz))
    n_exceed = 0
    obs = 0.5
    for _ in range(B):
        shuffled = col_orig.copy()
        rng.shuffle(shuffled)
        data_perm[:, x_col] = shuffled
        val = conditional_mi(
            factors_data=data_perm, x=x, y=y, z=z,
            var_is_nominal=None, factors_nbins=nbins,
            entropy_z=h_z, entropy_yz=h_yz,
            entropy_cache=None, can_use_x_cache=False, can_use_y_cache=False,
        )
        if val >= obs:
            n_exceed += 1
    return (n_exceed + 1) / (B + 1)


def bench():
    # Realistic swap-null shapes: n=600 (tau-auto small scene) and n=5000.
    for n in (600, 5000):
        for n_z in (3, 6):
            data, nbins = _make_data(n, 12, 8, seed=42)
            x_col, y_col = 0, 11
            z_cols = list(range(1, 1 + n_z))
            B = 199
            # warm
            old_null(data, nbins, x_col, y_col, z_cols, 3, 7)
            new_null(data, nbins, x_col, y_col, z_cols, 3, 7)
            # identity (same rng seed -> same permutations -> same p-value)
            p_old = old_null(data, nbins, x_col, y_col, z_cols, B, 123)
            p_new = new_null(data, nbins, x_col, y_col, z_cols, B, 123)
            N = 7
            t_old = min(_time(old_null, data, nbins, x_col, y_col, z_cols, B) for _ in range(N))
            t_new = min(_time(new_null, data, nbins, x_col, y_col, z_cols, B) for _ in range(N))
            print(f"n={n:5d} |z|={n_z} B={B}: OLD={t_old*1e3:8.2f}ms NEW={t_new*1e3:8.2f}ms "
                  f"speedup={t_old/t_new:5.2f}x  p_old={p_old:.4f} p_new={p_new:.4f} "
                  f"identical={p_old == p_new}")


def _time(fn, data, nbins, x_col, y_col, z_cols, B):
    t0 = time.perf_counter()
    fn(data, nbins, x_col, y_col, z_cols, B, 123)
    return time.perf_counter() - t0


if __name__ == "__main__":
    bench()
