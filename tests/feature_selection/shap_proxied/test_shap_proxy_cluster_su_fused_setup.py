"""Fused SU clustering setup regression locks (iter71).

iter71 folded ``_column_marginal`` (per-column Python ``bincount``+normalize) and
``_pack_bins_for_kernel``'s per-column entropy Python loop into a single
``@njit(parallel=True)`` sweep (``_compute_marginals_packed``) driven by a Python
wrapper ``_setup_su_kernel_inputs``. Locks:

1. Fused builder produces marginals + entropies + constant_mask byte-equivalent
   to the prior two-pass path on a randomized fixture.
2. Cluster labels from the parallel kernel are unchanged vs the serial Python
   loop (parity contract preserved).
3. At ``n_features >= 1500`` the fused-setup path is at least 15% faster than a
   ref implementation that runs the prior two-pass setup.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import (
    _column_marginal,
    _compute_marginals_packed,
    _pack_bins_for_kernel,
    _pairwise_su_edges,
    _resolve_columns,
    _setup_su_kernel_inputs,
    cluster_correlated_features_su,
)


def _quantile_bin(col: np.ndarray, n_bins: int) -> np.ndarray:
    col = np.asarray(col, dtype=np.float64)
    if np.unique(col).size <= 1:
        return np.zeros_like(col, dtype=np.int32)
    qs = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
    if qs.size <= 1:
        return np.zeros_like(col, dtype=np.int32)
    edges = qs[1:-1] if qs.size > 2 else qs[1:]
    return np.clip(np.digitize(col, edges, right=False), 0, max(0, qs.size - 2)).astype(np.int32)


def _build_synthetic_bins(n_samples: int, n_features: int, n_bins: int, seed: int):
    rng = np.random.default_rng(seed)
    n_blocks = max(1, n_features // 6)
    blocks = []
    for _ in range(n_blocks):
        z = rng.standard_normal(n_samples)
        for _k in range(3):
            blocks.append(z + 0.2 * rng.standard_normal(n_samples))
    while len(blocks) < n_features:
        blocks.append(rng.standard_normal(n_samples))
    X = np.column_stack(blocks[:n_features])
    names = [f"f{i}" for i in range(n_features)]
    bins = {n: _quantile_bin(X[:, i], n_bins=n_bins) for i, n in enumerate(names)}
    return bins, names


def test_fused_setup_matches_two_pass_path():
    """``_setup_su_kernel_inputs`` matches the prior two-pass builder element-wise."""
    bins, names = _build_synthetic_bins(n_samples=400, n_features=37, n_bins=8, seed=11)
    _, arrays = _resolve_columns(bins, names)

    # New fused path
    fused = _setup_su_kernel_inputs(arrays, nbins_hints=None)
    assert fused is not None
    bins_p_new, nbins_new, freqs_new, offsets_new, h_new, const_new = fused

    # Old two-pass reference
    marginals = [_column_marginal(a) for a in arrays]
    old = _pack_bins_for_kernel(arrays, marginals)
    assert old is not None
    bins_p_old, nbins_old, freqs_old, offsets_old, h_old, const_old = old

    assert np.array_equal(bins_p_new, bins_p_old), "bins_packed buffer diverged from two-pass reference"
    assert np.array_equal(nbins_new, nbins_old), "nbins diverged"
    assert np.array_equal(offsets_new, offsets_old), "freqs_offsets diverged"
    # freqs are floats; allow a tiny eps for accumulation order differences.
    assert np.allclose(freqs_new, freqs_old, atol=1e-12, rtol=1e-12), "freqs_packed diverged"
    assert np.allclose(h_new, h_old, atol=1e-12, rtol=1e-12), "h_marginals diverged"
    assert np.array_equal(const_new, const_old), "constant_mask diverged"


def test_fused_setup_with_nbins_hints():
    """Per-feature nbins_hints are honored (cardinality matches max(observed_max+1, hint))."""
    # f0 only realizes bins {0, 1}; hint forces nb=5.
    bins = {
        "f0": np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32),
        "f1": np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int32),
        "f2": np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32),
    }
    arrays = [bins["f0"], bins["f1"], bins["f2"]]
    fused = _setup_su_kernel_inputs(arrays, nbins_hints=[5, 4, 2])
    assert fused is not None
    bins_p, nbins, freqs, offsets, h, const = fused

    assert nbins[0] == 5, f"f0 nb should be 5 (hint), got {nbins[0]}"
    assert nbins[1] == 4, f"f1 nb should be 4 (hint), got {nbins[1]}"
    # f2: only bin 0 realized, hint=2 -> nb=2; constant_mask should be True
    assert nbins[2] == 2, f"f2 nb should be 2 (hint), got {nbins[2]}"

    # Padded bin slots have probability 0 (no samples) - constancy check sees
    # only the realized bin counts.
    assert not const[0], "f0 has two distinct bins -> not constant"
    assert not const[1], "f1 has three distinct bins -> not constant"
    assert const[2], "f2 is constant (all zeros)"

    # Probabilities normalize over the realized bins, padded bins stay 0.
    assert math.isclose(freqs[offsets[0] + 0], 0.5), f"f0 bin0 should be 0.5"
    assert math.isclose(freqs[offsets[0] + 1], 0.5), f"f0 bin1 should be 0.5"
    assert freqs[offsets[0] + 2] == 0.0, "f0 padded bin should be 0"
    assert freqs[offsets[0] + 3] == 0.0, "f0 padded bin should be 0"
    assert freqs[offsets[0] + 4] == 0.0, "f0 padded bin should be 0"


def test_fused_setup_parity_with_kernel():
    """Cluster labels produced via fused-setup parallel path == serial Python loop."""
    bins, names = _build_synthetic_bins(n_samples=900, n_features=120, n_bins=8, seed=17)
    serial = cluster_correlated_features_su(
        bins,
        threshold=0.35,
        feature_names=names,
        use_parallel=False,
    )
    parallel = cluster_correlated_features_su(
        bins,
        threshold=0.35,
        feature_names=names,
        use_parallel=True,
        parallel_min_features=10,
    )
    assert np.array_equal(serial, parallel), "fused-setup parallel kernel diverges from serial loop at width=120"


def test_fused_setup_speedup_at_width_1500():
    """At width=1500, fused setup path beats two-pass setup ref by >=15%.

    Compares ONLY the setup phase (calls _setup_su_kernel_inputs vs the chain
    _column_marginal + _pack_bins_for_kernel). The downstream pairwise kernel
    is identical in both cases so we isolate the contribution of the fold.
    """
    import numba

    if numba.get_num_threads() < 2:
        pytest.skip(f"numba reports {numba.get_num_threads()} thread(s); parallel setup needs >=2 cores")

    width = 1500
    n_samples = 1500
    bins, names = _build_synthetic_bins(
        n_samples=n_samples,
        n_features=width,
        n_bins=10,
        seed=5,
    )
    _, arrays = _resolve_columns(bins, names)

    # Warmup numba kernel (first call triggers JIT compile).
    _ = _setup_su_kernel_inputs(arrays[:4], None)

    # New fused-setup timing
    t0 = time.perf_counter()
    fused = _setup_su_kernel_inputs(arrays, nbins_hints=None)
    t_new = time.perf_counter() - t0
    assert fused is not None

    # Old two-pass reference timing
    t0 = time.perf_counter()
    marginals = [_column_marginal(a) for a in arrays]
    old = _pack_bins_for_kernel(arrays, marginals)
    t_old = time.perf_counter() - t0
    assert old is not None

    speedup = t_old / max(t_new, 1e-9)
    assert speedup >= 1.15, f"fused setup did not beat two-pass by >=15%: new={t_new * 1000:.2f}ms, old={t_old * 1000:.2f}ms, speedup={speedup:.2f}x"


def test_compute_marginals_packed_handles_padded_constant_column():
    """Constant column with nb>1 (padded) is correctly marked constant + h=0."""
    # Synthesize bins_packed where column 0 is constant (all zeros) but nb=3.
    bins_packed = np.array([[0, 0, 0, 0]], dtype=np.int32)
    nbins = np.array([3], dtype=np.int64)
    freqs_offsets = np.array([0, 3], dtype=np.int64)
    freqs_packed = np.empty(3, dtype=np.float64)
    h_marginals = np.empty(1, dtype=np.float64)
    constant_mask = np.empty(1, dtype=np.bool_)

    _compute_marginals_packed(
        bins_packed,
        nbins,
        freqs_offsets,
        freqs_packed,
        h_marginals,
        constant_mask,
    )

    assert constant_mask[0], "single-bin column with padded nb must be marked constant"
    assert h_marginals[0] == 0.0, "entropy of constant column must be 0"
    assert freqs_packed[0] == 1.0, "realized bin probability must be 1.0"
    assert freqs_packed[1] == 0.0, "padded bin probability must be 0.0"
    assert freqs_packed[2] == 0.0, "padded bin probability must be 0.0"
