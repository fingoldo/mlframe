"""Bit-identity tests for ``batch_mi_with_noise_gate`` vs per-column ``mi_direct``.

The batched FE-candidate MI + permutation noise-gate kernel must reproduce the
EXACT ``fe_mi`` a per-candidate ``mi_direct`` loop produces on the default FE path
(``parallelism='outer'``, ``n_workers=1`` -> ``parallel_mi_prange``, ``base_seed=0``).
Bit-identity is non-negotiable: any drift changes which engineered features MRMR keeps.

Covered: varied n / K / nbins / npermutations in {0, 3, 10} /
min_nonzero_confidence in {0.99, 0.0}, plus tie-heavy and pure-noise (rejection-
triggering) columns that must be zeroed identically.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory import (
    batch_mi_with_noise_gate,
    merge_vars,
)
from mlframe.feature_selection.filters.permutation import mi_direct


def _make_frame(n, K, nbins, seed):
    """Build a discretized (n, K) int frame with a MIX of column types:
    informative (correlated with y), tie-heavy (near-constant), and pure noise.
    Returns (disc_2d, classes_y, freqs_y).
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, size=n).astype(np.int32)
    cols = np.empty((n, K), dtype=np.int32)
    for k in range(K):
        kind = k % 4
        if kind == 0:
            # Informative: y plus a little noise, clipped into nbins.
            c = (y + rng.integers(0, 2, size=n)) % nbins
        elif kind == 1:
            # Pure noise.
            c = rng.integers(0, nbins, size=n)
        elif kind == 2:
            # Tie-heavy: mostly one bin, a few others.
            c = np.zeros(n, dtype=np.int64)
            idx = rng.choice(n, size=max(1, n // 20), replace=False)
            c[idx] = rng.integers(1, nbins, size=idx.size)
        else:
            # Strongly informative (y mapped, occasional flip).
            c = y.copy().astype(np.int64)
            flip = rng.choice(n, size=max(1, n // 10), replace=False)
            c[flip] = rng.integers(0, nbins, size=flip.size)
        cols[:, k] = (c % nbins).astype(np.int32)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1),
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64),
        dtype=np.int32,
    )
    return cols, classes_y, freqs_y


def _per_column_reference(disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y, npermutations, min_nonzero_confidence):
    """The ORIGINAL per-candidate path: loop mi_direct over columns exactly as the
    FE Phase-3 batch loop does (base_seed=0, parallelism='outer', n_workers=1,
    prefer_gpu False to keep the deterministic CPU path)."""
    K = disc_2d.shape[1]
    out = np.empty(K, dtype=np.float64)
    for ci in range(K):
        fe_mi, _ = mi_direct(
            disc_2d[:, ci].reshape(-1, 1),
            x=np.array([0], dtype=np.int64),
            y=None,
            factors_nbins=np.array([int(factors_nbins[ci])], dtype=np.int64),
            classes_y=classes_y,
            classes_y_safe=classes_y_safe,
            freqs_y=freqs_y,
            min_nonzero_confidence=min_nonzero_confidence,
            npermutations=npermutations,
            prefer_gpu=False,
        )
        out[ci] = fe_mi
    return out


@pytest.mark.parametrize("n,K,nbins", [(200, 8, 4), (500, 13, 6), (1000, 20, 5)])
@pytest.mark.parametrize("npermutations", [0, 3, 10])
@pytest.mark.parametrize("min_nonzero_confidence", [0.99, 0.0])
def test_batch_bit_identical_to_mi_direct(n, K, nbins, npermutations, min_nonzero_confidence):
    """batch_mi_with_noise_gate matches the per-column mi_direct reference bit-for-bit across the (n, K, nbins) grid."""
    disc_2d, classes_y, freqs_y = _make_frame(n, K, nbins, seed=1234 + n + K + nbins)
    classes_y_safe = classes_y.copy()
    factors_nbins = np.full(K, nbins, dtype=np.int64)

    ref = _per_column_reference(
        disc_2d, factors_nbins, classes_y, classes_y_safe, freqs_y,
        npermutations, min_nonzero_confidence,
    )
    got = batch_mi_with_noise_gate(
        disc_2d=disc_2d,
        factors_nbins=factors_nbins,
        classes_y=classes_y,
        classes_y_safe=classes_y_safe,
        freqs_y=freqs_y,
        npermutations=npermutations,
        base_seed=np.uint64(0),
        min_nonzero_confidence=float(min_nonzero_confidence),
        use_su=False,
        dtype=np.int32,
    )

    assert got.shape == ref.shape
    # EXACT float equality -- bit-identity is the contract.
    assert np.array_equal(got, ref), (
        f"mismatch n={n} K={K} nbins={nbins} nperm={npermutations} " f"mnc={min_nonzero_confidence}\n ref={ref}\n got={got}\n diff={got - ref}"
    )


def test_pure_noise_zeroed_identically():
    """A column of pure noise must be rejected (-> 0.0) by BOTH paths identically."""
    n = 800
    rng = np.random.default_rng(99)
    y = rng.integers(0, 4, size=n).astype(np.int32)
    noise = rng.integers(0, 6, size=n).astype(np.int32)
    disc_2d = noise.reshape(-1, 1)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1),
        vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None,
        factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64),
        dtype=np.int32,
    )
    factors_nbins = np.array([6], dtype=np.int64)
    ref = _per_column_reference(disc_2d, factors_nbins, classes_y, classes_y.copy(), freqs_y, npermutations=10, min_nonzero_confidence=0.99)
    got = batch_mi_with_noise_gate(
        disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
        classes_y_safe=classes_y.copy(), freqs_y=freqs_y, npermutations=10,
        base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=False, dtype=np.int32,
    )
    assert np.array_equal(got, ref)


def test_su_mode_bit_identical(monkeypatch):
    """When SU normalization is active, batched use_su=True matches per-column mi_direct."""
    import mlframe.feature_selection.filters.info_theory as it
    monkeypatch.setattr(it, "use_su_normalization", lambda: True)
    import mlframe.feature_selection.filters.permutation as perm
    monkeypatch.setattr(perm, "use_su_normalization", lambda: True)

    disc_2d, classes_y, freqs_y = _make_frame(400, 10, 5, seed=7)
    factors_nbins = np.full(10, 5, dtype=np.int64)
    ref = _per_column_reference(disc_2d, factors_nbins, classes_y, classes_y.copy(), freqs_y, npermutations=10, min_nonzero_confidence=0.99)
    got = batch_mi_with_noise_gate(
        disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
        classes_y_safe=classes_y.copy(), freqs_y=freqs_y, npermutations=10,
        base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=True, dtype=np.int32,
    )
    assert np.array_equal(got, ref)


# ---- F2 fused observed-MI kernel (v2) regression (2026-06-22) ---------------------------------------
# ``batch_mi_with_noise_gate_v2`` fuses the per-column dense-code write with the observed-MI joint
# accumulation (one n-row pass instead of two). It MUST be bit-identical to v1 (and thus to the per-column
# mi_direct reference) -- any drift would change which engineered features MRMR keeps. These pin the
# bit-identity across the same n/K/nbins/npermutations/min_nonzero_confidence grid + SU + pure-noise.
from mlframe.feature_selection.filters.info_theory import (
    batch_mi_with_noise_gate_v2,
    select_batch_mi_kernel,
)


@pytest.mark.parametrize("n,K,nbins", [(200, 8, 4), (500, 13, 6), (1000, 20, 5)])
@pytest.mark.parametrize("npermutations", [0, 3, 10])
@pytest.mark.parametrize("min_nonzero_confidence", [0.99, 0.0])
def test_v2_bit_identical_to_v1(n, K, nbins, npermutations, min_nonzero_confidence):
    """The fused v2 kernel matches v1 bit-for-bit across the (n, K, nbins) grid."""
    disc_2d, classes_y, freqs_y = _make_frame(n, K, nbins, seed=1234 + n + K + nbins)
    factors_nbins = np.full(K, nbins, dtype=np.int64)
    kw = dict(
        disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
        classes_y_safe=classes_y.copy(), freqs_y=freqs_y, npermutations=npermutations,
        base_seed=np.uint64(0), min_nonzero_confidence=float(min_nonzero_confidence),
        use_su=False, dtype=np.int32,
    )
    v1 = batch_mi_with_noise_gate(**kw)
    v2 = batch_mi_with_noise_gate_v2(**kw)
    assert np.array_equal(v1, v2), f"v2 != v1 n={n} K={K} nbins={nbins} nperm={npermutations} mnc={min_nonzero_confidence}"


def test_v2_su_mode_bit_identical():
    """The fused v2 kernel matches v1 bit-for-bit under SU normalization."""
    disc_2d, classes_y, freqs_y = _make_frame(400, 10, 5, seed=7)
    factors_nbins = np.full(10, 5, dtype=np.int64)
    kw = dict(
        disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
        classes_y_safe=classes_y.copy(), freqs_y=freqs_y, npermutations=10,
        base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=True, dtype=np.int32,
    )
    assert np.array_equal(batch_mi_with_noise_gate(**kw), batch_mi_with_noise_gate_v2(**kw))


def test_v2_pure_noise_zeroed_identically():
    """The fused v2 kernel matches v1 bit-for-bit on a pure-noise column (both zero it out)."""
    n = 800
    rng = np.random.default_rng(99)
    y = rng.integers(0, 4, size=n).astype(np.int32)
    disc_2d = rng.integers(0, 6, size=n).astype(np.int32).reshape(-1, 1)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=y.reshape(-1, 1), vars_indices=np.array([0], dtype=np.int64),
        var_is_nominal=None, factors_nbins=np.array([int(y.max()) + 1], dtype=np.int64), dtype=np.int32,
    )
    factors_nbins = np.array([6], dtype=np.int64)
    kw = dict(
        disc_2d=disc_2d, factors_nbins=factors_nbins, classes_y=classes_y,
        classes_y_safe=classes_y.copy(), freqs_y=freqs_y, npermutations=10,
        base_seed=np.uint64(0), min_nonzero_confidence=0.99, use_su=False, dtype=np.int32,
    )
    assert np.array_equal(batch_mi_with_noise_gate(**kw), batch_mi_with_noise_gate_v2(**kw))


def test_select_batch_mi_kernel_env_override(monkeypatch):
    """The env override forces a specific kernel; default returns a callable kernel."""
    monkeypatch.setenv("MLFRAME_BATCH_MI_KERNEL", "v1")
    assert select_batch_mi_kernel(30_000, 600) is batch_mi_with_noise_gate
    monkeypatch.setenv("MLFRAME_BATCH_MI_KERNEL", "v2")
    assert select_batch_mi_kernel(30_000, 600) is batch_mi_with_noise_gate_v2
    monkeypatch.delenv("MLFRAME_BATCH_MI_KERNEL", raising=False)
    assert callable(select_batch_mi_kernel(30_000, 600))


def test_sweep_uses_kernel_choice_as_the_decision_key(monkeypatch):
    """Regression (2026-07-16): _run_batch_mi_kernel_sweep called sweep_backend_grid(result_key=...), a
    kwarg pyutilz's sweep_backend_grid has never accepted (the real parameter is decision_key) -- every
    call raised TypeError, silently caught by a fallback re-call that OMITTED the key entirely, so every
    result region stored its winner under the default "backend_choice" key while select_batch_mi_kernel
    always reads "kernel_choice" (always missing -> always silently defaulted to "v2"). A per-host tuning
    sweep that measured v1 as the winner for some shape was therefore SILENTLY IGNORED: the
    kernel_tuning_cache path never actually influenced which kernel ran.

    Spies on the real sweep_backend_grid call to verify the decision_key it's invoked with, so this
    fails pre-fix (result_key was passed, decision_key wasn't -- or the call needed 2 tries) and passes
    post-fix (decision_key="kernel_choice" passed directly, first try)."""
    from mlframe.feature_selection.filters.info_theory import _batch_kernels as bk
    import pyutilz.dev.benchmarking as _bench

    calls = []
    orig = _bench.sweep_backend_grid

    def _spy(*a, **kw):
        """Record the kwargs sweep_backend_grid was called with, then short-circuit with a canned result."""
        calls.append(kw)
        # Short-circuit the real (slow) sweep -- we only need to observe the call signature.
        return [{"n_rows_max": None, "kernel_choice": "v1"}]

    monkeypatch.setattr(_bench, "sweep_backend_grid", _spy)
    try:
        bk._run_batch_mi_kernel_sweep()
    finally:
        monkeypatch.setattr(_bench, "sweep_backend_grid", orig)

    assert len(calls) == 1, f"sweep_backend_grid should be called exactly once (no TypeError-retry dead path); got {len(calls)} calls"
    assert calls[0].get("decision_key") == "kernel_choice", (
        f"sweep_backend_grid must be called with decision_key='kernel_choice' (pyutilz's real parameter "
        f"name) so select_batch_mi_kernel's res.get('kernel_choice', ...) actually finds the tuned "
        f"winner instead of silently defaulting to v2; got kwargs={calls[0]!r}"
    )
    assert "result_key" not in calls[0], "result_key is not a real sweep_backend_grid parameter -- it always raised TypeError"
