"""The pair-level residual test of the sufficient-summary early stop.

The marginal test sees no information in a product or an XOR, so the early stop used to switch the pair search off whenever the selected set already
explained most of y. These tests pin the pair-aware stop, the equivalence of the kernel versions and of the streamed and GPU paths, and the memory bound.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from mlframe.feature_selection.filters import _fe_sufficient_summary_pairs as sp
from mlframe.feature_selection.filters._fe_sufficient_summary import sufficient_summary_reached

K, KR = sp.PAIR_BINS, 10


def _problem(n: int, p: int, seed: int = 0):
    """Coarse int8 codes ``(p, n)`` and a residual that depends on the pair ``(3, 7)``."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, K, size=(p, n)).astype(np.int8)
    r = rng.integers(0, KR, size=n).astype(np.int64)
    r[: n // 2] = (codes[3, : n // 2].astype(np.int64) * 2 + codes[7, : n // 2]) % KR
    return codes, r


def test_kernel_versions_agree_bit_for_bit():
    """The first kernel and the load-balanced one return the same best MI and the same partner for every row."""
    codes, r = _problem(3000, 30)
    a = sp._pair_mi_best_per_first_v1(codes, r, K, KR)
    b = sp._pair_mi_best_per_first(codes, r, K, KR)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


@pytest.mark.parametrize("block_cols", [4, 7, 30, 64])
def test_streamed_blocks_find_the_same_best_pair_as_the_full_kernel(block_cols):
    """Scoring block pair by block pair covers every pair exactly once, whatever the block width."""
    codes, r = _problem(3000, 30)
    full_best, full_j = sp._pair_mi_best_per_first_v1(codes, r, K, KR)
    i = int(np.argmax(full_best))
    mi, pair = sp.best_pair_streamed(lambda c0, c1: codes[c0:c1], 30, r, K, KR, "njit", block_cols)
    assert mi == pytest.approx(float(full_best[i]), rel=1e-12)
    assert pair == (i, int(full_j[i])) == (3, 7)


@pytest.mark.parametrize("backend", ["gpu_fused", "gpu_dispatch"])
@pytest.mark.parametrize("block_cols", [16, 64])
def test_gpu_backends_match_the_cpu_backend(backend, block_cols):
    """The fused cupy kernel and the generic dispatcher path agree with the CPU kernel on the best pair and its MI (skipped without a GPU)."""
    from mlframe.feature_selection.filters.batch_pair_mi_gpu import _CUDA_AVAIL, _CUPY_AVAIL

    if backend == "gpu_fused" and not sp._fused_gpu_available():
        pytest.skip("no usable cupy device on this host")
    if backend == "gpu_dispatch" and not (_CUDA_AVAIL or _CUPY_AVAIL):
        pytest.skip("no GPU backend on this host")
    codes, r = _problem(4000, 40)
    cpu = sp.best_pair_streamed(lambda c0, c1: codes[c0:c1], 40, r, K, KR, "njit", block_cols)
    gpu = sp.best_pair_streamed(lambda c0, c1: codes[c0:c1], 40, r, K, KR, backend, block_cols)
    assert gpu[1] == cpu[1]
    # The fused kernel sums in float64 but takes its per-cell terms in the float32 crit type by default (the strict float64 comparison is the REAL test below);
    # the generic dispatcher keeps the target frequencies in the float32 crit dtype.
    assert gpu[0] == pytest.approx(cpu[0], rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("real", ["float", "double"])
def test_fused_kernel_is_type_generic_over_real(real):
    """One kernel source, compiled once per ``{REAL}`` type: both agree with the float64 CPU kernel on the best pair; ``float`` within its rounding, ``double`` tightly."""
    if not sp._fused_gpu_available():
        pytest.skip("no usable cupy device on this host")
    codes, r = _problem(4000, 40)
    cpu = sp.best_pair_streamed(lambda c0, c1: codes[c0:c1], 40, r, K, KR, "njit", 16)
    gpu = sp.best_pair_streamed(lambda c0, c1: codes[c0:c1], 40, r, K, KR, "gpu_fused", 16, real=real)
    assert gpu[1] == cpu[1]
    assert gpu[0] == pytest.approx(cpu[0], rel=1e-9 if real == "double" else 1e-5)


@pytest.mark.parametrize("real", ["float", "double"])
def test_fused_kernel_versions_agree(real):
    """Version 1 (a thread per cell group, tree reduction) and version 2 (a thread per cell, warp-shuffle reduction) return the same best pair and the same MI."""
    if not sp._fused_gpu_available():
        pytest.skip("no usable cupy device on this host")
    codes, r = _problem(4000, 40)
    results = {}
    old = sp.FUSED_VERSION
    try:
        for version in ("v1", "v2"):
            sp.FUSED_VERSION = version
            results[version] = sp.best_pair_streamed(lambda c0, c1: codes[c0:c1], 40, r, K, KR, "gpu_fused", 16, real=real)
    finally:
        sp.FUSED_VERSION = old
    assert results["v1"][1] == results["v2"][1] == (3, 7)
    assert results["v1"][0] == pytest.approx(results["v2"][0], rel=1e-12 if real == "double" else 1e-6)


def test_fallback_heuristic_matches_the_measured_crossover(monkeypatch):
    """Before the cache holds a sweep the choice follows the measured crossover: the CPU for small pairs x rows, the GPU above about 1e6 (when a GPU is usable)."""
    monkeypatch.setattr(sp, "_fused_gpu_available", lambda: True)
    assert sp._pair_residual_fallback_choice(1_500, 100) == "njit"
    assert sp._pair_residual_fallback_choice(5_000, 100) == "njit"
    assert sp._pair_residual_fallback_choice(20_000, 100) == "gpu_fused"
    assert sp._pair_residual_fallback_choice(1_500, 1_000) == "gpu_fused"
    assert sp._pair_residual_fallback_choice(100_000, 500_000) == "gpu_fused"
    monkeypatch.setattr(sp, "_fused_gpu_available", lambda: False)
    assert sp._pair_residual_fallback_choice(100_000, 500_000) == "njit"


def test_equivalence_tolerance_follows_the_real_type():
    """The tolerance is derived from the machine epsilon of the ``{REAL}`` type: about 8 eps for float32, the 1e-9 floor for double."""
    f_rtol, f_atol = sp.pair_equiv_tolerance("float")
    d_rtol, d_atol = sp.pair_equiv_tolerance("double")
    assert f_rtol == pytest.approx(8 * float(np.finfo(np.float32).eps)) and f_rtol > 100 * d_rtol
    assert d_rtol == 1e-9 and 0 < d_atol < d_rtol and 0 < f_atol < f_rtol


@pytest.mark.parametrize("real", ["float", "double"])
def test_the_sweep_tolerance_keeps_the_gpu_variant(real):
    """The sweep drops a variant that diverges from the CPU reference by more than its tolerance; the GPU result must stay inside the one for its ``{REAL}`` type."""
    if not sp._fused_gpu_available():
        pytest.skip("no usable cupy device on this host")
    codes, r, k, kr = sp._make_pair_residual_inputs({"n_samples": 5000, "n_pairs": 5000})
    p = codes.shape[0]
    cpu = sp.best_pair_streamed(lambda c0, c1: codes[c0:c1], p, r, k, kr, "njit", sp.BLOCK_COLS)[0]
    gpu = sp.best_pair_streamed(lambda c0, c1: codes[c0:c1], p, r, k, kr, "gpu_fused", sp.BLOCK_COLS, real=real)[0]
    rtol, atol = sp.pair_equiv_tolerance(real)
    assert abs(gpu - cpu) <= atol + rtol * abs(cpu), (real, gpu, cpu)


def test_kernel_sources_have_no_bare_floating_point_type():
    """Every kernel source uses the ``{REAL}`` / ``{ACC}`` placeholders only, so the precision is chosen in one place (the contract check reads the same sources)."""
    import re

    assert sp._FUSED_NAMES
    for src, _name in sp._FUSED_NAMES.values():
        assert "{REAL}" in src and "{ACC}" in src
        assert not re.search(r"\b(?:double|float)\b", src)


def test_fused_kernel_covers_every_pair_exactly_once_on_the_device():
    """The best pair is found wherever it sits relative to the block boundaries, so no block pair is skipped or scored twice (skipped without a GPU)."""
    if not sp._fused_gpu_available():
        pytest.skip("no usable cupy device on this host")
    for pair in [(0, 1), (0, 39), (15, 16), (16, 17), (38, 39)]:
        rng = np.random.default_rng(5)
        n, p = 3000, 40
        codes = rng.integers(0, K, size=(p, n)).astype(np.int8)
        r = rng.integers(0, KR, size=n).astype(np.int64)
        r[: n // 2] = (codes[pair[0], : n // 2].astype(np.int64) * 2 + codes[pair[1], : n // 2]) % KR
        found = sp.best_pair_streamed(lambda c0, c1, codes=codes: codes[c0:c1], p, r, K, KR, "gpu_fused", 16)[1]
        assert found == pair, (pair, found)


def test_block_coarsening_never_builds_an_n_by_p_int64_matrix():
    """The raw codes are coarsened one column at a time into int8 blocks, so the peak stays far below one n x p int64 matrix."""
    n, p = 3000, 800
    rng = np.random.default_rng(1)
    raw = rng.integers(0, 10, size=(n, p)).astype(np.int64)
    r = rng.integers(0, 10, size=n).astype(np.int64)
    levels = np.full(p, 10, dtype=np.int64)
    get_block = sp._coarse_block_getter(raw, levels, K, slice(None, None, 1))
    tracemalloc.start()
    sp.best_pair_streamed(get_block, p, r, K, KR, "njit", 64)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 0.25 * n * p * 8, f"peak {peak} bytes against one int64 matrix of {n * p * 8}"


def test_too_many_pairs_for_the_row_budget_is_not_certified():
    """When the pair count leaves too few rows inside the work budget the test is withheld (None), so the caller does not stop the FE search."""
    n, p = 1600, 2600
    raw = np.zeros((n, p), dtype=np.int64)
    res = sp.pair_residual_test(raw, np.full(p, 10, dtype=np.int64), np.zeros(n, dtype=np.int64), backend="njit")
    assert res is None


def test_pair_test_finds_the_planted_pair_and_ignores_noise():
    """On codes with a planted interaction the best pair is the planted one and clears the ceiling; on pure noise it does not."""
    n, p = 4000, 12
    rng = np.random.default_rng(2)
    raw = rng.integers(0, 10, size=(n, p)).astype(np.int64)
    nbins = np.full(p, 10, dtype=np.int64)
    planted = rng.integers(0, 10, size=n).astype(np.int64)
    planted[: n // 2] = (raw[: n // 2, 3] // 2 * 2 + raw[: n // 2, 7] // 2) % 10
    mi, ceiling, pair = sp.pair_residual_test(raw, nbins, planted, backend="njit")
    assert pair == (3, 7) and mi > ceiling
    noise = rng.integers(0, 10, size=n).astype(np.int64)
    mi, ceiling, _pair = sp.pair_residual_test(raw, nbins, noise, backend="njit")
    assert mi <= ceiling


def _bin10(v):
    """Equi-frequency 10-bin codes, the resolution the selector sees."""
    edges = np.quantile(np.asarray(v, dtype=np.float64), np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(edges, v).astype(np.int64)


def test_early_stop_is_withheld_when_the_residual_holds_a_balanced_product():
    """y = s + 0.5*sign(a*b) with s selected: every marginal MI with the residual is near zero (the sign product is balanced), yet the pair (a, b) carries it, so the stop must not fire."""
    n = 4000
    rng = np.random.default_rng(3)
    s, a, b = (rng.normal(size=n) for _ in range(3))
    noise = [rng.normal(size=n) for _ in range(4)]
    # Noise keeps the order of the residual inside each sign cluster from being decided by the ridge slack on s alone, as it would be in an exact toy.
    y = s + 0.45 * np.sign(a * b) + 0.2 * rng.normal(size=n)
    cols = [s, a, b, *noise, y]
    data = np.column_stack([_bin10(c) for c in cols]).astype(np.int64)
    names = ["s", "a", "b", "n0", "n1", "n2", "n3", "y"]
    verdict = sufficient_summary_reached(
        data=data,
        nbins=np.full(data.shape[1], 10, dtype=np.int64),
        y_continuous=y,
        target_col_idx=len(cols) - 1,
        selected_cols_idx=[0],
        selected_continuous={"s": s},
        raw_cols_idx=list(range(len(cols) - 1)),
        cols_names=names,
        quantization_nbins=10,
        random_seed=0,
    )
    assert not verdict.reached, verdict.reason
    assert verdict.blocking_pair is not None and set(verdict.blocking_pair) == {1, 2}, verdict.reason
