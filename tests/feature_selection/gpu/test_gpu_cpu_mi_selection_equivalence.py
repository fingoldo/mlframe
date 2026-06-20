"""GPU==CPU SELECTION-EQUIVALENCE guard for the MRMR permutation-MI path.

Two layers of protection, both auto-skipping on CUDA-unavailable hosts:

1. KERNEL UNIT (``test_compute_mi_from_classes_cuda_matches_cpu``): the CUDA
   single-permutation MI reducer ``compute_mi_from_classes_cuda`` must match the
   bit-exact CPU njit ``compute_mi_from_classes`` for EVERY target cardinality
   ``nbins_y in {2, 3, 5, 10}`` (binary class, multi-class, and quantile-binned
   regression targets). Pre-2026-06-11 the kernel hardcoded the joint-histogram
   row stride as ``i*2+j`` (stride 2), so it only matched a BINARY target
   (nbins_y==2); for any other cardinality it read the wrong joint cells and
   returned a garbage permutation-null MI. That garbage systematically
   over-rejected genuine candidates in the permutation gate, silently diverging
   the GPU-path MRMR selection from the CPU path. This test pins the fix.

2. END-TO-END SELECTION (``test_mrmr_gpu_cpu_selection_identical``): a full
   ``MRMR(use_gpu=True)`` vs ``MRMR(use_gpu=False)`` fit on small regression +
   classification fixtures must select the IDENTICAL feature set (the relevance
   MI is computed by the same CPU njit kernel in both paths, and after the kernel
   fix the GPU permutation null no longer over-rejects). This also covers a second
   pre-2026-06-11 bug: ``MRMR(use_gpu=True)`` CRASHED on every fit that reached the
   FE pair-search because ``screen_predictors`` returned a CuPy ``classes_y_safe``
   into the FE step's njit MI noise-gate (TypingError: Cannot determine Numba type
   of cupy.ndarray) -- so this test would not even run pre-fix.

GPU fixtures are kept SMALL (n<=4000, p<=20) so they fit a 4GB consumer GPU; the
CuPy memory pool is freed between fits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cp = pytest.importorskip("cupy")


def _need_cuda() -> bool:
    try:
        from pyutilz.core.pythonlib import is_cuda_available
        return is_cuda_available()
    except Exception:
        return False


pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not _need_cuda(), reason="no CUDA")]


def _free_gpu() -> None:
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Layer 1: CUDA MI-reducer kernel vs CPU njit, across target cardinalities.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nbins_x,nbins_y", [
    (4, 2),    # binary target -- the ONLY case the stride-2 bug happened to match
    (4, 3),    # 3-class target -- stride-2 bug read wrong cells
    (6, 5),    # 5-class
    (10, 10),  # quantile-binned regression target (10 bins) -- worst case
    (3, 7),    # asymmetric, nbins_y > nbins_x
])
def test_compute_mi_from_classes_cuda_matches_cpu(nbins_x, nbins_y):
    """The CUDA single-perm MI reducer must equal the CPU njit MI to ~1e-9 for
    EVERY target cardinality, not just nbins_y==2."""
    from mlframe.feature_selection.filters import gpu as g
    from mlframe.feature_selection.filters.info_theory import compute_mi_from_classes

    g._ensure_kernels_inited()
    _free_gpu()

    rng = np.random.default_rng(20260611)
    n = 4000
    classes_x = rng.integers(0, nbins_x, size=n).astype(np.int32)
    classes_y = rng.integers(0, nbins_y, size=n).astype(np.int32)

    # Marginals (float64), exactly as the MRMR merge_vars path builds them.
    freqs_x = np.bincount(classes_x, minlength=nbins_x).astype(np.float64) / n
    freqs_y = np.bincount(classes_y, minlength=nbins_y).astype(np.float64) / n

    cpu_mi = float(compute_mi_from_classes(classes_x, freqs_x, classes_y, freqs_y))

    # GPU: build the joint histogram via the production kernel, then reduce via
    # the production MI kernel -- the exact two-kernel chain mi_direct_gpu runs.
    cx = cp.asarray(classes_x)
    cy = cp.asarray(classes_y)
    fx = cp.asarray(freqs_x)
    fy = cp.asarray(freqs_y)
    joint = cp.zeros((nbins_x, nbins_y), dtype=cp.int32)
    totals = cp.zeros(1, dtype=cp.float64)

    block = 256
    grid = (n + block - 1) // block
    g.compute_joint_hist_cuda(
        (grid,), (block,),
        (cx, cy, joint, np.int32(n), np.int32(nbins_y)),
    )
    g.compute_mi_from_classes_cuda(
        (1,), (1,),
        (cx, fx, cy, fy, joint, totals,
         np.int32(n), np.int32(nbins_x), np.int32(nbins_y)),
    )
    gpu_mi = float(totals.get()[0])
    _free_gpu()

    assert cpu_mi > 0  # sanity: random data still has a tiny positive plug-in MI
    assert abs(gpu_mi - cpu_mi) < 1e-9, (
        f"CUDA MI reducer diverged from CPU njit at nbins_x={nbins_x}, "
        f"nbins_y={nbins_y}: gpu={gpu_mi!r} cpu={cpu_mi!r} "
        f"(abs diff {abs(gpu_mi - cpu_mi):.3e}). The joint-histogram row stride "
        f"in compute_mi_from_classes_cuda must be nbins_y, not a hardcoded 2."
    )


# ---------------------------------------------------------------------------
# Layer 2: end-to-end MRMR selection identity, forced-CPU vs forced-GPU.
# ---------------------------------------------------------------------------
def _fx_reg_ratio(seed, n):
    rng = np.random.default_rng(seed)
    a, b, e = (rng.uniform(0, 1, n) for _ in range(3))
    y = 0.30 * (a ** 2) / b + 0.01 * e
    return pd.DataFrame({"a": a, "b": b, "e": e}), pd.Series(y, name="y")


def _fx_reg_two_pairs(seed, n):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n)
    c = rng.uniform(1, 5, n); d = rng.uniform(0, 2 * np.pi, n)
    e = rng.normal(0, 1, n)
    y = a ** 2 / b + 3.0 * np.log(c) * np.sin(d) + 0.3 * e
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}), pd.Series(y, name="y")


def _fx_reg_mixed(seed, n):
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n)
    c = rng.uniform(0, 1, n); d = rng.uniform(0, 1, n); e = rng.uniform(0, 1, n)
    g = rng.uniform(0, 1, n); h = rng.uniform(0, 1, n)
    y = a ** 2 / b + 0.6 * (c - d) + e / 40.0
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e, "g": g, "h": h}), pd.Series(y, name="y")


def _fx_clf_binary(seed, n):
    rng = np.random.default_rng(seed)
    a, b, c, e = (rng.uniform(0, 1, n) for _ in range(4))
    logit = 2.0 * a - 1.5 * b + 0.8 * (a * c)
    p = 1.0 / (1.0 + np.exp(-(logit - logit.mean())))
    y = (rng.uniform(0, 1, n) < p).astype(int)
    return pd.DataFrame({"a": a, "b": b, "c": c, "e": e}), pd.Series(y, name="y")


_E2E_FIXTURES = [
    ("reg_ratio", _fx_reg_ratio, 4000),
    ("reg_two_pairs", _fx_reg_two_pairs, 4000),
    ("reg_mixed", _fx_reg_mixed, 4000),
    ("clf_binary", _fx_clf_binary, 4000),
]


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name,builder,n", _E2E_FIXTURES, ids=[f[0] for f in _E2E_FIXTURES])
def test_mrmr_gpu_cpu_selection_identical(name, builder, n):
    """Forced-GPU MRMR.fit must select the SAME feature set as forced-CPU.

    The relevance MI is the same CPU njit kernel in both paths; the GPU only
    accelerates the permutation null. After the CUDA-reducer stride fix the GPU
    permutation gate no longer over-rejects, so the selected sets match exactly.
    """
    from mlframe.feature_selection.filters.mrmr import MRMR

    seed = 42
    df, y = builder(seed, n)

    _free_gpu()
    cpu = MRMR(verbose=0, random_seed=seed, fe_max_steps=1, use_gpu=False,
               full_npermutations=64, baseline_npermutations=64)
    cpu.fit(df, y)
    sel_cpu = set(cpu.get_feature_names_out())

    _free_gpu()
    gpu = MRMR(verbose=0, random_seed=seed, fe_max_steps=1, use_gpu=True,
               full_npermutations=64, baseline_npermutations=64)
    gpu.fit(df, y)
    sel_gpu = set(gpu.get_feature_names_out())
    _free_gpu()

    assert sel_cpu == sel_gpu, (
        f"[{name}] GPU-path MRMR selected a DIFFERENT feature set than CPU:\n"
        f"  CPU-only ({len(sel_cpu)}): {sorted(sel_cpu)}\n"
        f"  GPU-only ({len(sel_gpu)}): {sorted(sel_gpu)}\n"
        f"  CPU\\GPU: {sorted(sel_cpu - sel_gpu)}\n"
        f"  GPU\\CPU: {sorted(sel_gpu - sel_cpu)}"
    )


# ---------------------------------------------------------------------------
# Layer 3: ADVERSARIAL near-tie selection identity, forced-CPU vs forced-GPU.
#
# Layer 2 confirms identity on benign fixtures where the permutation gate sits
# far from its accept/reject threshold. This layer pins the HARD cases the
# divergence fix was meant to survive -- the razor ties where a single divergent
# permutation outcome could flip the gate, or a wrong joint-histogram stride
# could shift the permutation-null MI just enough to over-reject:
#   A. TWO features whose relevance MI is within ~1e-4 -> razor add-order.
#   B. An engineered candidate at FP-epsilon of the noise-gate threshold.
#   C. accumulation-order stress at p~60 (many gate calls).
#   D. n at the joint-hist kernel_tuning_cache size-routing edge (4096) so the
#      GPU may route to a different kernel variant than neighbouring sizes.
# ``full_npermutations`` is raised to 128 (tighter than Layer 2's 64) so the
# gate threshold lands on a finer grid -- the tightest tie under which the
# 2026-06-11 probe still observed byte-identical selection on a cc 6.1 GPU.
# Both ``mi_perm >= original_mi`` comparisons (CPU permutation.py + GPU
# gpu.py:mi_direct_gpu_batched) test against the SAME CPU-njit ``original_mi``,
# so identity here is the real end-to-end guarantee, not merely kernel parity.
# ---------------------------------------------------------------------------
def _fx_adv_near_tie(seed, n):
    """Two near-identical informative features (MI within ~1e-4) + a distractor."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 1, n)
    y = base ** 2
    f1 = base + rng.normal(0, 1e-3, n)
    f2 = base + rng.normal(0, 1.0001e-3, n)  # MI within ~1e-4 of f1
    noise = rng.uniform(0, 1, n)
    return pd.DataFrame({"f1": f1, "f2": f2, "noise": noise}), pd.Series(y, name="y")


def _fx_adv_gate_epsilon(seed, n):
    """A weak candidate sitting where the permutation noise-gate flips."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 1, n); b = rng.uniform(0, 1, n)
    y = a + 0.5 * b
    weak = (a > 0.5).astype(float) * 0.02 + rng.uniform(0, 1, n)
    pure_noise = rng.uniform(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "weak": weak, "pure_noise": pure_noise}), pd.Series(y, name="y")


def _fx_adv_wide_p60(seed, n, p=60):
    """p~60: 5 informative + correlated-noise distractors -> many gate calls."""
    rng = np.random.default_rng(seed)
    cols = {}
    informative = []
    for k in range(5):
        cols[f"sig{k}"] = rng.uniform(0, 1, n)
        informative.append(cols[f"sig{k}"])
    y = sum((1.0 / (k + 1)) * v for k, v in enumerate(informative)) + 0.05 * rng.normal(0, 1, n)
    for k in range(p - 5):
        cols[f"d{k}"] = 0.001 * informative[k % 5] + rng.uniform(0, 1, n)
    return pd.DataFrame(cols), pd.Series(y, name="y")


def _fx_adv_dispatch_edge(seed, n):
    """n at the joint-hist dispatcher crossover (4096)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(1, 5, n); b = rng.uniform(1, 5, n); c = rng.uniform(1, 5, n)
    y = a ** 2 / b + np.log(c) + 0.1 * rng.normal(0, 1, n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "nz": rng.uniform(0, 1, n)}), pd.Series(y, name="y")


_ADV_FIXTURES = [
    ("adv_near_tie", _fx_adv_near_tie, 4000),
    ("adv_gate_epsilon", _fx_adv_gate_epsilon, 4000),
    ("adv_wide_p60", _fx_adv_wide_p60, 4000),
    ("adv_dispatch_edge", _fx_adv_dispatch_edge, 4096),
]


@pytest.mark.timeout(420)
@pytest.mark.parametrize("seed", [42, 1])
@pytest.mark.parametrize("name,builder,n", _ADV_FIXTURES, ids=[f[0] for f in _ADV_FIXTURES])
def test_mrmr_gpu_cpu_selection_identical_adversarial(name, builder, n, seed):
    """Forced-GPU MRMR.fit selects byte-identically to forced-CPU even at the
    razor ties (near-tie relevance MI, gate-epsilon candidates, p=60
    accumulation order, dispatcher-edge n). Pins the tightest tie that holds."""
    from mlframe.feature_selection.filters.mrmr import MRMR

    df, y = builder(seed, n)

    _free_gpu()
    cpu = MRMR(verbose=0, random_seed=seed, fe_max_steps=1, use_gpu=False,
               full_npermutations=128, baseline_npermutations=128)
    cpu.fit(df, y)
    sel_cpu = set(cpu.get_feature_names_out())

    _free_gpu()
    gpu = MRMR(verbose=0, random_seed=seed, fe_max_steps=1, use_gpu=True,
               full_npermutations=128, baseline_npermutations=128)
    gpu.fit(df, y)
    sel_gpu = set(gpu.get_feature_names_out())
    _free_gpu()

    assert sel_cpu == sel_gpu, (
        f"[{name} seed={seed}] adversarial near-tie: GPU-path MRMR selected a "
        f"DIFFERENT feature set than CPU:\n"
        f"  CPU-only ({len(sel_cpu)}): {sorted(sel_cpu)}\n"
        f"  GPU-only ({len(sel_gpu)}): {sorted(sel_gpu)}\n"
        f"  CPU\\GPU: {sorted(sel_cpu - sel_gpu)}\n"
        f"  GPU\\CPU: {sorted(sel_gpu - sel_cpu)}"
    )


# ---------------------------------------------------------------------------
# Perf guard: the joint-hist HW-aware fallback must probe GPU capability AT MOST
# ONCE per process. A cProfile of mi_direct_gpu_batched (n=5000, nperm=1024)
# showed gpu_capability_summary -> GPUtil -> an nvidia-smi SUBPROCESS at ~45% of
# the GPU-path wall because the fallback re-probed on every kernel-tuning-cache
# MISS call. _cached_cc_major memoises the immutable cc_major for the process so
# the hot dispatch path never re-shells nvidia-smi. This pins that contract.
# ---------------------------------------------------------------------------
def test_hw_aware_fallback_probes_capability_once(monkeypatch):
    """``_hw_aware_fallback`` must call ``gpu_capability_summary`` at most once
    across many lookups (process-lifetime cc_major memo)."""
    import mlframe.feature_selection._benchmarks.kernel_tuning_cache.dispatch as disp
    import pyutilz.system.gpu_dispatch as gd

    # Reset the process-lifetime memo so this test is order-independent.
    monkeypatch.setattr(disp, "_CC_MAJOR_CACHE", None, raising=False)

    calls = {"n": 0}
    real_summary = gd.gpu_capability_summary

    def _counting_summary(device_id=0):
        calls["n"] += 1
        return real_summary(device_id)

    monkeypatch.setattr(gd, "gpu_capability_summary", _counting_summary)

    # 50 lookups across two joint sizes -> at most ONE capability probe.
    first = disp._hw_aware_fallback(64)
    for _ in range(49):
        disp._hw_aware_fallback(64)
        disp._hw_aware_fallback(8192)
    assert calls["n"] <= 1, (
        f"_hw_aware_fallback re-probed GPU capability {calls['n']} times across "
        f"50 lookups; the cc_major memo must cap it at 1 (each probe shells "
        f"nvidia-smi for live VRAM the dispatch does not use)."
    )
    # And the fallback still returns a well-formed payload.
    assert set(first) == {"kernel_variant", "block_size"}
    # Restore real memo state for any later test in this process.
    monkeypatch.setattr(disp, "_CC_MAJOR_CACHE", None, raising=False)
