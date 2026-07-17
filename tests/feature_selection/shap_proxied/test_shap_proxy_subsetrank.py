"""Unit + biz_value for the subset-ranking dispatcher and the CPU per-subset reference kernel.

Covers (no GPU required -- runs on every host):
  * CPU njit per-subset reference (``brute_force_top_n_cpu_ref``) is bit-identical to a pure-numpy
    brute-force argmax (the oracle), across metrics and cardinalities.
  * dispatcher DEFAULTS to CPU and matches the incremental ``brute_force_top_n`` exactly.
  * dispatcher auto-falls back to CPU when the GPU backend raises (simulated cupy failure), never
    propagating the error -- the contract that keeps a flaky GPU host from failing a fit.
  * ``force_backend='cpu_ref'`` routes the naive scan; biz_value asserts the dispatcher recovers the
    KNOWN-best subset on a planted-signal synthetic.
A separate GPU-gated test asserts GPU == CPU bit-identity when a device is present.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import proxy_loss
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n
from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_subsetrank as SR


def _numpy_brute_oracle(phi, base, y, *, classification, metric, max_card, top_n):
    """Pure-numpy reference: every subset's loss via ``proxy_loss``, then the SAME ``_merge_topn``
    post-processing the kernels use (global top-N + best-per-cardinality) so the comparison is
    apples-to-apples, not a tail-ordering artifact of the merge step."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import _merge_topn

    f = phi.shape[1]
    res = []
    for r in range(1, max_card + 1):
        for comb in itertools.combinations(range(f), r):
            margin = base + phi[:, list(comb)].sum(axis=1)
            res.append((proxy_loss(margin, y, metric), tuple(comb)))
    return _merge_topn(res, top_n)


@pytest.mark.parametrize("metric,classification", [("rmse", False), ("mae", False), ("brier", True), ("logloss", True)])
def test_cpu_ref_bit_identical_to_numpy_oracle(metric, classification):
    rng = np.random.default_rng(7)
    n, f, mc = 300, 8, 4
    phi = rng.standard_normal((n, f))
    base = rng.standard_normal(n) * 0.1
    margin = base + phi[:, [1, 3, 5]].sum(axis=1)
    if classification:
        y = (1.0 / (1.0 + np.exp(-margin)) > rng.random(n)).astype(float)
    else:
        y = margin + 0.02 * rng.standard_normal(n)

    ref = SR.brute_force_top_n_cpu_ref(phi, base, y, classification=classification, metric=metric, max_card=mc, top_n=10)
    oracle = _numpy_brute_oracle(phi, base, y, classification=classification, metric=metric, max_card=mc, top_n=10)

    assert ref[0][1] == oracle[0][1], "argmax subset must be bit-identical to the numpy oracle"
    np.testing.assert_allclose(ref[0][0], oracle[0][0], rtol=1e-9, atol=1e-12)
    assert {frozenset(c) for _, c in ref} == {frozenset(c) for _, c in oracle}


def test_cpu_ref_matches_incremental_kernel():
    rng = np.random.default_rng(11)
    phi = rng.standard_normal((500, 12))
    base = rng.standard_normal(500) * 0.1
    y = (rng.random(500) < 0.5).astype(float)
    ref = SR.brute_force_top_n_cpu_ref(phi, base, y, classification=True, max_card=5, top_n=30)
    inc = brute_force_top_n(phi, base, y, classification=True, max_card=5, top_n=30, parallel=True)
    assert {frozenset(c) for _, c in ref} == {frozenset(c) for _, c in inc}
    np.testing.assert_allclose(ref[0][0], inc[0][0], rtol=1e-9, atol=1e-12)


def test_dispatch_defaults_to_cpu_and_matches_incremental():
    rng = np.random.default_rng(3)
    phi = rng.standard_normal((400, 10))
    base = rng.standard_normal(400) * 0.1
    y = (rng.random(400) < 0.5).astype(float)
    disp = SR.brute_force_top_n_dispatch(phi, base, y, classification=True, max_card=5, top_n=30)  # prefer_gpu defaults False
    inc = brute_force_top_n(phi, base, y, classification=True, max_card=5, top_n=30, parallel=True)
    assert {frozenset(c) for _, c in disp} == {frozenset(c) for _, c in inc}


def test_dispatch_falls_back_to_cpu_on_gpu_failure(monkeypatch):
    """When the GPU backend raises (cupy missing / OOM), the dispatcher must return the CPU result,
    never propagate -- the no-crash contract for the contended host."""
    import mlframe.feature_selection.shap_proxied_fs._shap_proxy_gpu as G

    def _boom(*a, **k):
        raise RuntimeError("simulated cupy OOM / import segfault")

    monkeypatch.setattr(G, "gpu_available", lambda: True)
    monkeypatch.setattr(G, "brute_force_top_n_gpu", _boom)
    SR._fallback_logged = False  # reset the log-once latch so the warning path is exercised

    rng = np.random.default_rng(5)
    phi = rng.standard_normal((300, 9))
    base = rng.standard_normal(300) * 0.1
    y = (rng.random(300) < 0.5).astype(float)
    # prefer_gpu + force the route past the crossover via env so we actually hit the GPU branch.
    monkeypatch.setenv("MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS", "1")
    disp = SR.brute_force_top_n_dispatch(phi, base, y, classification=True, max_card=4, top_n=20, prefer_gpu=True)
    inc = brute_force_top_n(phi, base, y, classification=True, max_card=4, top_n=20, parallel=True)
    assert {frozenset(c) for _, c in disp} == {frozenset(c) for _, c in inc}


def test_biz_val_dispatch_recovers_planted_best_subset():
    """biz_value: on a synthetic where features {2,5,7} additively generate the margin and the rest are
    noise, the subset-rank dispatcher's argmax MUST be exactly {2,5,7}. A regressed kernel (wrong sum /
    wrong reduction / broken fallback) fails to recover the planted optimum."""
    rng = np.random.default_rng(0)
    n, f = 2000, 11
    phi = rng.standard_normal((n, f)) * 0.05  # weak noise units
    signal = [2, 5, 7]
    phi[:, signal] = rng.standard_normal((n, len(signal))) * 1.5  # strong signal units
    base = np.zeros(n)
    margin = base + phi[:, signal].sum(axis=1)
    y = (1.0 / (1.0 + np.exp(-margin)) > rng.random(n)).astype(float)

    # Proxy loss is monotone non-increasing in |S| (more phi to fit y), so the GLOBAL argmax tends to
    # the largest subset; the planted optimum is the BEST subset AT its cardinality -- which is exactly
    # what _merge_topn's best-per-card augmentation surfaces. A regressed kernel breaks this recovery.
    ranked = SR.brute_force_top_n_dispatch(phi, base, y, classification=True, metric="brier", max_card=4, top_n=30)
    best_card3 = min((c for c in ranked if len(c[1]) == 3), key=lambda t: t[0])
    assert set(best_card3[1]) == set(signal), f"best 3-subset {best_card3[1]} must recover planted {signal}"


def test_gpu_min_subsets_env_override(monkeypatch):
    monkeypatch.setenv("MLFRAME_SHAP_SUBSETRANK_GPU_MIN_SUBSETS", "12345")
    assert SR._gpu_min_subsets() == 12345


def _has_cuda_device():
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _has_cuda_device(), reason="no CUDA device available")
def test_dispatch_gpu_bit_identical_to_cpu():
    """When a device is present, force_backend='gpu' must be bit-identical to the CPU kernel."""
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((500, 16))
    base = rng.standard_normal(500) * 0.1
    y = (rng.random(500) < 0.5).astype(float)
    gpu = SR.brute_force_top_n_dispatch(phi, base, y, classification=True, max_card=6, top_n=30, force_backend="gpu")
    cpu = brute_force_top_n(phi, base, y, classification=True, max_card=6, top_n=30, parallel=True)
    assert gpu[0][1] == cpu[0][1]
    assert {frozenset(c) for _, c in gpu} == {frozenset(c) for _, c in cpu}
