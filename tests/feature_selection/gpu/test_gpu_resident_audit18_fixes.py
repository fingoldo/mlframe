"""Regression tests for the chunk-18 audit fixes of the GPU-resident filter modules (prewarp precision, edge-width check, 1-D abs-corr fallback, visible fallbacks)."""

import logging

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_GPU = False
pytestmark = pytest.mark.skipif(not _HAS_GPU, reason="needs a CUDA device")

from mlframe.utils.log_throttle import reset_throttle_counts

_PKG = "mlframe.feature_selection.filters"


@pytest.fixture(autouse=True)
def _fresh_throttle():
    """Forget the log-throttle counts so each test sees its first occurrence."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _prewarp_spec_and_data():
    """A chebyshev prewarp whose operand has a spread of 1 on an offset of 1e8 (float32 spacing 8 there)."""
    x = 1.0e8 + np.linspace(0.0, 1.0, 257)
    spec = {"basis": "chebyshev", "preprocess": {"kind": "minmax", "lo": float(x.min()), "hi": float(x.max()), "clip": None}, "coef": [0.0, 1.0]}
    return x, spec


@pytest.mark.parametrize("break_kernel", [False, True])
def test_prewarp_replay_keeps_spread_on_a_large_offset(monkeypatch, break_kernel):
    """The device prewarp must centre the float64 operand before narrowing to the float32 buffer, on the fused kernel and on the cupy-chain fallback."""
    from mlframe.feature_selection.filters import _gpu_resident_fe as fe

    monkeypatch.setattr(fe, "crit_float_dtype", lambda: cp.float32)
    if break_kernel:
        def _boom(*_a, **_k):
            """Raise as a broken kernel or cache would."""
            raise RuntimeError("forced")

        monkeypatch.setattr(fe, "_get_prewarp_transform_kernel", _boom)
    x, spec = _prewarp_spec_and_data()
    got = cp.asnumpy(fe._gpu_apply_prewarp(cp, cp.asarray(x), spec))
    lo, hi = x.min(), x.max()
    want = 2.0 * (x - lo) / (hi - lo) - 1.0  # chebyshev degree 1 == the preprocessed axis
    assert np.max(np.abs(got - want)) < 1e-6


def test_searchsorted_codes_rejects_edges_of_the_wrong_width():
    """Edges with fewer columns than the candidate matrix would be read past their end by the binning kernel."""
    from mlframe.feature_selection.filters._gpu_resident_discretize import _searchsorted_codes

    cand = cp.asarray(np.random.default_rng(0).normal(size=(64, 5)).astype(np.float32))
    edges = cp.asarray(np.sort(np.random.default_rng(1).normal(size=(3, 4)), axis=0))
    with pytest.raises(ValueError, match="interior_edges"):
        _searchsorted_codes(cand, edges)


def test_abs_corr_cupy_fallback_handles_a_one_dimensional_candidate(monkeypatch):
    """The cupy-chain fallback must return one |corr| per column for a 1-D candidate, as the kernel path does."""
    from mlframe.feature_selection.filters import _gpu_resident_basis as grb

    rng = np.random.default_rng(2)
    y = rng.normal(size=200)
    c = y * 0.5 + rng.normal(size=200)

    def _boom(*_a, **_k):
        """Raise as a broken kernel or cache would."""
        raise RuntimeError("forced")

    monkeypatch.setattr(grb, "_get_abs_corr_kernel", _boom)
    got = cp.asnumpy(grb._gpu_batched_abs_corr(cp, cp.asarray(c), cp.asarray(y)))
    assert got.shape == (1,)
    assert abs(float(got[0]) - abs(np.corrcoef(c, y)[0, 1])) < 1e-6


def test_radix_quantile_failure_is_logged_as_a_warning(monkeypatch, caplog):
    """A failing radix-select quantile falls back to the sorting percentile, and says so at WARNING."""
    from mlframe.feature_selection.filters import _gpu_resident_basis as grb
    from mlframe.feature_selection.filters import _gpu_resident_select as grs

    def _boom(*_a, **_k):
        """Raise as a broken kernel or cache would."""
        raise RuntimeError("forced radix failure")

    monkeypatch.setattr(grs, "_radix_quantiles", _boom)
    m = cp.asarray(np.random.default_rng(3).normal(size=(500, 3)))
    with caplog.at_level(logging.WARNING, logger=grb.__name__):
        out = grb._batched_quantiles(cp, m, [0.5])
    assert out.shape == (1, 3)
    assert any(r.levelno == logging.WARNING and "forced radix failure" in r.getMessage() for r in caplog.records)


def test_fused_basis_preprocess_failure_is_logged_as_a_warning(monkeypatch, caplog):
    """A failing fused preprocess kernel falls back to the cupy chain, and says so at WARNING."""
    from mlframe.feature_selection.filters import _gpu_resident_basis as grb

    def _boom(*_a, **_k):
        """Raise as a broken kernel or cache would."""
        raise RuntimeError("forced preprocess failure")

    monkeypatch.setattr(grb, "_gpu_basis_preprocess_nonrobust_fused", _boom)
    m = cp.asarray(np.random.default_rng(4).normal(size=(300, 2)))
    with caplog.at_level(logging.WARNING, logger=grb.__name__):
        z = grb._gpu_basis_preprocess_batched(cp, m, "chebyshev", robust=False)
    assert z.shape == (300, 2)
    assert any(r.levelno == logging.WARNING and "forced preprocess failure" in r.getMessage() for r in caplog.records)


def test_extval_device_failure_is_logged_as_a_warning(monkeypatch, caplog):
    """A device fault in the ext-val materialise returns None for the host path, and says so at WARNING."""
    from mlframe.feature_selection.filters import _gpu_resident_extval as ev

    def _boom(*_a, **_k):
        """Raise as a broken kernel or cache would."""
        raise RuntimeError("forced extval failure")

    from mlframe.feature_selection.filters import _fe_gpu_vram

    monkeypatch.setattr(_fe_gpu_vram, "fe_gpu_has_vram_cushion", lambda *_a, **_k: True)
    monkeypatch.setattr(ev, "_candidate_matrix", _boom)
    a = np.random.default_rng(5).normal(size=400)
    b = [np.random.default_rng(6).normal(size=400)]
    with caplog.at_level(logging.WARNING, logger=ev.__name__):
        out = ev.gpu_materialise_extval_codes_host(a, b, np.asarray([0, 1], dtype=np.int32), 5)
    assert out is None
    assert any(r.levelno == logging.WARNING and "forced extval failure" in r.getMessage() for r in caplog.records)


def test_histgate_and_k_chunk_lookup_failures_are_logged_as_warnings(monkeypatch, caplog):
    """A failing tuning-cache lookup returns the default and says so at WARNING, for both tuners."""
    from mlframe.feature_selection.filters import _gpu_resident_histgate_ktc as hg
    from mlframe.feature_selection.filters import _gpu_resident_k_chunk_ktc as kc

    class _Spec:
        """A tuner spec whose lookup always fails."""

        def choose(self, **_k):
            """Raise as a broken cache would."""
            raise RuntimeError("forced cache failure")

    monkeypatch.setattr(hg, "_HISTGATE_THREADS_SPEC", _Spec())
    monkeypatch.setattr(kc, "_GPU_K_CHUNK_SPEC", _Spec())
    with caplog.at_level(logging.WARNING):
        assert hg.histgate_threads(1000) == hg._HISTGATE_THREADS_DEFAULT
        assert kc.gpu_k_chunk_vram_fraction(1000) == kc._GPU_K_CHUNK_VRAM_FRACTION_DEFAULT
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert sum("forced cache failure" in m for m in msgs) == 2


def test_pairs_tuning_lookup_failure_is_logged_as_a_warning(monkeypatch, caplog):
    """A failing multi-pair block-size lookup keeps the default block size and says so at WARNING."""
    from mlframe.feature_selection.filters import _gpu_pairs as gp
    from mlframe.feature_selection.filters import _kernel_tuning as kt

    class _Cache:
        """A tuning cache whose lookup always fails."""

        def lookup(self, *_a, **_k):
            """Raise as a broken cache would."""
            raise RuntimeError("forced pairs lookup failure")

    monkeypatch.setattr(kt, "get_kernel_tuning_cache", lambda: _Cache())
    rng = np.random.default_rng(7)
    fd = rng.integers(0, 4, size=(256, 3)).astype(np.int32)
    nb = np.full(3, 4, dtype=np.int64)
    y = rng.integers(0, 2, size=256).astype(np.int32)
    with caplog.at_level(logging.WARNING, logger=gp.__name__):
        mi = gp.mi_direct_gpu_batched_pairs(fd, np.asarray([0], dtype=np.int64), np.asarray([1], dtype=np.int64), nb, y, np.bincount(y) / 256.0)
    assert mi.shape == (1,)
    assert any(r.levelno == logging.WARNING and "forced pairs lookup failure" in r.getMessage() for r in caplog.records)
