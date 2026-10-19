"""Scoring one candidate matrix against many label vectors through a candidate cache returns exactly what independent calls return."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def _inputs(seed, n=3000, k=24, classes=3):
    """A column-major candidate block and a set of label vectors (the original and shuffles of it)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((k, n)).astype(np.float32)
    x[: k // 2] = np.round(x[: k // 2] * 3) / 3  # ties
    y = rng.integers(0, classes, n).astype(np.int64)
    ys = [y] + [rng.permutation(y) for _ in range(5)]
    return cp.asarray(x), [cp.asarray(v) for v in ys], int(classes)


@pytest.mark.parametrize("relax", [False, True])
@pytest.mark.parametrize("keep_dtype", [False, True])
def test_cached_candidates_give_the_same_mi_as_independent_calls(relax, keep_dtype):
    """Every score taken with a shared candidate cache equals the score of a call that prepares the candidates itself."""
    from mlframe.feature_selection.filters._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    x, ys, classes = _inputs(3)
    kw = dict(y_min=0, n_classes=classes, relax_binning=relax, keep_dtype=keep_dtype, x_is_cm=True)
    cache: dict = {}
    for y in ys:
        alone = np.asarray(_plugin_mi_classif_batch_cuda_resident(x, y, 10, **kw), dtype=np.float64)
        shared = np.asarray(_plugin_mi_classif_batch_cuda_resident(x, y, 10, candidate_cache=cache, **kw), dtype=np.float64)
        # The kernels are deterministic for a given configuration; the tolerance only absorbs a different (documented bit-equivalent) radix variant being picked by the tuning cache between two calls.
        np.testing.assert_allclose(shared, alone, rtol=1e-9, atol=1e-12)
    assert "state" in cache


def test_a_cache_is_not_shared_across_different_matrices_unless_the_caller_does_so():
    """The cache belongs to one candidate matrix: a fresh cache for a different matrix recomputes its edges."""
    from mlframe.feature_selection.filters._hermite_fe_mi import _plugin_mi_classif_batch_cuda_resident

    x1, ys, classes = _inputs(1)
    x2, _, _ = _inputs(2)
    kw = dict(y_min=0, n_classes=classes, relax_binning=True, keep_dtype=True, x_is_cm=True)
    first = np.asarray(_plugin_mi_classif_batch_cuda_resident(x1, ys[0], 10, candidate_cache={}, **kw), dtype=np.float64)
    second = np.asarray(_plugin_mi_classif_batch_cuda_resident(x2, ys[0], 10, candidate_cache={}, **kw), dtype=np.float64)
    assert not np.array_equal(first, second)
    again = np.asarray(_plugin_mi_classif_batch_cuda_resident(x2, ys[0], 10, **kw), dtype=np.float64)
    assert second.view(np.uint64).tolist() == again.view(np.uint64).tolist()


def test_binagg_null_floor_equals_the_loop_without_a_cache(monkeypatch):
    """The family-wise floor of the binned-aggregate gate is unchanged by the candidate cache (same shuffles, same maxima)."""
    from mlframe.feature_selection.filters import _binned_numeric_agg_resident as mod
    from mlframe.feature_selection.filters import _hermite_fe_mi as hm

    x, ys, classes = _inputs(5, n=2500, k=16)
    y_host = cp.asnumpy(ys[0])
    with_cache = mod._binagg_null_calibrated_floor(cp, x, ys[0], y_host, 0, classes, 10, 0.0)
    real = hm._plugin_mi_classif_batch_cuda_resident

    def without_cache(*args, **kwargs):
        """The same scorer with the candidate cache switched off."""
        kwargs.pop("candidate_cache", None)
        return real(*args, **kwargs)

    monkeypatch.setattr(hm, "_plugin_mi_classif_batch_cuda_resident", without_cache)
    plain = mod._binagg_null_calibrated_floor(cp, x, ys[0], y_host, 0, classes, 10, 0.0)
    assert with_cache == plain
