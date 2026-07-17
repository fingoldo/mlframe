"""RESIDENT UPLOAD (2026-07-13): ``screen_predictors(use_gpu=True)`` must upload the CuPy device buffer for
the MRMR target (``classes_y_safe``/``freqs_y_safe``) ONCE across repeated calls with an UNCHANGED target
(mirrors the 3-10 screen/FE rounds one ``MRMR.fit()`` call makes against the same y, the same round-carried
pattern already established by ``seed_maxt_floor_cache``/``seed_workers_pool``), instead of a fresh
``cp.asarray`` every round. Proves the ``resident_operand`` adoption fix engages and uploads the exact
values a raw ``cp.asarray`` would.

Note: ``screen_predictors(use_gpu=True)``'s CONFIRM loop draws permutations via ``cp.random.default_rng()``
with NO seed (pre-existing, unrelated to this fix -- the modern cupy Generator API is not covered by the
function's ``random_seed``/``cp.random.seed`` threading, only the legacy global generator is), so which
candidates get CONFIRMED can vary call to call even with identical inputs. These tests therefore do NOT
assert on ``selected_vars`` identity across calls; they assert on the (fully deterministic) upload count and
on the exact VALUES handed to ``resident_operand`` at this call site, which is what the fix actually changed.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_resident_operands import clear_fe_resident_operands
from mlframe.feature_selection.filters.info_theory import merge_vars
from mlframe.feature_selection.filters.screen import screen_predictors


def _gpu_available() -> bool:
    try:
        return cp.cuda.runtime.getDeviceCount() >= 1
    except Exception:  # pragma: no cover - no driver / no GPU
        return False


_GPU_AVAILABLE = _gpu_available()
if not _GPU_AVAILABLE:  # pragma: no cover - guarded at collection time
    pytest.skip("No CUDA device available", allow_module_level=True)


@pytest.fixture(autouse=True)
def _clear_resident_cache():
    clear_fe_resident_operands()
    yield
    clear_fe_resident_operands()


def _make_data(n=600, m=5, seed=0):
    """factors_data embeds the target as its LAST column (the natural ``targets_data is factors_data``
    case) so ``merge_vars(factors_data=factors_data, vars_indices=y, ...)`` reads it directly."""
    rng = np.random.default_rng(seed)
    preds = rng.integers(0, 4, size=(n, m)).astype(np.int32)
    y_col = ((preds[:, 0] + rng.integers(0, 2, size=n)) % 2).astype(np.int32).reshape(-1, 1)
    factors_data = np.column_stack([preds, y_col]).astype(np.int32)
    factors_nbins = np.array([4] * m + [2], dtype=np.int32)
    return factors_data, factors_nbins


def _common_kwargs(factors_data, factors_nbins, **overrides):
    base = dict(
        factors_data=factors_data,
        factors_nbins=factors_nbins,
        factors_names=[f"f{i}" for i in range(factors_data.shape[1])],
        y=np.array([factors_data.shape[1] - 1], dtype=np.int32),
        full_npermutations=5,
        baseline_npermutations=3,
        n_workers=1,
        use_gpu=True,
        verbose=0,
        random_seed=42,
    )
    base.update(overrides)
    return base


def _count_asarray_calls_matching(monkeypatch, target_values):
    calls = {"n": 0}
    orig = cp.asarray
    target = np.ascontiguousarray(target_values)

    def spy(a, *args, **kw):
        if isinstance(a, np.ndarray) and a.shape == target.shape and a.dtype == target.dtype and np.array_equal(a, target):
            calls["n"] += 1
        return orig(a, *args, **kw)

    monkeypatch.setattr(cp, "asarray", spy)
    return calls


@pytest.mark.gpu
def test_screen_predictors_gpu_uploads_target_once_across_rounds(monkeypatch):
    """Two screen_predictors(use_gpu=True) calls mirroring 2 screen/FE rounds of the SAME MRMR fit (target
    unchanged) must upload classes_y via cp.asarray only ONCE across both, not once per round."""
    fd, fn = _make_data(seed=3)
    classes_y, _freqs_y, _ = merge_vars(
        factors_data=fd,
        vars_indices=np.array([fd.shape[1] - 1]),
        var_is_nominal=None,
        factors_nbins=fn,
        dtype=np.int32,
    )
    y_calls = _count_asarray_calls_matching(monkeypatch, classes_y.astype(np.int32))

    out1 = screen_predictors(**_common_kwargs(fd, fn))
    out2 = screen_predictors(**_common_kwargs(fd, fn))

    assert y_calls["n"] == 1, f"classes_y-content cp.asarray called {y_calls['n']}x across 2 screen rounds (expected 1, resident)"
    # Both calls must complete and return the documented tuple shape; selected_vars identity across calls is
    # NOT asserted (see module docstring: the confirm loop's unseeded cp.random.default_rng() is a
    # pre-existing, unrelated source of call-to-call variation on the GPU path).
    assert isinstance(out1[0], list) and isinstance(out2[0], list)


@pytest.mark.gpu
def test_screen_predictors_gpu_resident_uploads_exact_target_values(monkeypatch):
    """The value handed to ``resident_operand`` at the ``_screen_predictors.py`` call site (and therefore
    the cached device buffer's content) must equal EXACTLY what the pre-fix raw ``cp.asarray(classes_y.astype
    (np.int32))`` / ``cp.asarray(freqs_y)`` would have produced -- proving the fix is a pure caching change,
    not a value change."""
    import mlframe.feature_selection.filters._fe_resident_operands as _rop_mod

    fd, fn = _make_data(seed=8)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=fd,
        vars_indices=np.array([fd.shape[1] - 1]),
        var_is_nominal=None,
        factors_nbins=fn,
        dtype=np.int32,
    )

    captured: dict = {}
    orig_resident_operand = _rop_mod.resident_operand

    def _capture(arr, key, *a, **kw):
        result = orig_resident_operand(arr, key, *a, **kw)
        if key in ("screen_classes_y", "screen_freqs_y"):
            captured[key] = cp.asnumpy(result)
        return result

    monkeypatch.setattr(_rop_mod, "resident_operand", _capture)

    screen_predictors(**_common_kwargs(fd, fn))

    assert "screen_classes_y" in captured and "screen_freqs_y" in captured
    np.testing.assert_array_equal(captured["screen_classes_y"], classes_y.astype(np.int32))
    np.testing.assert_allclose(captured["screen_freqs_y"], freqs_y.astype(np.float64), atol=0, rtol=0)


@pytest.mark.gpu
def test_screen_predictors_gpu_resident_matches_disabled_cache_values(monkeypatch):
    """Same exact-value check with the diagnostic cache DISABLED (``MLFRAME_FE_RESIDENT_OPERANDS=0``, the
    pre-fix raw-upload-every-call behaviour) -- the uploaded content must be identical either way."""
    fd, fn = _make_data(seed=15)
    classes_y, freqs_y, _ = merge_vars(
        factors_data=fd,
        vars_indices=np.array([fd.shape[1] - 1]),
        var_is_nominal=None,
        factors_nbins=fn,
        dtype=np.int32,
    )

    monkeypatch.setenv("MLFRAME_FE_RESIDENT_OPERANDS", "0")
    clear_fe_resident_operands()

    import mlframe.feature_selection.filters._fe_resident_operands as _rop_mod

    captured: dict = {}
    orig_resident_operand = _rop_mod.resident_operand

    def _capture(arr, key, *a, **kw):
        result = orig_resident_operand(arr, key, *a, **kw)
        if key in ("screen_classes_y", "screen_freqs_y"):
            captured[key] = cp.asnumpy(result)
        return result

    monkeypatch.setattr(_rop_mod, "resident_operand", _capture)

    screen_predictors(**_common_kwargs(fd, fn))

    np.testing.assert_array_equal(captured["screen_classes_y"], classes_y.astype(np.int32))
    np.testing.assert_allclose(captured["screen_freqs_y"], freqs_y.astype(np.float64), atol=0, rtol=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--tb=short"])
