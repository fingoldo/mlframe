"""Conditional-dispersion replay must bin ``x_j`` against the same float64 edges the fit used (a float32 edge moves a value that sits on it)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from tests._cuda_require import require_cuda_device

from mlframe.feature_selection.filters._extra_fe_families_dispersion import (
    apply_conditional_dispersion,
    build_conditional_dispersion_recipe,
    generate_conditional_dispersion_features,
)
from mlframe.feature_selection.filters.engineered_recipes import apply_recipe


def _frame() -> pd.DataFrame:
    """Two columns: ``xj`` holds two adjacent float32 values (the median edge is their float64 midpoint), ``xi`` differs per bin."""
    a = np.float32(1.0)
    b = np.nextafter(a, np.float32(2.0))
    rng = np.random.default_rng(0)
    xj = np.array([a] * 200 + [b] * 200, dtype=np.float32)
    xi = np.concatenate([rng.normal(0, 1, 200), rng.normal(5, 4, 200)]).astype(np.float32)
    return pd.DataFrame({"xi": xi, "xj": xj})


def test_dispersion_replay_matches_fit_for_a_value_on_a_rounded_edge() -> None:
    """The fit digitises against the float64 midpoint edge; replay through the recipe must give the fit column exactly."""
    X = _frame()
    enc, recipes = generate_conditional_dispersion_features(X, ["xi", "xj"], n_bins=2, kinds=("absz",))
    name = "xi__absz_by__xj"
    assert name in enc.columns
    p = recipes[name]
    rec = build_conditional_dispersion_recipe(
        name=name, x_i=p["x_i"], x_j=p["x_j"], edges=p["edges"], bin_mean=p["bin_mean"], bin_std=p["bin_std"],
        global_mean=p["global_mean"], global_std=p["global_std"], kind=p["kind"],
    )
    replay = apply_recipe(rec, X)
    np.testing.assert_array_equal(np.asarray(replay), enc[name].to_numpy())
    direct = apply_conditional_dispersion(X, {**p})
    np.testing.assert_array_equal(np.asarray(direct), enc[name].to_numpy())


def test_driver_accepts_missing_classes_y() -> None:
    """``classes_y`` is optional in the candidate-evaluation driver: an empty workload must not call ``.copy()`` on None."""
    from mlframe.feature_selection.filters._evaluation_driver import _evaluate_candidates_inner

    out = _evaluate_candidates_inner(
        workload=[], y=[0], best_gain=0.0, factors_data=np.zeros((4, 2), dtype=np.int32), factors_nbins=np.array([2, 2]),
        factors_names=["a", "b"], partial_gains={}, selected_vars=[], baseline_npermutations=0,
        classes_y=None, entropy_cache={}, cached_cond_MIs={}, verbose=0,
    )
    assert out[1] is None


def test_block_width_fallback_warning_is_throttled(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """A persistent block-width derivation failure inside the per-group launch loop must not log on every launch."""
    import pyutilz.performance.gpu_hw_launch as hw

    from mlframe.feature_selection.filters import _family_maxt_cuda_bits as bits
    from mlframe.utils.log_throttle import reset_throttle_counts

    def boom(*a, **k):
        """Always fail."""
        raise RuntimeError("no device")

    monkeypatch.setattr(hw, "fixed_shared_block_size", boom)
    monkeypatch.delenv("MLFRAME_GPU_BLOCK_FAMILY_BITS", raising=False)
    reset_throttle_counts()

    class _K:
        """Kernel stub."""
        num_regs = 32
        shared_size_bytes = 0

    with caplog.at_level(logging.WARNING, logger=bits.logger.name):
        for _ in range(30):
            assert bits.v2_block_width(_K(), 1024, 100) == bits._THREADS_V2
    assert sum("block width not derived" in r.getMessage() for r in caplog.records) <= 5


def test_device_bin_codes_use_float64_edges() -> None:
    """The device bin codes equal the host digitiser's on a value that sits on a float32-rounded edge."""
    cp = pytest.importorskip("cupy")
    require_cuda_device()
    from mlframe.feature_selection.filters._extra_fe_families import _digitize_with_edges
    from mlframe.feature_selection.filters._extra_fe_families_dispersion_resident import _bin_codes_block
    from mlframe.feature_selection.filters._fe_gpu_batch._devices import crit_float_dtype

    a = np.float32(1.0)
    b = np.nextafter(a, np.float32(2.0))
    x = np.array([a, b, a, b], dtype=np.float32)
    edges = np.array([float(a), (float(a) + float(b)) / 2.0, float(b)])
    host = _digitize_with_edges(x.astype(np.float64) if crit_float_dtype() == np.float64 else x, edges)
    dev = cp.asnumpy(_bin_codes_block(cp, [x], [edges]))[0]
    np.testing.assert_array_equal(dev, host)
