"""Cheap sanity tests for the shap_proxied_fs bench scripts.

Asserts the bench modules import cleanly (the entire iter21-23 rabbit hole started because the
scaling bench was producing single-seed numbers; we now have multi-seed aggregation + a noise-pool
diagnosis sweep, and we want CI to keep the modules buildable). Also exercises the pure-Python
aggregation helpers on a tiny synthetic case so a refactor that breaks the mean/std contract gets
caught without paying for a real fit."""

from __future__ import annotations

import pytest


def test_bench_shap_proxy_scaling_imports_and_aggregates():
    from mlframe.feature_selection._benchmarks import bench_shap_proxy_scaling as mod

    # Module surface that the bench script + downstream callers depend on.
    for name in (
        "make_wide",
        "_build_selector",
        "bench_width_single",
        "bench_width_multi_seed",
        "_agg",
        "print_multi_seed_table",
        "print_stage_breakdown",
        "main",
        "_STAGE_ORDER",
    ):
        assert hasattr(mod, name), name

    # _agg contract on 1, 2, N samples.
    one = mod._agg([0.875])
    assert one["mean"] == pytest.approx(0.875)
    assert one["std"] == 0.0
    assert one["min"] == 0.875 == one["max"]

    multi = mod._agg([1.0, 0.875, 0.75])
    assert multi["mean"] == pytest.approx(0.875)
    assert multi["min"] == 0.75
    assert multi["max"] == 1.0
    assert multi["std"] > 0.0

    # Empty -> nans, not a crash.
    empty = mod._agg([])
    assert empty["mean"] != empty["mean"]  # NaN

    # _STAGE_ORDER mirrors the pipeline stages we time.
    assert "prefilter" in mod._STAGE_ORDER
    assert "revalidation" in mod._STAGE_ORDER


def test_bench_shap_proxy_noise_pool_sweep_imports_and_aggregates():
    from mlframe.feature_selection._benchmarks import bench_shap_proxy_noise_pool_sweep as mod

    for name in ("_build_selector", "_make", "_agg", "run_cell", "main"):
        assert hasattr(mod, name), name

    # _agg contract mirrors the scaling bench (kept in sync deliberately).
    out = mod._agg([0.875, 1.0, 0.875])
    assert out["mean"] == pytest.approx((0.875 + 1.0 + 0.875) / 3)
    assert out["min"] == 0.875
    assert out["max"] == 1.0
    assert out["std"] > 0.0
