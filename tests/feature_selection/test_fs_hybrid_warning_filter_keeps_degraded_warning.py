"""Audit 2026-11-20 finding 12-F31: the two hard-bed benches must not install a catch-all warning filter.

A blanket ``ignore`` hid the "HybridRFECV degraded" UserWarning, so a failed RFECV fourth member silently returned the plain
selector's result. Only FutureWarning and DeprecationWarning are silenced now.
"""

import importlib.util
import warnings
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"


@pytest.mark.parametrize("name", ["round3_hard_bed_bench.py", "round3_hybrid_refine_bench.py"])
def test_bench_import_keeps_user_warnings_visible(name):
    """After importing the bench, no filter ignores every warning category."""
    saved = list(warnings.filters)
    try:
        spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), _DIR / name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        catch_all = [f for f in warnings.filters if f[0] == "ignore" and f[2] is Warning and f[1] is None]
        assert catch_all == [], f"{name} installs a catch-all ignore filter"
    finally:
        warnings.filters[:] = saved
