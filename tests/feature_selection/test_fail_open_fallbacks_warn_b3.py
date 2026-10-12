"""Batch 3: fallbacks for missing optional modules and failed helper calls log at WARNING.

Each test removes the optional module or makes the helper fail, then asserts a WARNING record. The auto-detect and
MDLP factorize fallbacks change the result, so their messages say the result is degraded. At DEBUG the same records are
dropped by default logging, so these tests fail on the old code.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd


def _warnings_mentioning(caplog, needle: str) -> list[logging.LogRecord]:
    """Return WARNING-or-higher records whose message contains needle."""
    return [r for r in caplog.records if r.levelno >= logging.WARNING and needle in r.getMessage()]


def test_register_default_tuning_cache_warns_when_pyutilz_is_missing(monkeypatch, caplog, tmp_path):
    """Without pyutilz's kernel-tuning cache the defaults are not registered and the hand-tuned fallbacks are used, with a WARNING."""
    from mlframe.feature_selection.filters import _kernel_tuning as mod

    tuning_json = tmp_path / "defaults.json"
    tuning_json.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mod, "_DEFAULT_TUNING_JSON", str(tuning_json))
    monkeypatch.setattr(mod, "_DEFAULTS_REGISTERED", False)
    monkeypatch.setitem(sys.modules, "pyutilz.performance.kernel_tuning.cache", None)
    with caplog.at_level(logging.DEBUG):
        mod._register_default_tuning_cache()
    assert _warnings_mentioning(caplog, "default kernel-tuning cache not registered")


def test_get_kernel_tuning_cache_warns_when_pyutilz_is_missing(monkeypatch, caplog):
    """Without pyutilz's cache class the getter returns None and warns that hand-tuned fallbacks are used."""
    from mlframe.feature_selection.filters import _kernel_tuning as mod

    monkeypatch.setattr(mod, "_CACHE_SINGLETON", None)
    monkeypatch.setitem(sys.modules, "pyutilz.performance.kernel_tuning.cache", None)
    with caplog.at_level(logging.DEBUG):
        assert mod.get_kernel_tuning_cache() is None
    assert _warnings_mentioning(caplog, "kernel_tuning.cache unavailable")


def test_auto_detect_group_cols_warns_and_degrades_when_detector_is_missing(monkeypatch, caplog):
    """A missing group-column detector falls back to a cardinality scan and says the result is degraded."""
    from mlframe.feature_selection.filters import _grouped_coerce_shared as mod

    monkeypatch.setitem(sys.modules, "mlframe.training.composite", None)
    X = pd.DataFrame({"g": ["a", "b"] * 50, "v": np.arange(100.0)})
    with caplog.at_level(logging.DEBUG):
        cols = mod.auto_detect_group_cols(X, max_cols=4, caller="test")
    assert isinstance(cols, list)
    assert _warnings_mentioning(caplog, "result degraded to the fallback cardinality scan")


def test_generate_target_shuffles_warns_when_numba_check_fails(monkeypatch, caplog):
    """A failing numba-availability check falls back to the numpy shuffle and says the result is degraded."""
    from mlframe.feature_selection.filters import _permutation_null as mod

    monkeypatch.setitem(sys.modules, "mlframe.feature_selection.filters._permutation_null_shufflegen_ktc", None)
    y = np.arange(40, dtype=np.int64) % 3
    with caplog.at_level(logging.DEBUG):
        out = mod._generate_target_shuffles(y, 3, np.int32, np.random.default_rng(0), random_seed=0)
    assert out.shape == (3, 40)
    assert _warnings_mentioning(caplog, "result degraded to the numpy shuffle path")


def test_mdlp_bin_edges_validated_warns_and_degrades_when_factorize_fails(monkeypatch, caplog):
    """A failing pandas factorize on string labels falls back to np.unique codes and says the result is degraded."""
    import pandas

    from mlframe.feature_selection.filters import _mdlp_validated_split as mod

    def _boom(*_a, **_k):
        """Raise to force the fallback."""
        raise RuntimeError("factorize broken")

    monkeypatch.setattr(pandas, "factorize", _boom)
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = np.array(["lo", "hi"] * 100, dtype=object)
    with caplog.at_level(logging.DEBUG):
        edges = mod.mdlp_bin_edges_validated(x, y)
    assert edges[0] == -np.inf and edges[-1] == np.inf
    assert _warnings_mentioning(caplog, "pandas factorize failed, result degraded")


def test_mdlp_bin_edges_oos_validated_warns_and_degrades_when_factorize_fails(monkeypatch, caplog):
    """The out-of-sample MDLP variant has the same factorize fallback and the same degraded-result warning."""
    import pandas

    from mlframe.feature_selection.filters import _mdlp_validated_split as mod

    def _boom(*_a, **_k):
        """Raise to force the fallback."""
        raise RuntimeError("factorize broken")

    monkeypatch.setattr(pandas, "factorize", _boom)
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = np.array(["lo", "hi"] * 100, dtype=object)
    with caplog.at_level(logging.DEBUG):
        mod.mdlp_bin_edges_oos_validated(x, y)
    assert _warnings_mentioning(caplog, "pandas factorize failed, result degraded")
