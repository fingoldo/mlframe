"""Baseline-debt wave 7: representative logging regression tests for the 16 genuine
broad_except_swallow sites fixed across training/cb/_cb_pool.py, training/core/
_phase_train_one_target.py, utils/_param_oracle.py, training/composite/cache.py, and
training/neural/data.py -- one spot-check per file rather than one test per site, since these
are uniform additive debug-log-on-failure changes with no behavior change on the success path
(already covered by each module's existing test suite).
"""

from __future__ import annotations

import logging


def test_cb_pool_recover_feature_names_logs_on_failure(caplog):
    """`_recover_cb_feature_names` must log when model introspection fails."""
    from mlframe.training.cb._cb_pool import _recover_cb_feature_names

    class _RaisingModel:
        """A model whose `feature_names_` access raises, forcing the except branch."""

        @property
        def feature_names_(self):
            """Always raises ``RuntimeError('boom')`` on access."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.cb._cb_pool"):
        out = _recover_cb_feature_names(_RaisingModel())
    assert out == ([], [])
    assert any("feature-name recovery failed" in rec.message for rec in caplog.records)


def test_phase_train_one_target_selector_kind_logs_on_failure(caplog):
    """`_selector_kind`'s marker-getattr except must log when introspection fails."""
    from mlframe.training.core._phase_train_one_target import _selector_kind

    class _RaisingSelector:
        """A selector whose `_mlframe_selector_kind_` access raises."""

        @property
        def _mlframe_selector_kind_(self):
            """Always raises ``RuntimeError('boom')`` on access."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.core._phase_train_one_target"):
        _selector_kind(_RaisingSelector())
    assert any("getattr(selector, '_mlframe_selector_kind_')" in rec.message for rec in caplog.records)


def test_param_oracle_rss_mb_logs_on_failure(caplog):
    """`_rss_mb` must log when the psutil RSS probe fails -- forced via sys.modules poisoning
    (psutil is imported lazily inside `_rss_mb`, so this is cheaper than mocking import machinery)."""
    import sys

    import mlframe.utils._param_oracle as po

    real_psutil = sys.modules.pop("psutil", None)
    sys.modules["psutil"] = None  # importing a None entry raises ImportError
    try:
        with caplog.at_level(logging.DEBUG, logger="mlframe.utils._param_oracle"):
            out = po._rss_mb()
    finally:
        sys.modules.pop("psutil", None)
        if real_psutil is not None:
            sys.modules["psutil"] = real_psutil
    assert out is None
    assert any("RSS probe unavailable" in rec.message for rec in caplog.records)


def test_composite_cache_int_digest_logs_on_failure(caplog, monkeypatch):
    """The int/bool column min/max digest except (`_col_stats`, a closure inside
    `data_signature`) must log on failure. Driven through a real `data_signature` call on a
    small int-column frame, with `np.min` monkeypatched (on the cache module's own `np`) to
    raise only for integer dtypes."""
    import pandas as pd

    import mlframe.training.composite.cache as cache_mod

    real_min = cache_mod.np.min

    def _raising_min(arr, *args, **kwargs):
        """Raise for integer/bool arrays only, delegating everything else to the real np.min."""
        if getattr(getattr(arr, "dtype", None), "kind", "") in ("i", "u", "b"):
            raise RuntimeError("boom")
        return real_min(arr, *args, **kwargs)

    monkeypatch.setattr(cache_mod.np, "min", _raising_min)

    df = pd.DataFrame({"f0": [1, 2, 3, 4, 5]})
    with caplog.at_level(logging.DEBUG, logger="mlframe.training.composite.cache"):
        sig = cache_mod.data_signature(df, target_col="y", feature_cols=["f0"])
    assert isinstance(sig, str) and len(sig) == 32
    assert any("cache: int/bool column min/max digest failed" in rec.message for rec in caplog.records)


def test_neural_data_byte_size_estimation_logs_on_failure(caplog):
    """`neural.data`'s byte-size estimator must log when introspection fails."""
    from mlframe.training.neural.data import _estimate_bytes

    class _RaisingSizeObj:
        """An object whose `.nbytes` access raises."""

        @property
        def nbytes(self):
            """Always raises ``RuntimeError('boom')`` on access."""
            raise RuntimeError("boom")

    with caplog.at_level(logging.DEBUG, logger="mlframe.training.neural.data"):
        out = _estimate_bytes(_RaisingSizeObj())
    assert out == 0
    assert any("byte-size estimation failed" in rec.message for rec in caplog.records)
