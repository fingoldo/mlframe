"""PipelineCache HIT/MISS observability.

Previously PipelineCache was constructed without a ``verbose`` arg, which
defaulted to ``False`` and silenced the HIT/MISS log lines - operators
triaging "why is this suite re-fitting?" had no signal. The default is
now ``verbose=True`` so the lines log by default; tight unit loops can
opt back out via ``PipelineCache(verbose=False)``.
"""
from __future__ import annotations

import logging

import pandas as pd
import pytest


def _log_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records]


def test_pipeline_cache_default_verbose_true_emits_hit_and_miss(caplog):
    """The zero-arg constructor must emit HIT and MISS lines."""
    from mlframe.training.strategies import PipelineCache

    caplog.set_level(logging.INFO, logger="mlframe.training.strategies")

    cache = PipelineCache()
    assert cache.verbose is True, (
        "PipelineCache() default verbose must be True so HIT/MISS lines "
        "log under the locked caller"
    )

    df = pd.DataFrame({"a": [1, 2, 3]})
    cache.set("k_a", df, df, df)
    cache.get("k_missing")  # MISS
    cache.get("k_a")        # HIT

    lines = _log_lines(caplog)
    assert any("PipelineCache MISS" in l and "k_missing" in l for l in lines), (
        f"expected MISS log line, got: {lines}"
    )
    assert any("PipelineCache HIT" in l and "k_a" in l for l in lines), (
        f"expected HIT log line, got: {lines}"
    )


def test_pipeline_cache_verbose_false_silences_logs(caplog):
    """Explicit ``verbose=False`` opts out of the HIT/MISS lines."""
    from mlframe.training.strategies import PipelineCache

    caplog.set_level(logging.INFO, logger="mlframe.training.strategies")

    cache = PipelineCache(verbose=False)
    df = pd.DataFrame({"a": [1, 2, 3]})
    cache.set("k_a", df, df, df)
    cache.get("k_missing")
    cache.get("k_a")

    lines = _log_lines(caplog)
    # Counters still bump (other regression nets check this); only the
    # log lines are gated by ``verbose``.
    assert cache.n_hits == 1
    assert cache.n_misses == 1
    assert not any("PipelineCache HIT" in l for l in lines), (
        f"verbose=False should suppress HIT line, got: {lines}"
    )
    assert not any("PipelineCache MISS" in l for l in lines), (
        f"verbose=False should suppress MISS line, got: {lines}"
    )
