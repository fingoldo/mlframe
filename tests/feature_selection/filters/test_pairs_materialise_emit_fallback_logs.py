"""The pair materialise / emit fallbacks log a WARNING (driven on CPU by stubbing the failing call), so a slower or degraded path is never silent."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.utils.log_throttle import reset_throttle_counts


@pytest.fixture(autouse=True)
def _fresh_throttles():
    """Forget the per-key log counts so each test sees its first occurrence."""
    reset_throttle_counts()
    yield
    reset_throttle_counts()


def _warned(caplog, text):
    """Whether a WARNING record contains ``text``."""
    return any(r.levelno >= logging.WARNING and text in r.getMessage() for r in caplog.records)


def _boom(*args, **kwargs):
    """Fail like a device or cache fault."""
    raise RuntimeError("boom")


def test_ktc_parallelism_failure_warns_and_uses_heuristic(monkeypatch, caplog):
    """A failing kernel-tuning-cache choice falls back to the heuristic and warns."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_materialise as mod

    monkeypatch.setattr(mod._FE_PARALLELISM_SPEC, "choose", _boom)
    with caplog.at_level(logging.WARNING):
        assert isinstance(mod._fe_use_parallel_kernels(8, True), bool)
    assert _warned(caplog, "kernel_tuning_cache serial-vs-parallel choice failed")


def test_batched_leader_corr_failure_warns(caplog):
    """A failing batched leader |corr| returns None (the per-leader path runs) and warns."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_emit as mod

    def _resolve(_c):
        """Never called."""
        return None

    _resolve.abs_corr_many = _boom  # type: ignore[attr-defined]
    with caplog.at_level(logging.WARNING):
        assert mod._batched_leader_usability(_resolve, [(0, 0, 1), (0, 0, 2)], np.zeros(3), np.ones(3, dtype=bool), True) is None
    assert _warned(caplog, "batched leader |corr| failed")


def test_leader_usability_failure_warns(caplog):
    """A failing leader usability computation scores the leader without it and warns."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_emit as mod

    cfgs = {(("a", "u"), ("b", "u"), 0): 1.0, (("a", "u"), ("b", "u"), 1): 1.0}
    with caplog.at_level(logging.WARNING):
        mod._emit_pair_features_step1_st_simplenamespace_long(
            lambda fe_tuple, cols_names: str(fe_tuple[2]), ["a", "b"], False, True, False, cfgs, 1.0, 0.5, np.zeros(4), object(), False, lambda c: None, None, None, {}, {}, lambda v: 0.0
        )
    assert _warned(caplog, "leader usability computation for")


def _multi_emit(resolve):
    """Run the multi-emit step with the given column resolver."""
    from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_emit as mod

    emitted = (("a", "u"), 0, 0)
    other = (("b", "u"), 0, 1)
    mod._emit_pair_features_step2_than_single_best(3, object(), False, {(emitted, 0)}, 1.0, 0.5, 0.9, str, resolve, {other: 1.0}, SimpleNamespace(), 0, [])
    return emitted, other


def test_resolve_emitted_column_failure_warns(caplog):
    """A failing resolve of an already-emitted column skips it in the diversity check and warns."""
    with caplog.at_level(logging.WARNING):
        _multi_emit(_boom)
    assert _warned(caplog, "resolving already-emitted column")


def test_resolve_candidate_column_failure_warns(caplog):
    """A failing resolve of a candidate column leaves it unemitted and warns."""
    with caplog.at_level(logging.WARNING):
        _multi_emit(_boom)
    assert _warned(caplog, "resolving column")
