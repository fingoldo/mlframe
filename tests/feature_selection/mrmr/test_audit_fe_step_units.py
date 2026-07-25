"""Direct unit tests for the FE-step sub-block helpers and the per-family wall diagnostic.

These helpers previously had no direct coverage: they were only exercised transitively through a full
(multi-minute) MRMR fit, so a regression in their gating/caching logic surfaced only as a downstream
selection change. Also pins the ScreenContext container defaults and the FE-family timing overlap caveat.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest


def test_non_numeric_column_indices_pandas_and_polars():
    """_non_numeric_column_indices must report the positional indices of non-numeric columns for both a pandas
    and a polars frame, and an empty set for an all-numeric one."""
    import pandas as pd

    from mlframe.feature_selection.filters._mrmr_fe_step._helpers import _non_numeric_column_indices

    pdf = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [3, 4]})
    assert _non_numeric_column_indices(pdf, list(pdf.columns)) == {1}
    assert _non_numeric_column_indices(pd.DataFrame({"a": [1.0], "c": [2]}), ["a", "c"]) == set()

    pl = pytest.importorskip("polars")
    poldf = pl.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "c": [3, 4]})
    assert _non_numeric_column_indices(poldf, list(poldf.columns)) == {1}


def test_synergy_bootstrap_can_supply_pool_gating():
    """_synergy_bootstrap_can_supply_pool must require the screen enabled, the FIRST FE step, and enough rows."""
    from mlframe.feature_selection.filters._mrmr_fe_step._helpers import _synergy_bootstrap_can_supply_pool

    class _S:
        """Minimal estimator stand-in carrying just the two synergy-bootstrap gating attributes."""

        fe_synergy_screen_max_features = 10
        fe_synergy_min_rows = 300

    data = np.zeros((500, 4))
    s = _S()
    assert _synergy_bootstrap_can_supply_pool(s, 0, data) is True
    assert _synergy_bootstrap_can_supply_pool(s, 1, data) is False, "only the first FE step may seed the pool"
    assert _synergy_bootstrap_can_supply_pool(s, 0, np.zeros((100, 4))) is False, "too few rows"
    s.fe_synergy_screen_max_features = 0
    assert _synergy_bootstrap_can_supply_pool(s, 0, data) is False, "screen disabled"


def test_get_col_codes_i64_caches_and_matches_uncached():
    """_get_col_codes_i64 must return contiguous int64 codes identical to the uncached path, and serve a repeat
    request for the same column from the cache (one copy per distinct column)."""
    from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_rank import _get_col_codes_i64

    data = np.arange(24, dtype=np.int32).reshape(6, 4)
    uncached = _get_col_codes_i64(data, 2, None)
    assert uncached.dtype == np.int64 and uncached.flags["C_CONTIGUOUS"]
    assert np.array_equal(uncached, data[:, 2].astype(np.int64))

    cache: dict = {}
    first = _get_col_codes_i64(data, 2, cache)
    second = _get_col_codes_i64(data, 2, cache)
    assert first is second, "a repeat request for the same column must be served from the cache"
    assert np.array_equal(first, uncached)
    assert set(cache) == {2}


def test_pair_gate_resident_enabled_honors_optout(monkeypatch):
    """_pair_gate_resident_enabled must return False when the MLFRAME_FE_GATE_RESIDENT_CANDS opt-out is set,
    regardless of the GPU-strict state."""
    from mlframe.feature_selection.filters._mrmr_fe_step._step_pairs_rank import _pair_gate_resident_enabled

    monkeypatch.setenv("MLFRAME_FE_GATE_RESIDENT_CANDS", "0")
    assert _pair_gate_resident_enabled() is False
    monkeypatch.setenv("MLFRAME_FE_GATE_RESIDENT_CANDS", "1")
    assert isinstance(_pair_gate_resident_enabled(), bool)  # the value depends on the host GPU; the bool contract does not


def test_fe_family_timer_records_wall_and_flags_concurrency():
    """fe_family_timer must accumulate per-family wall + invocation counts, and report peak concurrency > 1 when
    two families overlap (their walls then double-count, which the summary must be able to disclose)."""
    from mlframe.feature_selection.filters._fe_family_timing import (
        fe_family_timer,
        get_fe_family_max_concurrency,
        get_fe_family_wall,
        reset_fe_family_wall,
    )

    reset_fe_family_wall()
    with fe_family_timer("solo"):
        pass
    wall = get_fe_family_wall()
    assert "solo" in wall and wall["solo"][1] == 1
    assert get_fe_family_max_concurrency() == 1, "a single non-overlapping timer means no overlap"

    reset_fe_family_wall()
    both_in = threading.Barrier(2)
    done = threading.Barrier(2)

    def _worker(name):
        """Hold an fe_family_timer region open until both threads are inside, guaranteeing real overlap."""
        with fe_family_timer(name):
            both_in.wait(timeout=10)  # guarantee the two regions genuinely overlap
            done.wait(timeout=10)

    ts = [threading.Thread(target=_worker, args=(n,)) for n in ("famA", "famB")]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=15)
    assert get_fe_family_max_concurrency() == 2, "overlapping families must be detected so the summary can caveat it"
    reset_fe_family_wall()


def test_screen_context_container_defaults_are_empty_not_none():
    """ScreenContext's per-order mutable fields must default to real empty containers (never None), so no call site
    needs a null-check and the dataclass stays mypy-clean without per-field type: ignore."""
    from mlframe.feature_selection.filters._confirm_predictor_context import ScreenContext

    fields = ScreenContext.__dataclass_fields__
    for name, empty in (
        ("candidates", []),
        ("selected_vars", []),
        ("selected_interactions_vars", []),
        ("partial_gains", {}),
        ("added_candidates", set()),
        ("failed_candidates", set()),
    ):
        factory = fields[name].default_factory
        assert factory is not None, f"{name} must use a default_factory, not a None placeholder"
        assert factory() == empty
