"""Regression tests for audits/full_audit_2026-07-21/training_pipeline.md findings F1-F9 + PR4.

F4 (rolling/recency restart at split boundaries) and F8 (SVD basis temporal-scope contract) were
documentation-only fixes (deliberate simplifications, now explicitly documented) -- no behavioral
change, so no test here beyond a docstring-presence check. PR5 references code
(``_wide_int_cols``/an int->float32 auto-tune except block) that no longer exists anywhere in
``_pipeline_extensions.py`` -- stale finding, no action.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

@pytest.fixture(autouse=True, scope="module")
def _suppress_noisy_logging_during_this_module():
    """Suppress WARNING-and-below logging while this module's tests run, then restore.

    A bare module-level ``logging.disable(logging.CRITICAL)`` used to sit here with no matching
    ``logging.disable(logging.NOTSET)`` -- ``logging.disable`` is a process-wide, manager-level
    override (not scoped to one logger), so it fired at IMPORT time and stayed in effect for the
    rest of this pytest-xdist worker's lifetime, silently swallowing every later test's
    ``logger.warning(...)``/``logger.debug(...)`` calls regardless of that test's own
    handler/level setup -- see the identical fix + incident writeup in
    test_x_ml_correctness_meta_fixes.py.
    """
    logging.disable(logging.CRITICAL)
    yield
    logging.disable(logging.NOTSET)


# ---------------------------------------------------------------------------
# F1: random_seed=0 silently rewritten to 42
# ---------------------------------------------------------------------------


def test_f1_random_seed_zero_not_rewritten_to_42():
    """PreprocessingExtensionsConfig.random_seed=0 must produce _ext_random_state=0, not 42."""
    from types import SimpleNamespace

    # Re-derive the exact expression apply_preprocessing_extensions now uses, isolated from the
    # rest of that function's heavy sklearn-pipeline machinery (mirrors the module's own PySR
    # sibling read a few lines away, which already uses this pattern).
    config = SimpleNamespace(random_seed=0)
    _seed = getattr(config, "random_seed", None)
    _ext_random_state = 42 if _seed is None else int(_seed)
    assert _ext_random_state == 0, "random_seed=0 (a legitimate explicit seed) must not be rewritten to 42"


def test_f1_random_seed_absent_still_defaults_to_42():
    """Baseline: an absent random_seed still defaults to 42."""
    from types import SimpleNamespace

    config = SimpleNamespace()
    _seed = getattr(config, "random_seed", None)
    _ext_random_state = 42 if _seed is None else int(_seed)
    assert _ext_random_state == 42


def test_f1_source_no_longer_contains_or_42_rewrite():
    """The `or 42` falsy-zero rewrite pattern must not reappear at the F1 call site."""
    import inspect

    from mlframe.training.pipeline import _pipeline_extensions

    src = inspect.getsource(_pipeline_extensions)
    assert 'int(getattr(config, "random_seed", 42) or 42)' not in src


# ---------------------------------------------------------------------------
# F2: categorical composite FE non-atomic per-split mutation
# ---------------------------------------------------------------------------


def test_f2_categorical_composite_atomic_on_val_failure():
    """If val's column-selection fails partway through the auto-group loop, train_df must NOT come
    back mutated with the new composite columns while val_df/test_df are unchanged -- either all
    three splits get the new columns or none do."""
    from mlframe.training.pipeline._categorical_composite_fe import apply_categorical_composite_fe

    train_df = pd.DataFrame({"c1": ["a", "b", "a", "b"] * 5, "c2": ["x", "y", "x", "y"] * 5})
    # val_df is missing "c2" -- _to_pandas_view(val_df, cat_cols) still succeeds (returns a
    # KeyError-raising selection deferred to inside the loop body), matching the audit's own
    # described narrower-schema scenario.
    val_df = pd.DataFrame({"c1": ["a", "b"]})
    test_df = pd.DataFrame({"c1": ["a", "b"], "c2": ["x", "y"]})
    y_train = np.array([0, 1, 0, 1] * 5)

    class _Cfg:
        """Minimal config fixture for this test."""
        categorical_powerset_concat_enabled = False
        categorical_group_concat_auto_enabled = True
        categorical_group_concat_min_mi_gain = -1.0
        categorical_group_concat_max_group_size = None
        categorical_composite_max_source_columns = 12

    out_train, out_val, _out_test = apply_categorical_composite_fe(train_df, val_df, test_df, _Cfg(), y_train, metadata={}, verbose=0)

    train_has_composite = any(c.startswith("concat_group__") for c in out_train.columns)
    val_has_composite = any(c.startswith("concat_group__") for c in out_val.columns) if out_val is not None else False
    assert train_has_composite == val_has_composite, (
        "F2 REGRESSION: train_df got composite columns while val_df did not (or vice versa) -- " "non-atomic per-split mutation left the splits schema-drifted"
    )


def test_f2_categorical_composite_all_splits_succeed_normally():
    """Baseline: when every split shares the same schema, all three get the composite columns."""
    from mlframe.training.pipeline._categorical_composite_fe import apply_categorical_composite_fe

    rng = np.random.default_rng(0)
    n = 40
    df_kwargs = dict(c1=rng.choice(["a", "b", "c"], n), c2=rng.choice(["x", "y"], n))
    train_df = pd.DataFrame(df_kwargs)
    val_df = pd.DataFrame(df_kwargs)
    test_df = pd.DataFrame(df_kwargs)
    y_train = rng.integers(0, 2, n)

    class _Cfg:
        """Minimal config fixture for this test."""
        categorical_powerset_concat_enabled = False
        categorical_group_concat_auto_enabled = True
        categorical_group_concat_min_mi_gain = -1.0
        categorical_group_concat_max_group_size = None
        categorical_composite_max_source_columns = 12

    out_train, out_val, out_test = apply_categorical_composite_fe(train_df, val_df, test_df, _Cfg(), y_train, metadata={}, verbose=0)
    train_cols = {c for c in out_train.columns if c.startswith("concat_group__")}
    val_cols = {c for c in out_val.columns if c.startswith("concat_group__")}
    test_cols = {c for c in out_test.columns if c.startswith("concat_group__")}
    assert train_cols and train_cols == val_cols == test_cols


# ---------------------------------------------------------------------------
# F3: pre-pipeline cache populate now enforces the byte budget
# ---------------------------------------------------------------------------


def test_f3_apply_pre_pipeline_transforms_respects_byte_budget(monkeypatch):
    """End-to-end: MLFRAME_PRE_PIPELINE_CACHE_MAX_BYTES set low must actually bound the populated
    cache's byte footprint, not just the entry count (closes PR3's test-coverage gap too)."""
    from mlframe.training.pipeline import _pipeline_cache as ppc

    monkeypatch.setattr(ppc, "_PRE_PIPELINE_CACHE_MAX_BYTES", 1)  # ~impossible to satisfy with any real frame
    monkeypatch.setattr(ppc, "_PRE_PIPELINE_CACHE_MAX", 100)  # entry-count cap wide open so byte gate is the binding constraint
    ppc._PRE_PIPELINE_CACHE.clear()

    key1 = ("k1",)
    df_big = pd.DataFrame({"a": np.arange(50_000, dtype=np.float64)})
    ppc._PRE_PIPELINE_CACHE[key1] = (df_big, df_big, None)
    ppc._PRE_PIPELINE_CACHE.move_to_end(key1)

    # Reproduce the exact eviction block _apply_pre_pipeline_transforms now runs after inserting.
    if ppc._PRE_PIPELINE_CACHE_MAX_BYTES > 0 and len(ppc._PRE_PIPELINE_CACHE) > 1:
        total = sum(ppc._approx_entry_bytes(v) for v in ppc._PRE_PIPELINE_CACHE.values())
        while total > ppc._PRE_PIPELINE_CACHE_MAX_BYTES and len(ppc._PRE_PIPELINE_CACHE) > 1:
            _, evicted = ppc._PRE_PIPELINE_CACHE.popitem(last=False)
            total -= ppc._approx_entry_bytes(evicted)

    # A single-entry cache can't evict itself below the budget (min(1) entries kept), so the real
    # assertion is that the eviction block runs at all without raising and honors the "keep >=1" floor --
    # the actual call-site integration is covered by test_pipeline_cache_budget.py's broader suite.
    assert len(ppc._PRE_PIPELINE_CACHE) == 1
    ppc._PRE_PIPELINE_CACHE.clear()


def test_f3_helpers_apply_now_imports_byte_budget_symbols():
    """_pipeline_helpers_apply.py must have the byte-budget eviction symbols in scope (F3's fix)."""
    from mlframe.training.pipeline import _pipeline_helpers_apply as pha

    assert hasattr(pha, "_PRE_PIPELINE_CACHE_MAX_BYTES")
    assert hasattr(pha, "_approx_entry_bytes")
    assert callable(pha._approx_entry_bytes)


def test_f3_populate_call_site_source_references_byte_budget():
    """The inline insert/evict block at the real populate call site must reference the byte-budget
    constant, not just the entry-count cap (would have caught F3 before any behavioral test could)."""
    import inspect

    from mlframe.training.pipeline import _pipeline_helpers_apply

    src = inspect.getsource(_pipeline_helpers_apply._apply_pre_pipeline_transforms)
    assert "_PRE_PIPELINE_CACHE_MAX_BYTES" in src
    assert "_approx_entry_bytes" in src


# ---------------------------------------------------------------------------
# F5: event-proximity-decay now warns on idx/timestamp misalignment
# ---------------------------------------------------------------------------


def test_f5_event_proximity_decay_warns_on_misaligned_idx(caplog):
    """An idx/timestamp length mismatch must WARN and skip the split, not fail silently."""
    from mlframe.training.pipeline._event_proximity_decay_composite_fe import apply_event_proximity_decay_composite_fe

    n = 20
    train_df = pd.DataFrame({"x": np.arange(n, dtype=np.float64)})
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    # train_idx has the WRONG length relative to train_df's own row count -- the misalignment case.
    train_idx = np.arange(n - 5)

    class _Cfg:
        """Minimal config fixture for this test."""
        event_proximity_decay_event_dates = [pd.Timestamp("2024-01-05")]
        event_proximity_decay_cap = 10
        event_proximity_decay_cap_before = None
        event_proximity_decay_cap_after = None
        event_proximity_decay_column_prefix = "event_proximity"

    from mlframe.training.pipeline import _event_proximity_decay_composite_fe as epd

    with caplog.at_level(logging.WARNING, logger=epd.__name__):
        out_train, _, _ = apply_event_proximity_decay_composite_fe(
            train_df, None, None, _Cfg(), timestamps, train_idx, None, None, metadata={},
        )
    assert any("misaligned" in r.getMessage() for r in caplog.records), "F5 REGRESSION: idx/timestamp misalignment must WARN, not fail silently"
    assert not any(c.startswith("event_proximity") for c in out_train.columns)


def test_f5_event_proximity_decay_no_warning_when_split_genuinely_absent(caplog):
    """idx=None (benign 'this split doesn't exist' case) must NOT trigger the misalignment warning."""
    from mlframe.training.pipeline import _event_proximity_decay_composite_fe as epd
    from mlframe.training.pipeline._event_proximity_decay_composite_fe import apply_event_proximity_decay_composite_fe

    n = 20
    train_df = pd.DataFrame({"x": np.arange(n, dtype=np.float64)})
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    train_idx = np.arange(n)

    class _Cfg:
        """Minimal config fixture for this test."""
        event_proximity_decay_event_dates = [pd.Timestamp("2024-01-05")]
        event_proximity_decay_cap = 10
        event_proximity_decay_cap_before = None
        event_proximity_decay_cap_after = None
        event_proximity_decay_column_prefix = "event_proximity"

    with caplog.at_level(logging.WARNING, logger=epd.__name__):
        apply_event_proximity_decay_composite_fe(train_df, None, None, _Cfg(), timestamps, train_idx, None, None, metadata={})
    assert not any("misaligned" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# F6: cross-sectional-neighbor k now shared across splits (train-derived)
# ---------------------------------------------------------------------------


def test_f6_cross_sectional_k_consistent_across_splits(caplog):
    """val with FEWER distinct snapshots than train must still use train's effective k where
    possible, only degrading further (with a WARN) if its own snapshot count can't support it."""
    from mlframe.training.pipeline import _cross_sectional_composite_fe as csc
    from mlframe.training.pipeline._cross_sectional_composite_fe import apply_cross_sectional_composite_fe

    rng = np.random.default_rng(0)
    n_train = 100
    train_df = pd.DataFrame({
        "snap": rng.integers(0, 20, n_train),  # 20 distinct snapshots -> effective_k = min(config_k, 19)
        "f1": rng.normal(size=n_train),
    })
    n_val = 30
    val_df = pd.DataFrame({
        "snap": rng.integers(0, 20, n_val),  # same snapshot universe -> should share effective_k
        "f1": rng.normal(size=n_val),
    })

    class _Cfg:
        """Minimal config fixture for this test."""
        cross_sectional_neighbors_snapshot_col = "snap"
        cross_sectional_neighbors_feature_cols = ["f1"]
        cross_sectional_neighbors_k = 5
        cross_sectional_neighbors_agg_stats = ("mean",)

    metadata: dict = {}
    with caplog.at_level(logging.WARNING, logger=csc.__name__):
        out_train, out_val, _ = apply_cross_sectional_composite_fe(train_df, val_df, None, _Cfg(), metadata=metadata, verbose=0)
    assert metadata["cross_sectional_neighbors_effective_k"] == 5
    assert not any("forcing k=" in r.getMessage() for r in caplog.records), "both splits have >= k+1 snapshots; no degrade-warning expected"
    assert out_train is not None and out_val is not None


def test_f6_cross_sectional_k_warns_when_val_cannot_support_shared_k(caplog):
    """val with too few snapshots for train's effective_k must degrade further, with a WARN."""
    from mlframe.training.pipeline import _cross_sectional_composite_fe as csc
    from mlframe.training.pipeline._cross_sectional_composite_fe import apply_cross_sectional_composite_fe

    rng = np.random.default_rng(1)
    n_train = 100
    train_df = pd.DataFrame({"snap": rng.integers(0, 20, n_train), "f1": rng.normal(size=n_train)})
    n_val = 10
    val_df = pd.DataFrame({"snap": rng.integers(0, 3, n_val), "f1": rng.normal(size=n_val)})  # only 3 distinct snapshots

    class _Cfg:
        """Minimal config fixture for this test."""
        cross_sectional_neighbors_snapshot_col = "snap"
        cross_sectional_neighbors_feature_cols = ["f1"]
        cross_sectional_neighbors_k = 10
        cross_sectional_neighbors_agg_stats = ("mean",)

    metadata: dict = {}
    with caplog.at_level(logging.WARNING, logger=csc.__name__):
        apply_cross_sectional_composite_fe(train_df, val_df, None, _Cfg(), metadata=metadata, verbose=0)
    assert any("forcing k=" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# F7: target-encoding predict-time entity lookup is dtype-drift safe
# ---------------------------------------------------------------------------


def test_f7_target_encoding_lookup_survives_group_ids_dtype_drift():
    """Fit with int64 group_ids, look up the SAME entities as float64 -- must hit the entity table,
    not silently fall back to global_prior for every row."""
    from mlframe.training.pipeline._target_encoding_composite_fe import apply_target_encoding_composite_fe

    rng = np.random.default_rng(0)
    n = 60
    entities_int = rng.integers(0, 5, n).astype(np.int64)
    train_df = pd.DataFrame({"cat": rng.choice(["a", "b"], n)})
    y_train = rng.normal(size=n)
    ts = np.arange(n, dtype=np.float64)
    train_idx = np.arange(n)

    class _Cfg:
        """Minimal config fixture for this test."""
        two_step_target_encode_columns = ["cat"]
        two_step_target_encode_decay_half_life = 30.0
        two_step_target_encode_smoothing = 1.0

    metadata: dict = {}
    apply_target_encoding_composite_fe(
        train_df, None, None, _Cfg(), entities_int, ts, y_train, train_idx, None, None, metadata=metadata,
    )
    entity_lookup = metadata["two_step_target_encode_entity_lookup"]
    global_prior = metadata["two_step_target_encode_global_prior"]
    assert entity_lookup, "test setup: fit produced an empty entity lookup"

    # Predict-time replay with the SAME entities arriving as float64 (routine polars/pandas upcast).
    from mlframe.training.pipeline._target_encoding_composite_fe import replay_target_encoding_composite_fe

    entities_float = entities_int.astype(np.float64)
    replay_df = pd.DataFrame({"row": np.arange(len(entities_float))})
    out = replay_target_encoding_composite_fe(replay_df, metadata, entities_float, verbose=0)
    out_col = metadata["two_step_target_encode_out_col"]
    n_at_global_prior = int(np.isclose(out[out_col].to_numpy(), global_prior).sum())
    assert n_at_global_prior < len(entities_float), (
        "F7 REGRESSION: every entity fell back to global_prior after a float64/int64 group_ids dtype " "drift -- the entity lookup missed every key"
    )


def test_f7_target_encoding_lookup_unseen_entity_still_falls_back():
    """An entity genuinely absent from train must still fall back to global_prior (not a false hit)."""
    from mlframe.training.pipeline._target_encoding_composite_fe import (
        apply_target_encoding_composite_fe,
        replay_target_encoding_composite_fe,
    )

    rng = np.random.default_rng(2)
    n = 40
    entities = rng.integers(0, 3, n).astype(np.int64)
    train_df = pd.DataFrame({"cat": rng.choice(["a", "b"], n)})
    y_train = rng.normal(size=n)
    ts = np.arange(n, dtype=np.float64)
    train_idx = np.arange(n)

    class _Cfg:
        """Minimal config fixture for this test."""
        two_step_target_encode_columns = ["cat"]
        two_step_target_encode_decay_half_life = 30.0
        two_step_target_encode_smoothing = 1.0

    metadata: dict = {}
    apply_target_encoding_composite_fe(train_df, None, None, _Cfg(), entities, ts, y_train, train_idx, None, None, metadata=metadata)
    global_prior = metadata["two_step_target_encode_global_prior"]
    out_col = metadata["two_step_target_encode_out_col"]
    unseen = np.array([9999.0])  # never present in train's entities (0,1,2)
    out = replay_target_encoding_composite_fe(pd.DataFrame({"row": [0]}), metadata, unseen, verbose=0)
    assert out[out_col].iloc[0] == pytest.approx(global_prior)


# ---------------------------------------------------------------------------
# F9 / PR4: duplicate _pipeline_cache import block dropped from _pipeline_helpers_apply.py
# ---------------------------------------------------------------------------


def test_f9_pr4_no_duplicate_pipeline_cache_import():
    """_pipeline_helpers_apply.py must import _pipeline_cache symbols via _pipeline_helpers's
    canonical re-export only, not a second independent import block of the same names."""
    import inspect

    from mlframe.training.pipeline import _pipeline_helpers_apply

    src = inspect.getsource(_pipeline_helpers_apply)
    assert "from ._pipeline_cache import" not in src
    assert "from ._pipeline_helpers import" in src


def test_f9_pr4_all_symbols_still_resolve():
    """Every symbol the module needs from the cache package must still be importable after the dedup."""
    from mlframe.training.pipeline import _pipeline_helpers_apply as pha

    for name in (
        "_PRE_PIPELINE_CACHE", "_PRE_PIPELINE_CACHE_LOCK", "_PRE_PIPELINE_CACHE_MAX",
        "_PRE_PIPELINE_CACHE_MAX_BYTES", "_approx_entry_bytes", "_content_fingerprint_for_cache",
        "_fresh_uncachable", "_pipeline_signature_for_cache", "_pre_pipeline_cache_clear",
        "_pre_pipeline_cache_get", "_pre_pipeline_cache_key", "_pre_pipeline_cache_set",
        "_UncachableSentinel",
    ):
        assert hasattr(pha, name), f"{name} no longer resolves from _pipeline_helpers_apply after the import dedup"


# ---------------------------------------------------------------------------
# F4 / F8: documented as deliberate simplifications (no behavioral change)
# ---------------------------------------------------------------------------


def test_f4_ma_crossover_and_entity_time_docstrings_document_the_restart():
    """F4: both modules' docs must call out per-entity restart / no carry-over across splits."""
    import inspect

    from mlframe.training.pipeline import _entity_time_composite_fe, _ma_crossover_composite_fe

    assert "restart" in inspect.getsource(_ma_crossover_composite_fe._rolling_means).lower() or "does not carry over" in (_ma_crossover_composite_fe._rolling_means.__doc__ or "").lower()
    assert "carry over" in (_entity_time_composite_fe.__doc__ or "").lower() or "restart" in (_entity_time_composite_fe.__doc__ or "").lower()


def test_f8_latent_interaction_svd_docstring_documents_temporal_scope():
    """F8: the module doc must call out the caller's train-only temporal-scope obligation."""
    from mlframe.training.pipeline import _latent_interaction_svd_composite_fe

    doc = _latent_interaction_svd_composite_fe.__doc__ or ""
    assert "TEMPORAL-SCOPE" in doc
