"""Regression coverage for the 2026-05-21 io.py fixes:

1. ``save_mlframe_model`` emits a WARN when the dump exceeds 50 MB
   (catches DataLoader / trainer / optimizer-state bloat in production logs).
2. The Lightning bloat strip nullifies ``LightningModule._trainer`` and
   ``PytorchLightningEstimator.prediction_datamodule`` for the pickle pass,
   then restores them on the caller's in-memory payload.

The Lightning-bloat test reproduces the TVT-2026-05-21 scenario: a
LightningModule whose ``_trainer`` attribute carries a heavy DataLoader-ish
graph; the dump should be small AND the post-save in-memory object should
still have ``_trainer`` restored so predict() / continued training works.
"""

from __future__ import annotations

import logging
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from mlframe.training.io import save_mlframe_model


class _HeavyBlob:
    """Picklable payload-stand-in that's deliberately large enough to trip the 50 MB sensor."""

    def __init__(self, n_floats: int) -> None:
        # ~8 bytes per float64. 8 MB per 1M floats. We want >50 MB on disk,
        # so use 8M floats raw (~64 MB) and disable compression effectiveness
        # by filling with random bytes (zstd defaults to level 3; random data
        # compresses ~1:1).
        rng = np.random.default_rng(0)
        self.payload = rng.bytes(n_floats * 8)


def test_save_size_sensor_warns_above_threshold(caplog):
    """[save-size-sensor]: dumps > 50 MB trigger a hard WARN with the actionable hint."""

    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        big = _HeavyBlob(n_floats=8_000_000)  # ~64 MB random bytes -> ~64 MB on disk post-zstd
        with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
            ok = save_mlframe_model(big, fpath, verbose=0)
        assert ok is True
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        assert size_mb > 50.0, f"Test sentinel data didn't exceed 50 MB on disk (got {size_mb:.1f}); adjust n_floats."
        msgs = [rec.getMessage() for rec in caplog.records]
        assert any("[save-size-sensor]" in m for m in msgs), "Expected [save-size-sensor] WARN; got: " + " | ".join(msgs)
        # The hint must mention at least one actionable strip target.
        sensor_msg = next(m for m in msgs if "[save-size-sensor]" in m)
        assert "_trainer" in sensor_msg or "prediction_datamodule" in sensor_msg, f"Save-size sensor message must hint at strip targets; got: {sensor_msg!r}"
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_save_size_sensor_silent_below_threshold(caplog):
    """[save-size-sensor]: small dumps (<<50 MB) MUST NOT trigger the warn (no noise on healthy saves)."""

    tiny = SimpleNamespace(payload=np.zeros(100, dtype=np.float32))
    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
            ok = save_mlframe_model(tiny, fpath, verbose=0)
        assert ok is True
        msgs = [rec.getMessage() for rec in caplog.records]
        assert not any("[save-size-sensor]" in m for m in msgs), "Save-size sensor must stay silent for tiny dumps; got: " + " | ".join(msgs)
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


class _FakeLightningModule:
    """Mimics a LightningModule with a heavy ``_trainer`` attribute (DataLoader-shaped graph)."""

    def __init__(self) -> None:
        rng = np.random.default_rng(1)
        self._trainer = SimpleNamespace(
            # ~80 MB of "training data" the trainer would hold via DataLoader refs.
            train_dataloader_refs=rng.bytes(80 * 1024 * 1024),
        )
        # Small "weights" that we DO want to keep.
        self.state_dict_data = rng.standard_normal(1000).astype(np.float32)


class _FakeEstimator:
    """Mimics PytorchLightningEstimator: wraps a network + prediction_datamodule."""

    def __init__(self, network: _FakeLightningModule) -> None:
        self.network = network
        rng = np.random.default_rng(2)
        self.prediction_datamodule = SimpleNamespace(
            cached_dataset=rng.bytes(40 * 1024 * 1024),  # ~40 MB
        )


def test_save_size_sensor_auto_lean_retry_shrinks_oversized_dump(caplog):
    """E2.1 (2026-05-21): when a non-lean save exceeds the 50 MB sensor threshold
    AND the payload is a SimpleNamespace (lean is a no-op otherwise), the save
    auto-retries with ``lean=True`` and overwrites the file. Caller's payload
    is untouched (lean operates on a shallow copy)."""
    # Use 10M rows of random per-split arrays so the non-lean dump exceeds the 50 MB
    # sensor threshold after zstd-level-4 compression on random float32 data (~1:1).
    n_train = 10_000_000
    rng = np.random.default_rng(42)
    model_entry = SimpleNamespace(
        model=SimpleNamespace(some_small_state=np.zeros(100, dtype=np.float32)),
        test_preds=rng.standard_normal(1_000_000).astype(np.float32),
        test_probs=None,
        test_target=rng.standard_normal(1_000_000).astype(np.float32),
        val_preds=rng.standard_normal(1_000_000).astype(np.float32),
        val_probs=None,
        val_target=rng.standard_normal(1_000_000).astype(np.float32),
        train_preds=rng.standard_normal(n_train).astype(np.float32),  # ~40 MB
        train_probs=None,
        train_target=rng.standard_normal(n_train).astype(np.float32),  # ~40 MB
        oof_preds=rng.standard_normal(n_train).astype(np.float32),  # ~40 MB
        oof_probs=None,
        metrics={"test": {"R2": 0.95}},
        columns=["f0", "f1", "f2"],
        pre_pipeline=None,
        train_od_idx=None,
        val_od_idx=None,
        trainset_features_stats={f"f{i}": {"mean": 0.0, "std": 1.0} for i in range(50)},
    )

    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
            # Call WITHOUT lean=True; size-sensor should fire and auto-retry with lean.
            ok = save_mlframe_model(model_entry, fpath, verbose=1)
        assert ok is True
        size_mb_after = os.path.getsize(fpath) / (1024 * 1024)
        # Post auto-retry: lean strip removes preds/target/oof/stats; dump should be < 5 MB.
        assert size_mb_after < 5.0, (
            f"E2.1 auto_lean_retry failed: final dump is {size_mb_after:.1f} MB; expected the auto-retry with lean=True to strip the per-split arrays."
        )
        msgs = [r.getMessage() for r in caplog.records]
        # The E2.2 pre-pickle pre-check landed 2026-05-22 and supersedes the
        # post-save auto-retry on payloads it can detect upfront. Either path
        # is acceptable -- the contract is "oversized non-lean save ends up
        # small on disk", which the strict size assertion above already checks.
        assert any("auto-retrying with lean=True" in m or "[save-size-precheck]" in m for m in msgs), (
            f"E2.1: expected either the auto-retry log OR the pre-check log; got: {msgs}"
        )
        # Caller's in-memory payload must STILL have all fields (lean operates on a copy).
        assert model_entry.train_preds is not None
        assert model_entry.oof_preds is not None
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_pre_pickle_size_precheck_flips_lean_upfront(caplog):
    """E2.2 (2026-05-22): when ``pympler.asizeof`` on the in-memory payload
    exceeds ``auto_lean_pre_check_mb`` AND ``auto_lean_retry`` is on AND the
    caller didn't pin ``lean=`` already, the save flips lean=True BEFORE the
    fat pickle -- saves the fat-then-lean retry double-dump.

    Bench (`bench_pympler_pre_pickle_check.py`): asizeof=0.5ms vs fat save
    ~160ms at N=5M -- pre-check is essentially free and the savings on the
    naive case scale linearly with N."""
    # 10M-row payload: in-memory asizeof should comfortably exceed 100 MB
    # so the pre-check fires.
    n_train = 10_000_000
    rng = np.random.default_rng(42)
    model_entry = SimpleNamespace(
        model=SimpleNamespace(),
        train_preds=rng.standard_normal(n_train).astype(np.float32),
        train_target=rng.standard_normal(n_train).astype(np.float32),
        oof_preds=rng.standard_normal(n_train).astype(np.float32),
        metrics={},
        columns=[],
        pre_pipeline=None,
        test_preds=None,
        test_probs=None,
        test_target=None,
        val_preds=None,
        val_probs=None,
        val_target=None,
        train_probs=None,
        oof_probs=None,
        train_od_idx=None,
        val_od_idx=None,
        trainset_features_stats=None,
    )

    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
            ok = save_mlframe_model(model_entry, fpath, verbose=1)
        assert ok is True
        msgs = [r.getMessage() for r in caplog.records]
        assert any("[save-size-precheck]" in m for m in msgs), f"E2.2 pre-pickle pre-check did not fire on 10M-row payload; got msgs: {msgs}"
        # The pre-check WARN should NAME the threshold so operators see why it fired.
        precheck_msg = next(m for m in msgs if "[save-size-precheck]" in m)
        assert "flipping lean=True BEFORE the fat pickle" in precheck_msg
        # And the saved file should be SMALL (lean strip ran).
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        assert size_mb < 5.0, f"E2.2 pre-check fired but the saved dump is still {size_mb:.1f} MB; lean=True didn't actually strip the per-split arrays."
        # Caller's in-memory payload is UNCHANGED.
        assert model_entry.train_preds is not None
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_pre_pickle_precheck_disabled_when_threshold_zero(caplog):
    """E2.2 (negative): ``auto_lean_pre_check_mb=0`` disables the pre-check
    entirely. The post-save sensor + auto-retry path still kicks in if the
    fat dump exceeds 50 MB, so the user can keep the legacy 2-dump behaviour
    when they want full traceability."""
    n_train = 10_000_000
    rng = np.random.default_rng(42)
    model_entry = SimpleNamespace(
        model=SimpleNamespace(),
        train_preds=rng.standard_normal(n_train).astype(np.float32),
        train_target=rng.standard_normal(n_train).astype(np.float32),
        metrics={},
        columns=[],
        pre_pipeline=None,
        test_preds=None,
        test_probs=None,
        test_target=None,
        val_preds=None,
        val_probs=None,
        val_target=None,
        train_probs=None,
        oof_preds=None,
        oof_probs=None,
        train_od_idx=None,
        val_od_idx=None,
        trainset_features_stats=None,
    )
    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
            save_mlframe_model(model_entry, fpath, verbose=1, auto_lean_pre_check_mb=0.0)
        msgs = [r.getMessage() for r in caplog.records]
        # Pre-check WARN must NOT appear when disabled.
        assert not any("[save-size-precheck]" in m for m in msgs), f"E2.2 negative case: pre-check fired despite threshold=0. msgs: {msgs}"
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_save_size_sensor_no_auto_lean_when_disabled(caplog):
    """E2.1 (negative): when auto_lean_retry=False, the sensor fires and warns
    but does NOT overwrite the file with a lean variant."""
    # Larger payload so the sensor actually fires (>50 MB).
    n_train = 10_000_000
    rng = np.random.default_rng(42)
    model_entry = SimpleNamespace(
        model=SimpleNamespace(),
        train_preds=rng.standard_normal(n_train).astype(np.float32),
        train_target=rng.standard_normal(n_train).astype(np.float32),
        metrics={},
        columns=[],
        pre_pipeline=None,
        test_preds=None,
        test_probs=None,
        test_target=None,
        val_preds=None,
        val_probs=None,
        val_target=None,
        train_probs=None,
        oof_preds=None,
        oof_probs=None,
        train_od_idx=None,
        val_od_idx=None,
        trainset_features_stats=None,
    )

    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
            ok = save_mlframe_model(model_entry, fpath, verbose=1, auto_lean_retry=False)
        assert ok is True
        msgs = [r.getMessage() for r in caplog.records]
        # Sensor WARN fires; auto-retry WARN does NOT.
        assert any("[save-size-sensor]" in m for m in msgs)
        assert not any("auto-retrying with lean=True" in m for m in msgs)
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_lean_strip_covers_oof_preds_and_probs():
    """P0 #2 follow-up: ``oof_preds`` / ``oof_probs`` are large per-row arrays
    stamped on the model SimpleNamespace at trainer.py:955 when
    ``oof_n_splits >= 2``. They must be in _LEAN_STRIP_FIELDS so lean saves
    don't leak ~16-32 MB of OOF data per model on 4M-row training as soon as
    the caller flips oof_n_splits >= 2."""
    from mlframe.training.io import _LEAN_STRIP_FIELDS

    assert "oof_preds" in _LEAN_STRIP_FIELDS, (
        "oof_preds missing from _LEAN_STRIP_FIELDS -- lean saves will leak it whenever the caller stamps OOF on the model entry."
    )
    assert "oof_probs" in _LEAN_STRIP_FIELDS, "oof_probs missing from _LEAN_STRIP_FIELDS -- same risk on classifier paths."


def test_lean_save_strips_large_per_split_arrays_under_threshold():
    """2026-05-21 P0 #2: TVT prod log MLP dump = 135.8 MB on 4M-row training.
    Root cause: ``train_eval.py:895`` called save_mlframe_model WITHOUT
    ``lean=True``, so train_preds + train_target + trainset_features_stats
    (large per-row arrays at full-train cardinality) landed in the dump.

    Verifies the lean strip drops the documented field set so the saved file
    fits under the 50 MB sensor threshold on a 4M-row scenario."""
    n_train = 4_000_000
    rng = np.random.default_rng(42)
    # Synthesise the exact attribute layout trainer._train_and_evaluate_helper returns
    # (SimpleNamespace with the lean-stripped fields).
    model_entry = SimpleNamespace(
        model=SimpleNamespace(some_small_state=np.zeros(100, dtype=np.float32)),
        test_preds=rng.standard_normal(500_000).astype(np.float32),
        test_probs=None,
        test_target=rng.standard_normal(500_000).astype(np.float32),
        val_preds=rng.standard_normal(450_000).astype(np.float32),
        val_probs=None,
        val_target=rng.standard_normal(450_000).astype(np.float32),
        train_preds=rng.standard_normal(n_train).astype(np.float32),  # ~16 MB
        train_probs=None,
        train_target=rng.standard_normal(n_train).astype(np.float32),  # ~16 MB
        oof_preds=None,
        oof_probs=None,
        metrics={"test": {"R2": 0.95}, "val": {"R2": 0.94}, "train": {"R2": 0.97}},
        columns=["f0", "f1", "f2"],
        pre_pipeline=None,
        train_od_idx=None,
        val_od_idx=None,
        trainset_features_stats={f"f{i}": {"mean": 0.0, "std": 1.0} for i in range(50)},
    )

    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        ok = save_mlframe_model(model_entry, fpath, verbose=0, lean=True)
        assert ok is True
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        # With train/val/test preds + targets + features_stats stripped, only the
        # tiny model + metrics + columns remain -- well under 1 MB on this synthetic.
        assert size_mb < 1.0, (
            f"lean save did not strip the per-split arrays as expected; got {size_mb:.1f} MB. "
            f"Check that _LEAN_STRIP_FIELDS covers train_preds / train_target / trainset_features_stats."
        )
        # Sanity: in-memory entry STILL has the stripped fields (lean operates on a shallow copy).
        assert model_entry.train_preds is not None
        assert model_entry.train_target is not None
        assert model_entry.trainset_features_stats is not None
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)


def test_lightning_bloat_strip_shrinks_dump_and_restores_payload():
    """The bloat strip nullifies ``_trainer`` + ``prediction_datamodule`` during pickle, restoring
    them on the in-memory payload afterward. Dump must be tiny (<5 MB) despite the in-memory
    payload weighing ~120 MB."""

    net = _FakeLightningModule()
    est = _FakeEstimator(net)
    payload = SimpleNamespace(model=est)

    # Pre-save sanity: confirm the heavy attrs are present in memory.
    assert payload.model.network._trainer is not None
    assert payload.model.prediction_datamodule is not None

    with tempfile.NamedTemporaryFile(suffix=".dump", delete=False) as tf:
        fpath = tf.name
    try:
        ok = save_mlframe_model(payload, fpath, verbose=0)
        assert ok is True
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        # In-memory the payload is ~120 MB. Strip should drop _trainer (80 MB) AND
        # prediction_datamodule (40 MB), leaving only the tiny state-dict (~4 KB).
        # zstd on random bytes is near-no-op, so without the strip we'd see ~120 MB on disk.
        assert size_mb < 5.0, (
            f"Dump too big ({size_mb:.1f} MB): Lightning bloat strip didn't fire. Expected <5 MB after stripping ``_trainer`` + ``prediction_datamodule``."
        )

        # Post-save: in-memory caller must still have BOTH attrs (the strip is transient).
        assert payload.model.network._trainer is not None, "Bloat strip failed to restore _trainer on the caller's payload."
        assert payload.model.prediction_datamodule is not None, "Bloat strip failed to restore prediction_datamodule on the caller's payload."
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)
