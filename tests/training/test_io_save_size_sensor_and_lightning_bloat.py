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
        assert any("[save-size-sensor]" in m for m in msgs), (
            "Expected [save-size-sensor] WARN; got: " + " | ".join(msgs)
        )
        # The hint must mention at least one actionable strip target.
        sensor_msg = next(m for m in msgs if "[save-size-sensor]" in m)
        assert "_trainer" in sensor_msg or "prediction_datamodule" in sensor_msg, (
            f"Save-size sensor message must hint at strip targets; got: {sensor_msg!r}"
        )
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
        assert not any("[save-size-sensor]" in m for m in msgs), (
            "Save-size sensor must stay silent for tiny dumps; got: " + " | ".join(msgs)
        )
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
        oof_preds=None, oof_probs=None,
        metrics={"test": {"R2": 0.95}, "val": {"R2": 0.94}, "train": {"R2": 0.97}},
        columns=["f0", "f1", "f2"],
        pre_pipeline=None,
        train_od_idx=None, val_od_idx=None,
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
            f"Dump too big ({size_mb:.1f} MB): Lightning bloat strip didn't fire. "
            f"Expected <5 MB after stripping ``_trainer`` + ``prediction_datamodule``."
        )

        # Post-save: in-memory caller must still have BOTH attrs (the strip is transient).
        assert payload.model.network._trainer is not None, (
            "Bloat strip failed to restore _trainer on the caller's payload."
        )
        assert payload.model.prediction_datamodule is not None, (
            "Bloat strip failed to restore prediction_datamodule on the caller's payload."
        )
    finally:
        if os.path.exists(fpath):
            os.remove(fpath)
