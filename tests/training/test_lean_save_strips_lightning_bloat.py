"""Lean save must strip Lightning Trainer / DataModule / DataLoader bloat.

A 2.4 GB MLP dump survived the name-only strip because the object carrying
the 4M-row training dataset was a Lightning Trainer held under a non-canonical
attr name. The strip now also matches BY TYPE (module lightning* / torch.utils
.data + class Trainer/*DataModule/*DataLoader/*Dataset), so the heavy training
objects never reach the pickle regardless of the attr name. The caller's
in-memory object is restored after the save.

Uses INCOMPRESSIBLE random arrays so the on-disk size is a faithful proxy for
"did the payload survive" (zstd would crush an all-zeros array regardless of
the strip). We assert via dump SIZE (not load): the loader's _SafeUnpickler
allowlist intentionally blocks arbitrary test classes, and size already
proves the strip (bloat dropped, small weights kept).
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from mlframe.training.io import save_mlframe_model

_RNG = np.random.default_rng(0)
_BLOAT_N = 5_000_000          # ~40 MB float64, incompressible
_WEIGHTS_N = 200_000          # ~0.8 MB float32


class _FakeTrainer:
    __module__ = "lightning.pytorch.trainer.trainer"

    def __init__(self):
        self.big = _RNG.standard_normal(_BLOAT_N)  # retained dataloaders proxy


class _FakeDataModule:
    __module__ = "lightning.pytorch.core.datamodule"

    def __init__(self):
        self.big = _RNG.standard_normal(_BLOAT_N)


class _Weights:
    """Stand-in for the fitted network -- small, MUST survive the strip."""
    def __init__(self):
        self.w = _RNG.standard_normal(_WEIGHTS_N).astype(np.float32)


class _Wrapper:
    def __init__(self):
        self.estimator_ = _Weights()                     # must survive
        self.some_lightning_trainer = _FakeTrainer()     # non-canonical name
        self.prediction_datamodule = _FakeDataModule()   # canonical name


def _dump_size(obj) -> int:
    with tempfile.TemporaryDirectory() as d:
        fpath = os.path.join(d, "m.dump")
        save_mlframe_model(obj, fpath, lean=True, verbose=0)
        return os.path.getsize(fpath)


def test_lean_save_strips_heavy_lightning_objects_by_type() -> None:
    w = _Wrapper()
    bloat_bytes = w.some_lightning_trainer.big.nbytes  # ~40 MB
    weights_bytes = w.estimator_.w.nbytes              # ~0.8 MB
    size = _dump_size(w)
    # Stripped: dump must be far below even ONE bloat array (two are held).
    assert size < bloat_bytes // 4, (
        f"dump {size} retains bloat (one trainer array = {bloat_bytes})"
    )
    # But the small fitted weights MUST survive (dump not ~empty).
    assert size > weights_bytes // 2, (
        f"dump {size} too small -- weights stripped too (weights={weights_bytes})"
    )
    # Caller's in-memory object is restored after save (strip is temporary).
    assert w.some_lightning_trainer is not None
    assert w.prediction_datamodule is not None
    assert isinstance(w.some_lightning_trainer.big, np.ndarray)
    assert w.estimator_.w.shape == (_WEIGHTS_N,)


def test_plain_heavy_object_is_not_stripped() -> None:
    # Only training-bloat TYPES are stripped; a plain heavy attr survives.
    class _Plain:
        def __init__(self):
            self.payload = _RNG.standard_normal(_BLOAT_N)

    class _Holder:
        def __init__(self):
            self.keep = _Plain()              # plain -> survives (heavy dump)
            self.trainer = _FakeTrainer()     # canonical name -> stripped

    h = _Holder()
    bloat_bytes = h.keep.payload.nbytes
    size = _dump_size(h)
    # The plain payload survives, so the dump is at least ~its size; the
    # trainer's array is gone, so it is well under TWO bloat arrays.
    assert size > bloat_bytes // 2, f"plain heavy object wrongly stripped ({size})"
    assert size < bloat_bytes * 2, f"trainer not stripped ({size} vs {bloat_bytes})"
