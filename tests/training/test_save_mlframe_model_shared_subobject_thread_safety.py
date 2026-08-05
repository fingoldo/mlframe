"""TRAINING_LOOSE_A-3 regression test: save_mlframe_model's pre-pickle strip/restore of a SHARED nested
sub-object must be serialized (not interleaved) across concurrent calls.

The bug (fixed): _collect_pre_dump_swaps mutates the shared nested object graph in place (torch.compile
swap, Lightning-bloat nulling) and restores only in a `finally` block; `lean=True` only shallow-copies the
top-level SimpleNamespace, so nested sub-objects are the same live references the caller's model still
holds. Two concurrent save_mlframe_model calls on bundles sharing a nested sub-object could interleave
their strip/restore windows -- one call's dump could see the other's stripped/nulled state, or a restore
could race a strip, potentially leaving the shared sub-object permanently stripped after both calls return.
"""

from __future__ import annotations

import os
import tempfile
import threading
from types import SimpleNamespace

import pytest

from mlframe.training.io import save_mlframe_model

pytestmark = pytest.mark.fast


class _FakeLightningModule:
    """Duck-types as a lightning.LightningModule for _looks_like_training_bloat's type check."""

    __module__ = "lightning.pytorch.core.module"


class _FakeTrainer(_FakeLightningModule):
    """Duck-types as a lightning Trainer -- class name ends with 'Trainer'."""


def _make_shared_nested_bundle():
    """A nested sub-object (with a Lightning-bloat-shaped attribute) shared by reference across two
    top-level model bundles, mimicking two per-model saves referencing a shared preprocessing pipeline."""
    shared_inner = SimpleNamespace(_trainer=_FakeTrainer(), weight=1.23)
    bundle_a = SimpleNamespace(name="a", inner=shared_inner)
    bundle_b = SimpleNamespace(name="b", inner=shared_inner)
    return shared_inner, bundle_a, bundle_b


def test_concurrent_saves_of_bundles_sharing_a_nested_subobject_do_not_corrupt_it():
    """Many threads repeatedly saving two bundles that share a nested sub-object must never leave the
    shared sub-object's stripped attribute permanently nulled after all saves complete."""
    shared_inner, bundle_a, bundle_b = _make_shared_nested_bundle()
    errors: list[BaseException] = []

    with tempfile.TemporaryDirectory() as tmpdir:

        def _worker(idx):
            """Repeatedly save bundle_a and bundle_b (sharing `shared_inner`) from multiple threads."""
            try:
                for i in range(15):
                    fpath = os.path.join(tmpdir, f"m_{idx}_{i}.dump")
                    ok_a = save_mlframe_model(bundle_a, fpath + ".a", verbose=0)
                    ok_b = save_mlframe_model(bundle_b, fpath + ".b", verbose=0)
                    assert ok_a and ok_b
            except BaseException as e:  # must capture ANY exception a race could raise
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(t,)) for t in range(6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

    assert not errors, f"concurrent saves of a shared nested sub-object raised: {errors}"
    # The shared sub-object's stripped Lightning-bloat attr must be restored (not left None) after
    # every concurrent save completes -- a corrupted restore would silently break subsequent predict/fit
    # on the caller's live in-memory object.
    assert shared_inner._trainer is not None
    assert isinstance(shared_inner._trainer, _FakeTrainer)
    assert shared_inner.weight == 1.23
