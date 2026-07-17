"""Regression sensor: ``atomic_write_bytes`` and ``save_mlframe_model``
must support the ``fsync=`` / ``durable=`` kwarg to skip the ~400ms
per-file ``nt.fsync`` cost on Windows.

The 2026-05-19 multi-model full-pipeline profile (lgb+xgb x binary x
60k, seed=20260527) surfaced this:

    save wall: 7.11s  models_saved=15  total=19.06 MB
    {built-in method nt.fsync}:  15 calls, 6.086s tottime (86% of save wall)

The fix gates the fsync on a kwarg (default True for safety):

* ``atomic_write_bytes(..., fsync: bool = True)``
* ``save_mlframe_model(..., durable: bool = True)`` forwards to ``fsync``.

The harness ``_profile_fuzz_1m`` passes ``durable=False`` for its
tempdir-bound bench saves. Production saves remain crash-durable.

This sensor pins the kwarg surface; a regression that drops the kwarg
or hard-codes the fsync call would fail.
"""

from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path



def test_atomic_write_bytes_accepts_fsync_kwarg():
    """The kwarg must be keyword-only and default to False (perf-over-
    legacy policy 2026-05-20: per-file fsync on Windows costs ~400ms and
    the ``write-tmp-then-rename`` atomicity holds without it; only the
    post-rename DURABILITY window is shortened, and the worst case for
    ML model bundles is 'retrain', not 'corruption')."""
    from mlframe.training.io import atomic_write_bytes

    sig = inspect.signature(atomic_write_bytes)
    assert "fsync" in sig.parameters, "atomic_write_bytes lost the ``fsync`` kwarg; the per-file 400ms nt.fsync cost on Windows can no longer be opted into."
    p = sig.parameters["fsync"]
    assert p.default is False, (
        f"atomic_write_bytes ``fsync`` default changed from False to {p.default!r}; "
        "post-2026-05-20 the default is fast-by-default (skip fsync); "
        "callers needing crash durability must opt in explicitly."
    )
    assert p.kind == inspect.Parameter.KEYWORD_ONLY, f"atomic_write_bytes ``fsync`` must be keyword-only (got {p.kind})."


def test_save_mlframe_model_accepts_durable_kwarg():
    """``save_mlframe_model`` must accept ``durable=True/False`` and forward
    it to atomic_write_bytes.fsync. Default flipped to False 2026-05-20
    so production save speed wins by default (5x measured); callers who
    need crash-durable saves pass ``durable=True`` explicitly."""
    from mlframe.training.io import save_mlframe_model

    sig = inspect.signature(save_mlframe_model)
    assert "durable" in sig.parameters, (
        "save_mlframe_model lost the ``durable`` kwarg; bench / test paths can no longer opt out of the per-file 400ms fsync cost."
    )
    p = sig.parameters["durable"]
    assert p.default is False, (
        f"save_mlframe_model ``durable`` default changed from False to {p.default!r}; "
        "post-2026-05-20 the default is fast-by-default (no fsync); "
        "callers needing crash durability must opt in explicitly."
    )


def test_save_mlframe_model_durable_false_skips_fsync(monkeypatch, tmp_path):
    """When called with ``durable=False`` the underlying os.fsync MUST NOT
    fire. We monkeypatch os.fsync with a counter and verify the count."""
    from mlframe.training import io

    fsync_calls = {"n": 0}
    _orig_fsync = os.fsync

    def _spy_fsync(fd):
        fsync_calls["n"] += 1
        return _orig_fsync(fd)

    monkeypatch.setattr(io.os, "fsync", _spy_fsync)

    # Save a trivial object (avoid the dill heavy walk to keep the test
    # fast and focused on the fsync gate).
    from types import SimpleNamespace

    model = SimpleNamespace(payload=[1, 2, 3], name="sensor")

    out_path = tmp_path / "sensor.dump"
    ok = io.save_mlframe_model(
        model,
        str(out_path),
        verbose=0,
        lean=False,
        durable=False,
    )
    assert ok, "save_mlframe_model returned False on durable=False"
    assert out_path.exists(), "save_mlframe_model did not produce the output file"
    assert fsync_calls["n"] == 0, f"durable=False called os.fsync {fsync_calls['n']} times; expected 0. The fsync skip kwarg is broken."


def test_save_mlframe_model_durable_true_does_fsync(monkeypatch, tmp_path):
    """Explicit ``durable=True`` MUST call os.fsync (callers that need
    crash durability opt in by passing the kwarg; a regression that
    silently drops the call when explicitly requested would weaken
    their guarantee)."""
    from mlframe.training import io

    fsync_calls = {"n": 0}
    _orig_fsync = os.fsync

    def _spy_fsync(fd):
        fsync_calls["n"] += 1
        return _orig_fsync(fd)

    monkeypatch.setattr(io.os, "fsync", _spy_fsync)

    from types import SimpleNamespace

    model = SimpleNamespace(payload=[1, 2, 3], name="sensor")

    out_path = tmp_path / "sensor.dump"
    ok = io.save_mlframe_model(
        model,
        str(out_path),
        verbose=0,
        lean=False,
        durable=True,
    )
    assert ok
    assert fsync_calls["n"] >= 1, (
        "save_mlframe_model with explicit durable=True did NOT call os.fsync; callers that opt-in to crash-durability now silently lose it."
    )


def test_save_mlframe_model_default_skips_fsync(monkeypatch, tmp_path):
    """Default behaviour (no kwarg passed) MUST skip os.fsync. A regression
    that flips the default back to True would re-introduce the 400ms-per-
    file Windows penalty for every production save (5x save-wall regression
    on the 15-model multi-model fuzz profile that surfaced the fix)."""
    from mlframe.training import io

    fsync_calls = {"n": 0}
    _orig_fsync = os.fsync

    def _spy_fsync(fd):
        fsync_calls["n"] += 1
        return _orig_fsync(fd)

    monkeypatch.setattr(io.os, "fsync", _spy_fsync)

    from types import SimpleNamespace

    model = SimpleNamespace(payload=[1, 2, 3], name="sensor")

    out_path = tmp_path / "sensor_default.dump"
    ok = io.save_mlframe_model(model, str(out_path), verbose=0, lean=False)  # default durable=False
    assert ok
    assert fsync_calls["n"] == 0, (
        f"save_mlframe_model default behaviour called os.fsync "
        f"{fsync_calls['n']} times; expected 0. The fast-by-default policy "
        f"has regressed -- check that durable's default is still False."
    )


def test_save_roundtrip_with_durable_false_still_loads():
    """The fsync skip MUST NOT compromise the saved bundle's loadability
    in-process. Save with durable=False, load back, assert payload matches."""
    from mlframe.training.io import save_mlframe_model, load_mlframe_model

    from types import SimpleNamespace

    model = SimpleNamespace(payload=["alpha", "beta", "gamma"], counter=42)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "roundtrip.dump"
        assert save_mlframe_model(
            model,
            str(out_path),
            verbose=0,
            lean=False,
            durable=False,
        )
        loaded = load_mlframe_model(str(out_path))
        assert loaded is not None
        assert loaded.payload == model.payload
        assert loaded.counter == model.counter
