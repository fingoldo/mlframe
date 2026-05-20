"""Wave-19 P0 #1 sensor: save_mlframe_model writes .meta.json sidecar; load
validates library-version drift.

Pre-fix (before 2026-05-20), save_mlframe_model dumped the model bundle
via pickle/dill+zstd with NO mlframe.__version__ stamp and NO library-
version envelope. Two-line version skew (catboost 1.2 -> 1.3 booster
internals, mlframe v0.90 -> v0.91 attribute rename) silently unpickled
and raised a cryptic AttributeError deep inside predict().

Post-fix: a sibling .meta.json sidecar carries
``{sidecar_version, saved_at_utc, lib_versions: {mlframe, lightgbm,
catboost, xgboost, ...}}``. load_mlframe_model calls
``validate_load_meta_sidecar`` before unpickling; library-version drift
is WARN-logged by default, or raises with ``strict_version=True``.

The sidecar is best-effort:
- Sidecar IO failure during save does NOT block the bundle write.
- Missing sidecar on load (legacy bundle) does NOT block the load --
  back-compat by design.
- Lib version drift is WARN-only by default (booster libs are
  typically forward-compatible for minor versions).
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

import orjson
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def tmp_bundle_dir():
    d = tempfile.mkdtemp(prefix="mlframe_io_test_")
    try:
        yield Path(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_sidecar_written_on_save(tmp_bundle_dir):
    from mlframe.training.io import (
        save_mlframe_model, _meta_sidecar_path,
    )
    bundle = tmp_bundle_dir / "test.dump"
    model = SimpleNamespace(name="hello", values=[1, 2, 3])
    assert save_mlframe_model(model, str(bundle), verbose=0) is True
    sidecar = Path(_meta_sidecar_path(str(bundle)))
    assert sidecar.exists(), (
        f"Wave 19 P0 #1 regression: .meta.json sidecar was NOT written "
        f"next to {bundle}. Bundle has no library-version envelope; "
        f"booster minor upgrades will silently break loads."
    )


def test_sidecar_fields_present(tmp_bundle_dir):
    from mlframe.training.io import save_mlframe_model, load_save_meta_sidecar
    bundle = tmp_bundle_dir / "test.dump"
    save_mlframe_model(SimpleNamespace(x=1), str(bundle), verbose=0)
    meta = load_save_meta_sidecar(str(bundle))
    assert meta is not None
    assert meta.get("sidecar_version") == 1
    assert "saved_at_utc" in meta
    assert "lib_versions" in meta
    # mlframe must always be recorded (the package being saved).
    assert "mlframe" in meta["lib_versions"], (
        "lib_versions must include mlframe.__version__"
    )


def test_legacy_bundle_no_sidecar_loads_silently(tmp_bundle_dir, caplog):
    """Pre-2026-05-20 bundles have no sidecar. Load must succeed silently
    (back-compat is the contract)."""
    from mlframe.training.io import save_mlframe_model, load_mlframe_model, _meta_sidecar_path
    bundle = tmp_bundle_dir / "test.dump"
    save_mlframe_model(SimpleNamespace(payload=42), str(bundle), verbose=0)
    # Delete sidecar to simulate legacy bundle.
    sidecar = Path(_meta_sidecar_path(str(bundle)))
    sidecar.unlink()
    assert not sidecar.exists()
    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        loaded = load_mlframe_model(str(bundle))
    assert loaded is not None
    assert loaded.payload == 42
    # No version-related WARN should fire when sidecar is absent.
    version_warns = [
        r for r in caplog.records
        if "version" in r.message.lower() or "drift" in r.message.lower()
    ]
    assert version_warns == [], (
        f"legacy bundles must load silently; got: "
        f"{[r.message for r in version_warns]}"
    )


def test_lib_version_drift_warns(tmp_bundle_dir, caplog):
    """Manually fudge the sidecar to claim a different mlframe version,
    then load -- the drift must surface as a WARN."""
    from mlframe.training.io import (
        save_mlframe_model, load_mlframe_model, _meta_sidecar_path,
    )
    bundle = tmp_bundle_dir / "test.dump"
    save_mlframe_model(SimpleNamespace(payload=7), str(bundle), verbose=0)
    sidecar = Path(_meta_sidecar_path(str(bundle)))
    meta = orjson.loads(sidecar.read_bytes())
    meta["lib_versions"]["mlframe"] = "0.001-FAKE-OLD"
    sidecar.write_bytes(orjson.dumps(meta))

    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        loaded = load_mlframe_model(str(bundle))
    assert loaded is not None
    assert loaded.payload == 7
    drift_warns = [
        r for r in caplog.records
        if "drift" in r.message.lower() and "mlframe" in r.message.lower()
    ]
    assert drift_warns, (
        f"expected WARN naming mlframe version drift; got: "
        f"{[r.message for r in caplog.records]}"
    )


def test_lib_version_drift_strict_raises(tmp_bundle_dir):
    """``strict_version=True`` must raise (not just WARN) on drift."""
    from mlframe.training.io import (
        save_mlframe_model, load_mlframe_model, _meta_sidecar_path,
    )
    bundle = tmp_bundle_dir / "test.dump"
    save_mlframe_model(SimpleNamespace(payload=7), str(bundle), verbose=0)
    sidecar = Path(_meta_sidecar_path(str(bundle)))
    meta = orjson.loads(sidecar.read_bytes())
    meta["lib_versions"]["mlframe"] = "0.001-FAKE-OLD"
    sidecar.write_bytes(orjson.dumps(meta))

    with pytest.raises(ValueError, match="drift detected"):
        load_mlframe_model(str(bundle), strict_version=True)


def test_corrupt_sidecar_falls_back_to_warn(tmp_bundle_dir, caplog):
    """Corrupt sidecar JSON: WARN, then proceed (bundle may still load)."""
    from mlframe.training.io import (
        save_mlframe_model, load_mlframe_model, _meta_sidecar_path,
    )
    bundle = tmp_bundle_dir / "test.dump"
    save_mlframe_model(SimpleNamespace(payload=99), str(bundle), verbose=0)
    sidecar = Path(_meta_sidecar_path(str(bundle)))
    sidecar.write_text("{ this is not JSON", encoding="utf-8")  # corrupt

    with caplog.at_level(logging.WARNING, logger="mlframe.training.io"):
        loaded = load_mlframe_model(str(bundle))
    assert loaded is not None
    assert loaded.payload == 99
    # WARN must fire naming the read failure.
    assert any(
        "failed to read" in rec.message
        for rec in caplog.records
    ), f"expected WARN on corrupt sidecar; got: {[r.message for r in caplog.records]}"


def test_collect_lib_versions_includes_required():
    """The lib_versions snapshot must include at least mlframe + numpy +
    one booster library that's actually installed (so the drift check
    has something to compare)."""
    from mlframe.training.io import _collect_lib_versions
    libs = _collect_lib_versions()
    assert "mlframe" in libs
    assert "numpy" in libs
    # At least one booster should be present; pin nothing specific.
    boosters = {"lightgbm", "xgboost", "catboost"}
    assert boosters & set(libs), (
        f"none of {boosters} found in lib_versions snapshot {libs}"
    )
