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
        save_mlframe_model,
        _meta_sidecar_path,
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
    assert "mlframe" in meta["lib_versions"], "lib_versions must include mlframe.__version__"


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
    version_warns = [r for r in caplog.records if "version" in r.message.lower() or "drift" in r.message.lower()]
    assert version_warns == [], f"legacy bundles must load silently; got: {[r.message for r in version_warns]}"


def test_lib_version_drift_warns(tmp_bundle_dir, caplog):
    """Manually fudge the sidecar to claim a different mlframe version,
    then load -- the drift must surface as a WARN."""
    from mlframe.training.io import (
        save_mlframe_model,
        load_mlframe_model,
        _meta_sidecar_path,
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
    drift_warns = [r for r in caplog.records if "drift" in r.message.lower() and "mlframe" in r.message.lower()]
    assert drift_warns, f"expected WARN naming mlframe version drift; got: {[r.message for r in caplog.records]}"


def test_lib_version_drift_strict_raises(tmp_bundle_dir):
    """``strict_version=True`` must raise (not just WARN) on drift."""
    from mlframe.training.io import (
        save_mlframe_model,
        load_mlframe_model,
        _meta_sidecar_path,
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
        save_mlframe_model,
        load_mlframe_model,
        _meta_sidecar_path,
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
    assert any("failed to read" in rec.message for rec in caplog.records), f"expected WARN on corrupt sidecar; got: {[r.message for r in caplog.records]}"


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
    assert boosters & set(libs), f"none of {boosters} found in lib_versions snapshot {libs}"


def test_collect_lib_versions_does_not_force_import_absent_lib(monkeypatch):
    """Regression: the version sidecar must NEVER force-load a heavy optional
    dep. The pre-fix code did ``__import__(name)`` for each listed lib, which
    on a lean tree-only / CPU-serving process paid ~14s of cold imports and
    pulled the whole neural stack resident purely to stamp version strings.

    We inject a sentinel that is IMPORTABLE (a stdlib module not yet loaded in
    this process) but has no distribution metadata under its import-name. Post-fix
    ``_collect_lib_versions`` consults ``importlib.metadata`` (no import side
    effect) + a sys.modules fallback, so the sentinel stays absent from
    sys.modules. Pre-fix ``__import__(name)`` would force-load it -- exactly the
    mechanism that dragged the whole neural stack (torch/transformers) resident
    on a lean serving process just to stamp version strings.

    Fail-on-pre-fix: verified that pre-fix code (`__import__('imaplib')`) loads
    the sentinel into sys.modules, so this assertion FAILS pre-fix and PASSES
    post-fix.
    """
    import sys
    from mlframe.training import io as _io

    # Importable stdlib module, plausibly absent from the test process. If it is
    # already imported we cannot tell forced from pre-existing -> use a fresh one.
    sentinel = "imaplib"
    if sentinel in sys.modules:
        sentinel = "telnetlib"
    if sentinel in sys.modules:
        pytest.skip("no importable-but-unloaded stdlib sentinel available")
    # Splice the sentinel into the lib list under an import-name with no matching
    # distribution metadata. A correct (no-side-effect) implementation records
    # nothing for it and -- crucially -- never imports it.
    patched = _io._LIB_VERSION_DISTS + ((sentinel, sentinel),)
    monkeypatch.setattr(_io, "_LIB_VERSION_DISTS", patched)

    libs = _io._collect_lib_versions()

    assert sentinel not in sys.modules, (
        "version sidecar must not import an absent lib; pre-fix __import__ path "
        "force-loaded the heavy optional stack (torch/transformers) just to "
        "stamp versions"
    )
    assert sentinel not in libs  # no metadata, not loaded -> absent/None -> omitted
    # Real installed libs are still reported correctly without the import cost.
    assert libs.get("numpy") and libs.get("mlframe")


def test_collect_lib_versions_does_not_load_uninstalled_heavy_dep(monkeypatch):
    """Pin the exact pre-fix failure mode: an installed-but-not-yet-imported
    heavy lib must NOT be dragged into sys.modules by the sidecar.

    Pick a lib from the list that is plausibly absent from the test process.
    If torch happens to already be imported we skip (the assertion can't
    distinguish forced from pre-existing). Pre-fix ``__import__('torch')`` would
    load it; post-fix ``importlib.metadata.version('torch')`` reads metadata
    only -- torch stays out of sys.modules.
    """
    import sys
    from mlframe.training import io as _io

    heavy = "torch"
    if heavy in sys.modules:
        pytest.skip("torch already imported in this process; cannot assert non-load")

    _io._collect_lib_versions()
    assert heavy not in sys.modules, "pre-fix __import__('torch') would force-load torch; post-fix must read metadata only and leave torch out of sys.modules"


def test_collect_lib_versions_memoised_avoids_metadata_reparse(monkeypatch):
    """``_collect_lib_versions`` re-parsed ~15 RFC822 METADATA files via
    ``importlib.metadata.version`` on EVERY save AND every load (~10 ms/call --
    ~83% of a small-bundle load wall). Installed metadata is immutable for the
    process lifetime, so the result is memoised per ``_LIB_VERSION_DISTS``
    identity. This pins: the second call does NOT hit ``importlib.metadata.version``.

    Fail-on-pre-fix: pre-fix code called ``_md.version`` once per dist on every
    invocation, so the call counter would be > the dist-list length after two
    calls; post-fix the second call is served from the memo and adds nothing.
    """
    from importlib import metadata as _md
    from mlframe.training import io as _io

    _io._lib_versions_cache_clear()
    calls = {"n": 0}
    _real = _md.version

    def _counting_version(name):
        calls["n"] += 1
        return _real(name)

    monkeypatch.setattr(_md, "version", _counting_version)

    first = _io._collect_lib_versions()
    after_first = calls["n"]
    assert after_first > 0, "first call must consult importlib.metadata"
    second = _io._collect_lib_versions()
    assert calls["n"] == after_first, (
        f"second call must be served from the memo without re-parsing METADATA (saw {calls['n'] - after_first} extra importlib.metadata.version calls)"
    )
    assert first == second
    # Callers may mutate the returned dict; the memo must be insulated.
    second["__scratch__"] = "x"
    assert "__scratch__" not in _io._collect_lib_versions()


def test_collect_lib_versions_memo_respects_dist_list_swap(monkeypatch):
    """The memo is keyed on ``_LIB_VERSION_DISTS`` identity, so a test that
    monkeypatches a different dist tuple recomputes (no stale memo leakage)."""
    from mlframe.training import io as _io

    _io._lib_versions_cache_clear()
    base = _io._collect_lib_versions()
    assert "numpy" in base
    # Swap to a one-entry list -> new tuple identity -> memo miss -> recompute.
    monkeypatch.setattr(_io, "_LIB_VERSION_DISTS", (("numpy", "numpy"),))
    swapped = _io._collect_lib_versions()
    assert set(swapped) == {"numpy"}, f"memo must recompute for a swapped dist list; got {sorted(swapped)}"
