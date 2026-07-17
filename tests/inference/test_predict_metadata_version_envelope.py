"""Wave-19 P0 #2 sensor: metadata schema_version + composite_target_env_signature
must be validated at predict-load.

Pre-fix (before 2026-05-20) the write side at
``_phase_config_setup.py:312`` populated ``metadata["schema_version"] = 2``
and ``_phase_helpers.py:253`` populated
``metadata["composite_target_env_signature"]`` -- but the READ side at
``predict.py:649-659`` and ``predict.py:2047-2058`` (two predict entry
points) NEVER validated either field. The version stamp was dead-code
file-size cost with zero protection.

This sensor pins the post-fix contract: ``_validate_metadata_version_envelope``
runs at every metadata load and enforces:

- Missing schema_version + no composite specs -> v1 legacy, INFO accept.
- Missing schema_version + composite specs PRESENT -> raise (composite
  contract changed at v2; loading would silently apply wrong semantics).
- schema_version not in {1, 2} -> raise (unsupported).
- schema_version < current -> WARN, continue.
- composite_target_env_signature drift vs live env_signature() -> WARN.
"""

from __future__ import annotations

import logging

import pytest


def test_legacy_v1_bundle_no_composite_accepted():
    """Bundles written before schema_version=2 existed must still load
    (back-compat). INFO-level message, no warning, no raise."""
    from mlframe.training.core.predict import _validate_metadata_version_envelope

    # Empty metadata dict represents a stripped-down legacy bundle.
    _validate_metadata_version_envelope({}, "fake/legacy/path")


def test_legacy_bundle_with_composite_specs_raises():
    """If composite_target_specs are present without schema_version, the
    v1 vs v2 semantic difference is unknowable -- refuse to load rather
    than silently applying potentially-wrong spec interpretation."""
    from mlframe.training.core.predict import _validate_metadata_version_envelope

    with pytest.raises(ValueError, match="composite-spec contract requires schema_version >= 2"):
        _validate_metadata_version_envelope(
            {"composite_target_specs": {"target1": {"name": "a"}}},
            "fake/path",
        )


def test_unsupported_future_schema_version_raises():
    """A bundle written by a future mlframe (schema_version=99) must
    refuse to load on this build rather than silently running wrong-
    semantics code."""
    from mlframe.training.core.predict import _validate_metadata_version_envelope

    with pytest.raises(ValueError, match="unsupported schema_version=99"):
        _validate_metadata_version_envelope(
            {"schema_version": 99},
            "fake/path",
        )


def test_current_schema_version_accepted_silently(caplog):
    """schema_version equal to current must accept with no warning."""
    from mlframe.training.core.predict import (
        _validate_metadata_version_envelope,
        _CURRENT_SCHEMA_VERSION,
    )

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        _validate_metadata_version_envelope(
            {"schema_version": _CURRENT_SCHEMA_VERSION},
            "fake/path",
        )
    # No WARN/ERROR records expected.
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns == [], f"unexpected warnings on current schema: {[r.message for r in warns]}"


def test_old_schema_version_warns_but_proceeds(caplog):
    """schema_version below current must WARN (so the operator sees the
    skew) but NOT raise -- back-compat is the contract."""
    from mlframe.training.core.predict import _validate_metadata_version_envelope

    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        # Pre-fix this passed silently; post-fix emits a WARN.
        _validate_metadata_version_envelope({"schema_version": 1}, "fake/path")
    assert any("schema_version=1" in rec.message for rec in caplog.records if rec.levelno >= logging.WARNING), "expected WARN naming the old schema_version"


def test_composite_env_signature_drift_warns_when_supplied(caplog, monkeypatch):
    """If the bundle recorded a composite_target_env_signature and the live
    env signature differs, WARN both signatures. Pre-fix: silent."""
    from mlframe.training.core.predict import _validate_metadata_version_envelope

    # Monkey-patch the live env_signature to a known-different value.
    import mlframe.training.composite as _composite

    def _fake_live_sig():
        """Helper that fake live sig."""
        return {"mlframe": "0.99-LIVE", "catboost": "9.9.9"}

    monkeypatch.setattr(_composite, "env_signature", _fake_live_sig)

    metadata = {
        "schema_version": 2,
        "composite_target_env_signature": {
            "mlframe": "0.50-STORED",
            "catboost": "1.0.0",
        },
    }
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core.predict"):
        _validate_metadata_version_envelope(metadata, "fake/path")
    assert any("env signature drift" in rec.message for rec in caplog.records), f"expected env-skew WARN; got: {[r.message for r in caplog.records]}"


def test_validator_wired_at_both_predict_entry_points():
    """Source-level guard that the validator is called at BOTH
    metadata-load sites in predict.py (the suite-predict path AND the
    predict_from_models path). Pre-fix only the suite-predict path
    existed; predict_from_models loaded metadata with no checks."""
    import pathlib
    import mlframe as _mlframe

    # After the 2026-05-21 monolith split, the two entry points moved to
    # ``_predict_main.py``; the 2026-05-22 sub-split further moved each
    # into its own ``_predict_main_from_models.py`` /
    # ``_predict_main_suite.py`` sibling. Concat all five files so the
    # source-pattern sensor still matches both call sites + the helper
    # definition that stayed in parent.
    _core = pathlib.Path(_mlframe.__file__).resolve().parent / "training" / "core"
    src = "\n".join(
        (_core / nm).read_text(encoding="utf-8")
        for nm in (
            "predict.py",
            "_predict_main.py",
            "_predict_main_from_models.py",
            "_predict_main_suite.py",
        )
        if (_core / nm).exists()
    )
    # The validator name must appear at LEAST twice in call positions
    # (def + 2 call sites = 3 total occurrences).
    occurrences = src.count("_validate_metadata_version_envelope")
    assert occurrences >= 3, (
        f"Wave 19 P0 #2 regression: _validate_metadata_version_envelope "
        f"appears {occurrences} times; expected >= 3 (one def + 2 call "
        f"sites). The second predict entry point at predict_from_models "
        f"must also call the validator."
    )


def test_non_dict_metadata_does_not_crash():
    """Some legacy bundles used SimpleNamespace. Validator must NOT
    crash on those -- nothing to validate, just return."""
    from types import SimpleNamespace
    from mlframe.training.core.predict import _validate_metadata_version_envelope

    _validate_metadata_version_envelope(SimpleNamespace(), "fake/path")
    _validate_metadata_version_envelope(None, "fake/path")  # type: ignore[arg-type]
