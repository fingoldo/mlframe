"""Sensor tests for the centralised sha256-sidecar pickle helpers."""

from __future__ import annotations

import os
import pickle

import pytest

from mlframe.utils.safe_pickle import (
    PickleVerificationError,
    safe_dump,
    safe_load,
    verify_sidecar,
    write_sidecar,
)


def _make_pickle(path: str, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def test_verify_sidecar_missing_default_fail_closed(tmp_path, monkeypatch):
    """No sidecar + no env opt-in -> verify returns False (default-strict)."""
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    assert verify_sidecar(str(p)) is False


def test_verify_sidecar_missing_env_opt_in_passes(tmp_path, monkeypatch):
    """No sidecar + env opt-in -> verify returns True (legacy escape)."""
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    assert verify_sidecar(str(p)) is True


def test_verify_sidecar_explicit_allow_unverified_overrides_env(tmp_path, monkeypatch):
    """Per-call ``allow_unverified=True`` ignores the env var (set or unset)."""
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    assert verify_sidecar(str(p), allow_unverified=True) is True
    assert verify_sidecar(str(p), allow_unverified=False) is False


def test_verify_sidecar_matching_digest_returns_true(tmp_path):
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    write_sidecar(str(p))
    assert verify_sidecar(str(p)) is True


def test_verify_sidecar_mismatched_digest_returns_false(tmp_path):
    """Tampered file (digest no longer matches sidecar) returns False even
    with the env opt-in -- corrupt sidecar is a security signal, not legacy
    behaviour."""
    p = tmp_path / "bundle.pkl"
    _make_pickle(str(p), {"a": 1})
    write_sidecar(str(p))
    # Tamper with the file post-sidecar; sidecar now describes the old bytes.
    with open(p, "ab") as f:
        f.write(b"extra-bytes-after-pickle")
    assert verify_sidecar(str(p)) is False


def test_safe_load_round_trip_with_sidecar(tmp_path):
    p = tmp_path / "round.pkl"
    safe_dump({"k": [1, 2, 3]}, str(p))
    obj = safe_load(str(p))
    assert obj == {"k": [1, 2, 3]}


def test_safe_load_missing_sidecar_raises(tmp_path, monkeypatch):
    """Missing sidecar -> PickleVerificationError under default-strict."""
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    p = tmp_path / "no_sidecar.pkl"
    _make_pickle(str(p), {"x": 9})
    with pytest.raises(PickleVerificationError):
        safe_load(str(p))


def test_safe_load_missing_sidecar_env_opt_in_loads(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    p = tmp_path / "legacy.pkl"
    _make_pickle(str(p), {"x": 9})
    assert safe_load(str(p)) == {"x": 9}


def test_safe_load_corrupt_sidecar_raises(tmp_path, monkeypatch):
    """Digest mismatch raises even with env opt-in set."""
    monkeypatch.setenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", "1")
    p = tmp_path / "corrupt.pkl"
    safe_dump({"k": "v"}, str(p))
    # Mutate file post-sidecar so digests diverge.
    with open(p, "ab") as f:
        f.write(b"\x00")
    with pytest.raises(PickleVerificationError):
        safe_load(str(p))


def test_write_sidecar_is_idempotent(tmp_path):
    p = tmp_path / "idem.pkl"
    _make_pickle(str(p), [1, 2, 3])
    write_sidecar(str(p))
    digest_first = (tmp_path / "idem.pkl.sha256").read_text(encoding="utf-8")
    write_sidecar(str(p))
    digest_second = (tmp_path / "idem.pkl.sha256").read_text(encoding="utf-8")
    assert digest_first == digest_second


def test_legacy_predict_shim_delegates(tmp_path, monkeypatch):
    """Back-compat shim ``inference.predict._verify_sidecar`` routes through the new module."""
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    from mlframe.inference.predict import _verify_sidecar

    p = tmp_path / "shim.pkl"
    _make_pickle(str(p), {"a": 1})
    assert _verify_sidecar(str(p)) is False
    write_sidecar(str(p))
    assert _verify_sidecar(str(p)) is True


def test_legacy_pipelines_shim_delegates(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFRAME_ALLOW_UNVERIFIED_PICKLE", raising=False)
    from mlframe.estimators.pipelines import _verify_sidecar

    p = tmp_path / "shim2.pkl"
    _make_pickle(str(p), {"b": 2})
    assert _verify_sidecar(str(p)) is False
    write_sidecar(str(p))
    assert _verify_sidecar(str(p)) is True
