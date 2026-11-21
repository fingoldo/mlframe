"""Audit 2026-11-20 finding 28-F14: a bundle that exists but cannot be read raises, it does not fall back to an unverified open."""

import builtins

import pytest

from mlframe.training.io import _open_verified_bundle


def test_existing_but_unreadable_bundle_raises_instead_of_reopening(tmp_path, monkeypatch):
    """The first open fails with OSError on a file that exists: the helper re-raises rather than returning None."""
    bundle = tmp_path / "model.dump"
    bundle.write_bytes(b"not really a bundle")
    real_open = builtins.open

    def failing_open(file, *args, **kwargs):
        """Fail the open of the bundle path only, as a transient Windows lock would."""
        if str(file) == str(bundle):
            raise OSError("simulated transient lock")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)
    with pytest.raises(OSError, match="simulated transient lock"):
        _open_verified_bundle(str(bundle))
