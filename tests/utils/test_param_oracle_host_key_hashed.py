"""The Param-Oracle host key does not carry the machine's name when the hardware fingerprint is unavailable (LOG5-1)."""

import builtins
import platform

from mlframe.utils import _param_oracle as po


def _break_fingerprint_import(monkeypatch):
    """Make ``from pyutilz...cache import hw_fingerprint`` raise so the fallback runs."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        """Raise for the kernel-tuning cache module, delegate for everything else."""
        if name.startswith("pyutilz.performance.kernel_tuning.cache"):
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_fallback_host_key_is_a_digest_not_the_host_name(monkeypatch):
    """With the fingerprint unavailable the key is ``node_`` plus 16 hex digits and the node name appears nowhere in it."""
    _break_fingerprint_import(monkeypatch)
    monkeypatch.setattr(platform, "node", lambda: "secret-build-host-17")
    key = po._host_key()
    assert key.startswith("node_") and len(key) == len("node_") + 16
    assert "secret-build-host-17" not in key
    assert key == po._host_key()


def test_different_hosts_get_different_keys(monkeypatch):
    """Two node names give two keys, so hashing does not merge hosts."""
    _break_fingerprint_import(monkeypatch)
    monkeypatch.setattr(platform, "node", lambda: "host-a")
    first = po._host_key()
    monkeypatch.setattr(platform, "node", lambda: "host-b")
    assert po._host_key() != first
