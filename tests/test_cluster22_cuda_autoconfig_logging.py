"""Cluster 22 regression: the CUDA auto-config never crashes its own error handler (audit 2026-11-20, finding 22-F9).

``import logging`` inside the function body made ``logging`` a local name for the whole function. When the filesystem probe raised before that import ran,
the ``except`` handler read an unbound local and the import of the package died with UnboundLocalError instead of swallowing the probe failure.
"""

import os
import sysconfig

import mlframe


def test_probe_failure_is_swallowed_not_an_unbound_local(monkeypatch):
    """A failing probe with no CUDA variables set returns quietly (22-F9)."""
    for key in [k for k in os.environ if k in ("CUDA_HOME", "CUDA_PATH") or k.startswith("CUDA_PATH_V")]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("MLFRAME_NO_CUDA_AUTOCONFIG", raising=False)

    def boom(*args, **kwargs):
        """Fail like a broken sysconfig, before any inner import in the function has run."""
        raise RuntimeError("sysconfig unavailable")

    monkeypatch.setattr(sysconfig, "get_paths", boom)
    mlframe._autoconfigure_cuda_home()
    # Quiet means no exception and no CUDA variable set on the way out.
    leaked = [k for k in ("CUDA_HOME", "CUDA_PATH") if k in os.environ]
    assert leaked == []
