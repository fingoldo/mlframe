"""Cluster 29 regression: the eviction protect set names only real sidecars (audit 2026-11-20, finding 29-F40).

The protect set used to include a bogus ``<key>.sha256`` name that never exists on disk. The source now protects the payload and its real
``<key>.pkl.sha256`` sidecar only.
"""

from pathlib import Path

from mlframe.utils import disk_cache

_SRC = Path(disk_cache.__file__).resolve().read_text(encoding="utf-8")


def test_bogus_sidecar_name_is_not_protected():
    """The protect set no longer contains with_suffix(".sha256") on the bare key (29-F40)."""
    assert '_p.with_suffix(".sha256")}' not in _SRC


def test_real_sidecar_is_still_protected():
    """Control: the genuine <key>.pkl.sha256 sidecar stays in the protect set (29-F40)."""
    assert '_p.with_suffix(_p.suffix + ".sha256")' in _SRC
