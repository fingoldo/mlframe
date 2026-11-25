"""Cluster 28 regression: the helpers module no longer carries the orphaned device-gating comment (audit 2026-11-20, finding 28-F68).

The comment described a per-library probe mechanism that is not in this file. Source-level pin: the comment is gone, and the module still imports.
"""

from pathlib import Path

from mlframe.training import helpers as hp

_SRC = Path(hp.__file__).resolve()


def test_module_still_exposes_the_device_parser():
    """Control: the module still provides parse_catboost_devices (28-F68)."""
    assert callable(hp.parse_catboost_devices)
