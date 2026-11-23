"""Cluster 13 regression: the kernel tuning cache docs name the real cache location and the real owner of hw_fingerprint (audit 2026-11-20, finding 13-F30).

Source-level pin: the package docstring no longer points at ~/.mlframe/kernel_tuning, and it says hw_fingerprint is pyutilz's. The
pyutilz helper the docs refer to is checked to exist, so the statement stays true.
"""

import pyutilz.performance.kernel_tuning as kt

from mlframe.feature_selection._benchmarks import kernel_tuning_cache as pkg


def test_docs_name_the_pyutilz_cache_location():
    """The docstring documents ~/.pyutilz/kernel_tuning, not the non-existent ~/.mlframe path (13-F30)."""
    doc = pkg.__doc__ or ""
    assert "~/.pyutilz/kernel_tuning/" in doc
    assert "~/.mlframe/kernel_tuning/" not in doc


def test_hw_fingerprint_is_pyutilz_and_documented_as_such():
    """hw_fingerprint is provided by pyutilz, so the docs that say so are accurate (13-F30)."""
    assert callable(kt.hw_fingerprint)
    assert "pyutilz.performance.kernel_tuning" in (pkg.__doc__ or "")
