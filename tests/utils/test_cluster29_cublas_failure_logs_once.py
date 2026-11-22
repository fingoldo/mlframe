"""Audit 2026-11-20 finding 29-F16: once the cuBLAS probe has failed, repeated checks do not repeat the library lookup.

The version lookup imports cupy and walks package metadata. The test fails the probe, calls the check several times, and counts lookups.
"""

from mlframe.utils import cublas_probe as cp_module


def test_failed_probe_looks_up_libraries_once(monkeypatch):
    """Three checks on a failed probe return False and run the library lookup a single time."""
    lookups = []

    def counting_versions():
        """Record each library lookup."""
        lookups.append(1)
        return {}

    monkeypatch.setattr(cp_module, "_RESULT", (False, "probe failed"))
    monkeypatch.setattr(cp_module, "_FAILURE_LOGGED", False)
    monkeypatch.setattr(cp_module, "_library_versions", counting_versions)
    results = [cp_module.cublas_matmul_ok() for _ in range(3)]
    assert results == [False, False, False]
    assert len(lookups) == 1
