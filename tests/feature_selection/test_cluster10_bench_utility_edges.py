"""Cluster 10 regression: edge handling in benchmark utilities (audit 2026-11-20, findings 10-F8 and 10-F9).

10-F8: redundant columns without any informative column raise a clear ValueError.
10-F9: a failed log-file write is reported once on stderr; the console output continues.
"""

import io

import pytest

from mlframe.feature_selection._benchmarks import _bench_shared
from mlframe.feature_selection._benchmarks._shap_proxy_regime_data import make_regime_dataset


def test_redundant_without_informative_raises_clear_error():
    """n_informative=0 with n_redundant>0 is rejected with the documented message (10-F8)."""
    with pytest.raises(ValueError, match="redundant columns need at least one informative"):
        make_regime_dataset(n_samples=50, n_informative=0, n_redundant=2, n_noise=1, seed=0)


class _BrokenStream(io.StringIO):
    """A file object whose write always fails, like a closed or full log handle."""

    def write(self, s):
        """Raise OSError on every write."""
        raise OSError("disk full")


def test_failed_log_write_is_reported_once_on_stderr(capsys):
    """Console output continues, and the broken log is announced exactly once (10-F9)."""
    captured: list[str] = []

    def console(*a, **kw):
        """Record console writes; the broken log stream is passed as the file keyword."""
        if kw.get("file") is None:
            captured.append(" ".join(str(x) for x in a))
            return
        kw["file"].write(" ".join(str(x) for x in a))

    tee = _bench_shared.make_tee_print(console, _BrokenStream())
    tee("first")
    tee("second")
    tee("third")
    assert captured == ["first", "second", "third"]
    err = capsys.readouterr().err
    assert err.count("log file write failed") == 1
    assert "further output goes to the console only" in err
