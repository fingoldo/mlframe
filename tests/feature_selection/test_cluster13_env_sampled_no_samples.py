"""Cluster 13 regression: the env-sampled fit script reports an empty sample set instead of crashing on the statistics (audit 2026-11-20, finding 13-F34).

Source-level pin, since the script runs a real fit at module level: the no-samples exit comes before the first statistic is computed.
"""

from pathlib import Path

from mlframe.feature_selection import _benchmarks as bench

_SCRIPT = Path(bench.__file__).resolve().parent / "profiling" / "env_sampled_fit.py"


def test_no_samples_exit_precedes_statistics():
    """The empty-sample guard is placed before any statistic over samples (13-F34)."""
    text = _SCRIPT.read_text(encoding="utf-8")
    guard = text.index("if not samples:")
    first_stat = text.index("own = [s[1] for s in samples]")
    assert guard < first_stat
    assert "no samples: the fit was shorter than the sampling interval" in text
