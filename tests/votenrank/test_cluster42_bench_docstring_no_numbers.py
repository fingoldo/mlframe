"""Cluster 42 regression: the confidence-gated bench docstring carries no dated timing numbers (audit 2026-11-20, finding 42-F15).

Source-level pin: the docstring no longer quotes the dated millisecond figures that would drift from the code, and it still points at the
run command and the dispatch module.
"""

from pathlib import Path

from mlframe.votenrank._benchmarks import bench_confidence_gated_blend as bench

_SRC = Path(bench.__file__).resolve()


def test_docstring_has_no_dated_timings():
    """The dated millisecond figures are gone from the docstring (42-F15)."""
    doc = bench.__doc__ or ""
    assert "2026-07-10" not in doc
    assert "ms" not in doc.replace("ms.", "")


def test_docstring_keeps_the_run_command_and_dispatch_pointer():
    """The run command and the dispatch-module pointer remain (42-F15)."""
    doc = bench.__doc__ or ""
    assert "python -m mlframe.votenrank._benchmarks.bench_confidence_gated_blend" in doc
    assert "_confidence_gated_blend_ktc_dispatch.py" in doc
