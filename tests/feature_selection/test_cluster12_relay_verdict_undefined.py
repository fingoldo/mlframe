"""Cluster 12 regression: the residual-relay bench reports an undefined verdict when no relay variant ran (audit 2026-11-20, finding 12-F92).

Source-level pin, since the bench needs real beds: a NaN best (no dropped features) returns before the comparison, so the printed verdict
is never a misleading "no gain / KILL" from a NaN.
"""

from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid" / "round4_residual_relay_bench.py"


def test_nan_best_gives_undefined_verdict_before_comparison():
    """The NaN guard precedes the verdict comparison and prints 'undefined' (12-F92)."""
    text = _BENCH.read_text(encoding="utf-8")
    guard = text.index("if best != best:")
    comparison = text.index("-> residual-relay HELPS")
    assert guard < comparison
    assert "VERDICT: undefined, no relay variant ran" in text
