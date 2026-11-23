"""Cluster 12 regression: the MRMR cap cost bench reports a median over repeats, not one timing sample (audit 2026-11-20, finding 12-F95).

Source-level pin, since the bench needs the full MRMR fit: a REPEATS constant (overridable by CAP_REPEATS) exists, each cell is fitted
REPEATS times, and the stored time is the median.
"""

from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid" / "round4_mrmr_cap_largen_cost_bench.py"


def test_timing_cells_are_medians_over_repeats():
    """Each (n, cap) cell repeats its fit and stores the median (12-F95)."""
    text = _BENCH.read_text(encoding="utf-8")
    assert 'REPEATS = int(os.environ.get("CAP_REPEATS", "3"))' in text
    assert "for _ in range(REPEATS):" in text
    assert "ft = float(np.median(samples))" in text
