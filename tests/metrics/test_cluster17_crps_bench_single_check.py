"""Cluster 17 regression: the CRPS bench checks the shipped function against the loop with one computation and one comparison (audit 2026-11-20, finding 17-F29).

Source-level pin: the shipped value is computed once into a variable and compared once; the duplicate call and the redundant isclose(atol=0)
are gone.
"""

from pathlib import Path

from mlframe.metrics._benchmarks import bench_crps_fused_per_alpha as bench

_SRC = Path(bench.__file__).resolve()


def test_shipped_check_computes_and_compares_once():
    """The verify step calls crps_from_quantiles once and compares with a single inequality (17-F29)."""
    text = _SRC.read_text(encoding="utf-8")
    assert text.count("shipped = crps_from_quantiles(y, p, a.tolist())") == 1
    assert "if shipped != r_before:" in text
    assert "np.isclose(crps_from_quantiles" not in text
