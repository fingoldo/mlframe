"""Cluster 12 regression: the round-3 benches print the real group sizes and drop a discarded model fit (audit 2026-11-20, finding 12-F90).

Source-level pins, because the benches need the real beds: the strong and weak denominators come from len(strong) and len(weak), not the
literals /4 and /8, and noise_floor no longer fits a throwaway LGBMClassifier before building its shadow.
"""

from pathlib import Path

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid"


def test_hard_bed_denominators_are_group_sizes():
    """The strong and weak counts divide by the actual group sizes (12-F90)."""
    text = (_BENCH / "round3_hard_bed_bench.py").read_text(encoding="utf-8")
    assert "/{len(strong)}" in text and "/{len(weak)}" in text
    assert "/4 " not in text and "/8 " not in text


def test_noise_floor_has_no_discarded_fit():
    """noise_floor fits one model (m2 for importances), not an extra discarded one (12-F90)."""
    text = (_BENCH / "round3_rfecv_levers_bench.py").read_text(encoding="utf-8")
    assert "LGBMClassifier(n_estimators=200, verbose=-1).fit(Xa, ya)\n    rng = np.random.default_rng(0)" not in text
