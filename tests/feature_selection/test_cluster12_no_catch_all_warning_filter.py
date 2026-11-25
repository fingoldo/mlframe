"""Cluster 12 regression: fs_hybrid bench files do not install a catch-all warning filter (audit 2026-11-20, finding 12-F31).

A bare ``warnings.filterwarnings("ignore")`` hides every warning, including the ones that flag wrong results. The benches now silence only
FutureWarning and DeprecationWarning.
"""

from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid"


def test_no_bench_installs_a_catch_all_ignore():
    """No fs_hybrid bench file contains a bare catch-all ignore filter (12-F31)."""
    offenders = [p.name for p in BENCH.glob("*.py") if 'filterwarnings("ignore")' in p.read_text(encoding="utf-8")]
    assert offenders == []


def test_named_benches_silence_only_future_and_deprecation():
    """The two benches named in the finding silence only FutureWarning and DeprecationWarning (12-F31)."""
    for name in ("round3_hard_bed_bench.py", "round3_hybrid_refine_bench.py"):
        src = (BENCH / name).read_text(encoding="utf-8")
        assert "category=FutureWarning" in src
