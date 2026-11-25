"""Cluster 13 regression: the stage sampler prints at most 12 leaves per stage (audit 2026-11-20, finding 13-F35).

Source-level pin, since the script is module-level and needs a profile: the leaf list is sliced before printing, not printed in full
and then cut.
"""

from pathlib import Path

from mlframe.feature_selection import _benchmarks as bench

_SCRIPT = Path(bench.__file__).resolve().parent / "profiling" / "all_families_stage_sampler.py"


def test_leaves_are_sliced_before_printing():
    """The stage's leaves are filtered and capped at 12 before the print loop (13-F35)."""
    text = _SCRIPT.read_text(encoding="utf-8")
    assert "[item for item in both.most_common() if item[0][0] == f][:12]" in text
    assert '[print(f"{v:7.1f}  {k[1]}") for k,v in both.most_common() if k[0]==f][:12]' not in text
