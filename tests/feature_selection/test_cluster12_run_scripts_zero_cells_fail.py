"""Cluster 12 regression: the run scripts exit non-zero when no cell executed (audit 2026-11-20, finding 12-F99).

Source-level pin: each entry point requires a positive cell count for exit 0, so a run that wrote nothing cannot pass a CI step.
Running the mains needs the real beds, so the behaviour is pinned through the entry-point guard, not by executing them.
"""

from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid"


@pytest.mark.parametrize("name", ["run_phase0.py", "run_scm_beds.py", "run_synth_control.py"])
def test_zero_cells_is_a_failing_exit(name):
    """The entry point exits 0 only for a positive cell count (12-F99)."""
    text = (_BENCH / name).read_text(encoding="utf-8")
    assert "SystemExit(0 if main() > 0 else 1)" in text
    assert "SystemExit(0 if main() >= 0 else 1)" not in text
