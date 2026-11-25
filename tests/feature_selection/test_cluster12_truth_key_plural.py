"""Cluster 12 regression: the quadratic truth key is the plural everywhere, and the hard_synth2 docstring states the real noise count (audit 2026-11-20, finding 12-F88).

Source-level pins: the generators and the diagnostic reader use "quadratic_operands" (the key the adversarial module lists), never the
singular, and hard_synth2's docstring says 1970 noise columns (2000 - 15 linear - 15 redundant), matching the code.
"""

from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fs_hybrid"


@pytest.mark.parametrize("name", ["hard_synth.py", "synth.py", "round4_mdlp_collapse_diag.py"])
def test_no_singular_quadratic_key(name):
    """The singular key literal is gone from the generators and the reader (12-F88)."""
    text = (_BENCH / name).read_text(encoding="utf-8")
    assert '"quadratic_operand"' not in text
    assert "quadratic_operands" in text


def test_hard_synth2_docstring_states_1970_noise():
    """The docstring's noise count matches the code: p - 15 linear - 15 redundant = 1970 (12-F88)."""
    text = (_BENCH / "hard_synth2.py").read_text(encoding="utf-8")
    assert "1970 noise" in text
    assert "1980 noise" not in text
