"""Text-hygiene lints over ``src/mlframe``.

These checks used to live in the test suite as assertions over ``inspect.getsource`` output. That is a
source-text assertion: it pins the wording of the code rather than its behaviour, and the test-quality gate
rightly flags it. The same stale patterns are checked here instead, by reading the production files from
disk and asserting the bad text is absent. Each pattern is a regression guard for a defect that was fixed
once and must not reappear.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FILTERS_DIR = _REPO_ROOT / "src" / "mlframe" / "feature_selection" / "filters"


def _read_src(path: Path) -> str:
    """Return the UTF-8 text of a production source file."""
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "modname",
    [
        "_fe_raw_redundancy_drop",
        "_fe_sufficient_summary",
        "_fe_cmi_redundancy_null",
        "_extra_fe_families_dispersion_resident",
        "_meta_fe_recommender",
        "_conditional_gate_fe",
    ],
)
def test_no_stale_self_referential_log_line_numbers(modname):
    """None of the filter modules' debug-log messages embed a '<file>.py:<N>' string that goes stale as the file grows."""
    src = _read_src(_FILTERS_DIR / f"{modname}.py")
    assert not re.search(r"suppressed in \S+\.py:\d+", src), f"{modname}.py still has a stale line-number log string"


def test_no_mojibake_in_cat_interactions():
    """The cat_interactions comment corrupted by an encoding round-trip must not reappear."""
    src = _read_src(_FILTERS_DIR / "cat_interactions.py")
    assert "РІР‚В°РІвЂљВ¬" not in src


def test_no_dead_resident_y_isinstance_check_in_fe_cmi_perm_null_gpu():
    """The dead `isinstance(y_h, cp.ndarray)` branches in _fe_cmi_perm_null_gpu.py must stay removed."""
    src = _read_src(_FILTERS_DIR / "_fe_cmi_perm_null_gpu.py")
    assert not any(line.strip().startswith("if isinstance(y_h, cp.ndarray)") for line in src.splitlines())
