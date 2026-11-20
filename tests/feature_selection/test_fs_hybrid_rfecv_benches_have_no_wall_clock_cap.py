"""Audit 2026-11-20 finding 12-F35: the RFECV comparison benches budget by refits only, not by wall-clock time.

A wall-clock cap made results depend on machine speed and load; the refit count is the only budget now.
"""

import ast
from pathlib import Path

import pytest

_DIR = Path(__file__).resolve().parents[2] / "src/mlframe/feature_selection/_benchmarks/fs_hybrid"
_FILES = ["round2_rfecv_nrepeats.py", "round2_rfecv_cv_sensitivity.py", "round3_rfecv_bench.py", "round2_rfecv_crossseed_bench.py", "round3_rfecv_levers_bench.py"]


@pytest.mark.parametrize("name", _FILES)
def test_search_config_has_no_wall_clock_cap(name):
    """Every SearchConfig call in the bench passes max_runtime_mins=None (no cap) and keeps its max_refits."""
    tree = ast.parse((_DIR / name).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "SearchConfig"]
    assert calls, f"{name} has no SearchConfig call"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "max_runtime_mins" in kw and isinstance(kw["max_runtime_mins"], ast.Constant) and kw["max_runtime_mins"].value is None
        assert "max_refits" in kw
