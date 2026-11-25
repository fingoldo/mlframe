"""Cluster 11 regression: the fe_operator_factory benchmarks install no catch-all warning filter (audit 2026-11-20, finding 11-F6).

A bare ignore hides every warning, including the ones that flag wrong results. The benchmark modules silence only FutureWarning and
DeprecationWarning, the same narrowing applied to the fs_hybrid benches (12-F31).
"""

from pathlib import Path

FACTORY = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "_benchmarks" / "fe_operator_factory"


def test_no_factory_module_installs_a_catch_all_ignore():
    """No fe_operator_factory module contains a bare simplefilter or filterwarnings ignore (11-F6)."""
    offenders = []
    for path in FACTORY.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if 'simplefilter("ignore")' in text or 'filterwarnings("ignore")' in text:
            offenders.append(path.name)
    assert offenders == []
