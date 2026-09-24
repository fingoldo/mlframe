"""Blocking complexity ratchet: no new function over McCabe complexity 25, and the complex ones may not get worse.

Complexity is ruff's C901 number, keyed by ``path::Qual.name`` (py_ci_shared.function_complexity). The threshold is
this codebase's measured distribution (median 2, p90 9, p95 13, p99 30) read against the functions a reader actually
struggles with, not a copied default. The baseline holds the functions over it today, each a tracked refactor in
audits/full_audit_2026-09-20/complexity_refactor.md; a refactor that lowers one must lower or drop its ceiling:
``python tests/test_meta/regen_baselines.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("py_ci_shared.function_complexity")
pytest.importorskip("ruff")

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).resolve().parent / "_function_complexity_baseline.json"
LIMIT = 25


def _src_files() -> list[Path]:
    """Production modules, minus frozen bench copies (the same set the function-length ratchet measures)."""
    return sorted(p for p in (REPO_ROOT / "src").rglob("*.py") if "_benchmarks" not in p.parts and "_cpx36_baseline" not in p.parts)


def test_functions_do_not_get_more_complex(request):
    from py_ci_shared.function_complexity import assert_complexity_does_not_grow

    assert_complexity_does_not_grow(_src_files(), REPO_ROOT, BASELINE, limit=LIMIT, min_functions=5000, request=request)


def regenerate_baseline(path: Path = BASELINE) -> None:
    """Rewrite the complexity ceilings from the current tree. Called by ``regen_baselines.py``."""
    from py_ci_shared.function_complexity import function_complexities, write_complexity_baseline

    write_complexity_baseline(path, function_complexities(_src_files(), REPO_ROOT, limit=LIMIT)[0])
