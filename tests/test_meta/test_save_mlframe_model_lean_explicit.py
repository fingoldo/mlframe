"""Meta-test (E2.3, 2026-05-22): every ``save_mlframe_model(...)`` call site
in the production source tree must pass ``lean=`` explicitly OR carry the
``# forensic_save`` marker on the call line.

Rationale: ``lean`` defaults to False to preserve forensic round-trip
parity, but the TVT-2026-05-21 prod log showed that a default-lean=False
save leaks ~120 MB of per-split arrays + trainset_features_stats on a
4M-row model. The fix at train_eval.py:905 flipped the train-time save to
lean=True, but a future contributor adding a new save site could
re-introduce the bloat by accident. This meta-test makes the lean choice
EXPLICIT at every call site: either pick a side or annotate ``# forensic_save``
to opt in to the fat default. New save site without either -> CI fails.

The test parses each .py under src/ via ast, walks all Call nodes, and
flags any ``save_mlframe_model(...)`` call whose kwargs lack ``lean`` AND
whose source line lacks ``# forensic_save``.
"""
from __future__ import annotations

import ast
import pathlib

from tests.test_meta._shared_ast_cache import parsed_ast, source_text

SRC_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe"


def _is_test_adjacent(path: pathlib.Path) -> bool:
    """Benchmarks / profilers are not prod save sites: they legitimately exercise the default forensic save to measure its cost.

    Mirrors the exemption in ``test_no_underscore_imports_cross_package._is_test_adjacent`` so the two source-hygiene
    sensors agree on what counts as production code.
    """
    if "_benchmarks" in path.parts:
        return True
    return path.name.startswith("_profile_") or path.name.startswith("_bench_")


def _iter_python_files(root: pathlib.Path):
    """Yield every non-test-adjacent ``.py`` file under ``root``."""
    for path in root.rglob("*.py"):
        if not _is_test_adjacent(path):
            yield path


def _find_save_calls_without_explicit_lean() -> list[tuple[str, int, str]]:
    """Walk every .py under src/mlframe; return (path, lineno, line) for every
    ``save_mlframe_model(...)`` call missing ``lean=`` AND missing the
    ``# forensic_save`` annotation."""
    violations: list[tuple[str, int, str]] = []
    for path in _iter_python_files(SRC_ROOT):
        tree = parsed_ast(path)
        if tree is None:
            continue
        src = source_text(path)
        lines = src.splitlines() if src is not None else []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Match `save_mlframe_model(...)` (bare attr or dotted attr).
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name != "save_mlframe_model":
                continue
            kw_names = {kw.arg for kw in (node.keywords or [])}
            if "lean" in kw_names:
                continue
            # Look at the source line for the forensic-save annotation.
            lineno = getattr(node, "lineno", 1)
            line_text = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
            if "# forensic_save" in line_text or "# forensic save" in line_text:
                continue
            # io.py:386 itself defines the function; the recursive auto-retry
            # call at io.py:637 passes lean=True via positional? No -- it
            # passes lean=True via kwarg, so kw_names includes lean. Anything
            # that lands here is a real missing-lean violation.
            violations.append((str(path.relative_to(SRC_ROOT.parent.parent)), int(lineno), line_text.strip()))
    return violations


def test_every_save_mlframe_model_call_has_explicit_lean_or_forensic_marker():
    """E2.3: every ``save_mlframe_model(...)`` call must pass ``lean=`` explicitly or carry ``# forensic_save``."""
    violations = _find_save_calls_without_explicit_lean()
    assert not violations, (
        "E2.3 (2026-05-22): ``save_mlframe_model(...)`` call must pass "
        "``lean=...`` explicitly OR carry a ``# forensic_save`` comment on the "
        "call line. The lean default is False (forensic round-trip parity) "
        "but real prod save sites should opt INTO lean=True to avoid leaking "
        "16-32 MB per-split arrays at 4M-row scale (TVT-2026-05-21 prod "
        "incident). Found violations:\n" + "\n".join(f"  {p}:{ln}  {ln_text}" for p, ln, ln_text in violations)
    )
