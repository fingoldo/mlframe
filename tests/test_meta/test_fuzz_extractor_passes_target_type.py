"""Every fuzz suite tells its features-and-targets extractor the exact target type, never only a regression flag.

``SimpleFeaturesAndTargetsExtractor(regression=...)`` is a boolean. A combo whose ``target_type`` is ``quantile_regression`` or
``multi_target_regression`` was handed ``regression=False`` by every suite that wrote ``regression=(combo.target_type == "regression")``,
the extractor resolved the continuous column to binary classification, and LightGBM's classifier encoder failed with "y contains
previously unseen labels" (c0002 in deep-nightly). Three suites carried the bug for months because each built the extractor on its own;
this scan fails the next one that does.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FUZZ_DIR = REPO_ROOT / "tests" / "training" / "fuzz"
EXTRACTOR = "SimpleFeaturesAndTargetsExtractor"
# The scan must see the suites that build one; a moved directory would otherwise pass with nothing to check.
MIN_CONSTRUCTIONS = 5
# Files that build the flag-only extractor on purpose, each with the reason.
ALLOWED_FLAG_ONLY = {"test_fuzz_target_type_mapping.py": "constructs the flag-only extractor to pin that it resolves a quantile-regression combo to binary classification"}


def constructions_without_target_type(source: str) -> list[int]:
    """Line numbers of ``SimpleFeaturesAndTargetsExtractor(...)`` calls that pass neither ``target_type=`` nor ``**kwargs``."""
    bad = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name != EXTRACTOR:
            continue
        if not any(kw.arg == "target_type" or kw.arg is None for kw in node.keywords):
            bad.append(node.lineno)
    return sorted(bad)


def _count_constructions(source: str) -> int:
    """Number of extractor constructions in ``source``."""
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and (getattr(node.func, "id", None) == EXTRACTOR or getattr(node.func, "attr", None) == EXTRACTOR)
    )


def test_a_boolean_flag_alone_is_found() -> None:
    """The scan reports a call that passes only ``regression=`` and accepts the explicit and the ``**kwargs`` forms."""
    flag_only = f'fte = {EXTRACTOR}(target_column="t", regression=(combo.target_type == "regression"))\n'
    explicit = f'fte = {EXTRACTOR}(target_column="t", regression=True, target_type=tt)\n'
    spread = f"fte = {EXTRACTOR}(target_column='t', **kw)\n"
    assert constructions_without_target_type(flag_only) == [1]
    assert constructions_without_target_type(explicit) == []
    assert constructions_without_target_type(spread) == []


def test_every_fuzz_suite_passes_the_exact_target_type() -> None:
    """No fuzz file builds the extractor from the regression flag alone."""
    total = 0
    offenders: dict[str, list[int]] = {}
    for path in sorted(FUZZ_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        total += _count_constructions(source)
        lines = constructions_without_target_type(source)
        if lines and path.name not in ALLOWED_FLAG_ONLY:
            offenders[path.name] = lines
    assert total >= MIN_CONSTRUCTIONS, f"only {total} extractor constructions found under {FUZZ_DIR}; the scan no longer sees the fuzz suites"
    assert not offenders, (
        f"extractor built without target_type (a quantile- or multi-target regression combo then resolves to binary classification): {offenders}. "
        "Pass target_type=target_type_for_combo(combo, target_col) from tests/training/fuzz/_target_type.py."
    )
