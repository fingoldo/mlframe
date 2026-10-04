"""Meta-test enforcing the cross-module ``@njit`` invariant.

INVARIANT (per ``mlframe/feature_selection/filters/_numba_utils.py``):
    All @njit helpers called from > 1 filters submodule live in
    ``_numba_utils.py``. Single-module njit helpers stay in their owner.

This test fails if any filters submodule ``X.py`` (other than
``_numba_utils.py``) imports an ``@njit``-decorated symbol from a sibling
filters submodule. Such cross-module imports cause numba's dispatcher to
recompile against each importer's module path, producing silent cache
misses and Windows file-lock races during pytest-xdist runs.

The check is greppy on purpose -- a structural ``ast`` walk would be more
precise but pulls in pyutilz / mypy noise. False positives are vanishingly
rare given the current package shape.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests._known_gap import known_gap

FILTERS_DIR = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters"
ALLOWED_HOST = "_numba_utils"
NJIT_NAMES_HINT = re.compile(r"^@njit", re.MULTILINE)
# Sibling-module @njit imports that predate this check actually running (it used to look in a directory that does not exist);
# a violation outside this set fails, and the set draining to empty closes the gap.
_KNOWN_VIOLATIONS = frozenset(
    {
        "_confirm_predictor.py imports @njit symbol(s) ['get_fleuret_criteria_confidence'] from sibling fleuret.py (should live in _numba_utils.py)",
        (
            "_fe_cpu_batch.py imports @njit symbol(s) ['_plugin_mi_classif_edge_njit', 'plugin_mi_classif_batch_edge_njit'] from sibling _fe_edge_mi.py"
            " (should live in _numba_utils.py)"
        ),
        (
            "_mdlp_validated_split.py imports @njit symbol(s) ['_entropy_from_counts_njit', '_mdlp_best_split_njit'] from sibling supervised_binning.py"
            " (should live in _numba_utils.py)"
        ),
        (
            "_numba_polynom_optimizer.py imports @njit symbol(s) ['_plugin_mi_classif_njit', '_plugin_mi_regression_njit'] from sibling _hermite_fe_mi.py"
            " (should live in _numba_utils.py)"
        ),
        "_orthogonal_jmim_fe.py imports @njit symbol(s) ['_joint_mi_3d_njit'] from sibling _jmim_scorer.py (should live in _numba_utils.py)",
        "_synergy_detector.py imports @njit symbol(s) ['_pair_mm_mi_njit'] from sibling _fe_synergy_screen.py (should live in _numba_utils.py)",
        "fleuret.py imports @njit symbol(s) ['distribute_permutations'] from sibling permutation.py (should live in _numba_utils.py)",
        "fleuret.py imports @njit symbol(s) ['evaluate_gain'] from sibling evaluation.py (should live in _numba_utils.py)",
    }
)
RELATIVE_IMPORT_RE = re.compile(
    r"^from\s+\.(\w+)\s+import\s+([^\n#]+)",
    re.MULTILINE,
)


def _enumerate_njit_names(text: str) -> set[str]:
    """Return the set of names defined directly under ``@njit`` in this source text."""
    names = set()
    for m in re.finditer(
        r"^@njit[^\n]*\n(?:[^\n]*\n)*?def\s+(\w+)\s*\(",
        text,
        re.MULTILINE,
    ):
        names.add(m.group(1))
    return names


def _violations(sources: dict[str, str]) -> list[str]:
    """Cross-module ``@njit`` imports among the ``{submodule stem: source text}`` mapping."""
    submodules = sorted(sources)
    njit_names_by_module = {sub: _enumerate_njit_names(sources[sub]) for sub in submodules}

    violations: list[str] = []
    for importer in submodules:
        if importer == ALLOWED_HOST:
            continue
        text = sources[importer]
        for m in RELATIVE_IMPORT_RE.finditer(text):
            source_module = m.group(1)
            if source_module in (ALLOWED_HOST, "_internals", "_legacy"):
                # Allowed to import @njit helpers from `_numba_utils`.
                # ``_legacy`` is the migration scaffold (etap 1-10) and is
                # exempted; the invariant only applies between the new
                # submodules.
                continue
            if source_module not in njit_names_by_module:
                continue
            imported = {n.strip() for n in m.group(2).split(",") if n.strip()}
            crossed = imported & njit_names_by_module[source_module]
            if crossed:
                violations.append(f"{importer}.py imports @njit symbol(s) {sorted(crossed)} from sibling {source_module}.py (should live in _numba_utils.py)")

    return violations


def test_cross_module_njit_detector_catches_a_sibling_import_and_passes_the_shared_host():
    """Importing an ``@njit`` helper from a sibling is reported; importing it from ``_numba_utils`` is not."""
    owner = "@njit(cache=True)\ndef kern(x):\n    return x\n"
    bad = _violations({"a": owner, "b": "from .a import kern\n"})
    assert bad == ["b.py imports @njit symbol(s) ['kern'] from sibling a.py (should live in _numba_utils.py)"]
    assert _violations({"_numba_utils": owner, "b": "from ._numba_utils import kern\n"}) == []


def test_no_cross_module_njit_imports():
    """No filters submodule imports an ``@njit`` helper from a sibling submodule."""
    assert FILTERS_DIR.is_dir(), f"filters package not found at {FILTERS_DIR}"
    sources = {p.stem: p.read_text(encoding="utf-8") for p in FILTERS_DIR.glob("*.py") if p.stem != "__init__"}
    assert sources, "no filters submodules found"
    violations = _violations(sources)
    new = sorted(set(violations) - _KNOWN_VIOLATIONS)
    assert not new, "New cross-module @njit imports detected:\n  " + "\n  ".join(new)
    known_gap(f"{len(violations)} sibling-module @njit imports remain; move the helpers to _numba_utils.py", gap_closed=not violations)
