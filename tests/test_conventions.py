"""Meta-tests enforcing repo conventions.

- No stdlib ``json`` imports in test files (MEMORY.md: always orjson).
- ``mlframe.calibration.post._compile_pattern`` actually caches compiled patterns.
- No ``ensure_installed(...)`` calls in test files (use ``pytest.importorskip``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

_JSON_IMPORT_RE = re.compile(r"^\s*(?:import\s+json\b|from\s+json\s+import\b)", re.MULTILINE)
_ENSURE_INSTALLED_RE = re.compile(r"\bensure_installed\s*\(")

# Files that legitimately use stdlib ``json`` (the orjson rule targets hot PRODUCTION paths, not these
# test-only uses). The gate still catches any NEW non-whitelisted test file. Express as POSIX-rel paths.
_STDLIB_JSON_WHITELIST = {
    # Subprocess workers / embedded ``python -c`` scripts: json is the IPC serializer in a FRESH child
    # interpreter (orjson need not be importable there); json.dumps/loads of the payload is correct + simplest.
    "feature_selection/mrmr/fe/test_biz_value_mrmr_fe_canonical.py",
    "feature_selection/mrmr/fe/test_mrmr_fe_fixes_adversarial.py",
    "feature_selection/mrmr/core/test_f2_param_robustness.py",
    "feature_selection/mrmr/core/test_mrmr_endtoend_invariants.py",
    "feature_selection/test_suite_fe_linear_recovery.py",
    "feature_selection/_suite_fe_worker.py",
    # Per-test artifact-ledger dumps with an orjson-first path and a stdlib-json fallback for robustness.
    "feature_selection/mrmr/core/test_mrmr_create_keep_drop.py",
    "feature_selection/mrmr/core/test_mrmr_distribution_profiles.py",
    "feature_selection/mrmr/core/test_mrmr_weak_f2_seed_stability.py",
    # Round-trips a serving spec through stdlib json specifically to prove stdlib-json-compatibility of the export.
    "training/composite/cache/test_composite_serving_export.py",
    # Patches production's stdlib json.dumps (the cache-signature hash path); must target the same module.
    "training/composite/cache/test_composite_cache_edge.py",
    # Asserts strict-JSON cleanliness of a report by checking the serialized text has NO bare ``NaN``/``Infinity``
    # tokens -- this is exactly stdlib json's lenient behaviour under test (orjson raises on non-finite, so it
    # cannot express the negative assertion the test makes).
    "training/composite/test_biz_val_regime_headroom.py",
    "training/composite/test_biz_val_value_report.py",
    # Writes ``features.dump.json`` fixtures parsed by production ``_load_features_file`` (stdlib-json loader);
    # the fixture writer mirrors that loader's format.
    "inference/test_predict_load_features_and_branches.py",
}


def _iter_test_files() -> list[Path]:
    """Returns ``[p for p in TESTS_DIR.rglob('*.py') if p.name != 'test_conventions.py']``."""
    return [p for p in TESTS_DIR.rglob("*.py") if p.name != "test_conventions.py"]


def test_no_stdlib_json_in_tests() -> None:
    """No stdlib json in tests."""
    offenders: list[str] = []
    for path in _iter_test_files():
        if path.relative_to(TESTS_DIR).as_posix() in _STDLIB_JSON_WHITELIST:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if _JSON_IMPORT_RE.search(text):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert not offenders, "Test files must use orjson instead of stdlib json (or add a justified entry to _STDLIB_JSON_WHITELIST); offenders: " + ", ".join(
        offenders
    )


def test_no_ensure_installed_in_tests() -> None:
    """No ensure installed in tests."""
    offenders: list[str] = []
    for path in _iter_test_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if _ENSURE_INSTALLED_RE.search(text):
            offenders.append(str(path.relative_to(TESTS_DIR)))
    assert not offenders, "Test files must use pytest.importorskip(...) instead of ensure_installed(...); offenders: " + ", ".join(offenders)


def test_postcalibration_compile_pattern_caches_and_compiles() -> None:
    """mlframe.calibration.post._compile_pattern compiles a pattern and reuses the cached object on a repeat call."""
    pytest.importorskip("sklearn")
    postcalibration = pytest.importorskip("mlframe.calibration.post")
    compile_pattern = getattr(postcalibration, "_compile_pattern", None)
    assert callable(compile_pattern), "_compile_pattern helper must exist"
    compiled = compile_pattern("foo.*bar")
    assert isinstance(compiled, re.Pattern)
    assert compiled.match("foo123bar") is not None
    # lru_cache: an identical pattern string must return the SAME compiled object, not just an equal one.
    assert compile_pattern("foo.*bar") is compiled


_PROCESS_MARKER_RE = re.compile(r"\b(FIX\d|BUG\d|ROOT CAUSE \d|OPT-[A-Z]\b|ND-\d|finding[- ]#?\d+|Wave \d|Layer-?\d|iter\d{2,}|\d{4}-\d{2}-\d{2})")
# Modules whose comments have been cleaned. Kept as a list rather than applied repo-wide, per the no-unscoped-rewrite rule: widen it as other
# packages are cleaned, do not swap it for a whole-tree glob in one go.
_MARKER_FREE_FILES = (
    "feature_selection/filters/_mrmr_fe_step",
    "feature_selection/filters/_mrmr_validate_transform.py",
    "feature_selection/filters/_mrmr_fingerprints.py",
)


def test_no_process_markers_in_cleaned_package_comments() -> None:
    """Phase markers, finding IDs and bug numbers belong in git history, not in the code that outlived them.

    ``OPT-A``, ``BUG2 FIX``, ``ROOT CAUSE 5`` and ``finding-#21`` say nothing to a reader who does not have the review thread, and two of them
    referred to findings that no longer resolve to anything in-tree. The WHY prose beside them is what is worth keeping.
    """
    src = TESTS_DIR.parent / "src" / "mlframe"
    offenders: list[str] = []
    for entry in _MARKER_FREE_FILES:
        target = src / entry
        for path in [target] if target.is_file() else sorted(target.rglob("*.py")):
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                stripped = line.strip()
                if not stripped.startswith("#"):
                    continue
                found = _PROCESS_MARKER_RE.search(stripped)
                if found:
                    offenders.append(f"{path.relative_to(src)}:{lineno}: {found.group(0)}")
    assert not offenders, "process/audit markers in comments (keep the reasoning, drop the label): " + ", ".join(offenders)


# A line number is a reference that rots: six comments in the fit-impl package cited lines from the pre-split 10056-line monolith, and one of
# them built a load-bearing argument ("this read happens AFTER it, so the freshly repopulated attribute is authoritative") on a number the
# reader could not check. Naming the function or section instead says the same thing and stays true.
_LINE_CITATION_RE = re.compile(r"line\s*~?\s*\d{3,}")
_CITATION_FREE_PACKAGES = ("feature_selection/filters/_mrmr_fit_impl",)


def test_no_stale_line_number_citations_in_comments() -> None:
    """Comments must refer to code by name, not by a line number that no longer points anywhere."""
    src = TESTS_DIR.parent / "src" / "mlframe"
    offenders: list[str] = []
    for package in _CITATION_FREE_PACKAGES:
        for path in sorted((src / package).rglob("*.py")):
            for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#") and _LINE_CITATION_RE.search(stripped):
                    offenders.append(f"{path.relative_to(src)}:{lineno}")
    assert not offenders, "comment(s) citing a line number instead of a name: " + ", ".join(offenders)
