"""Meta-tests enforcing repo conventions.

- No stdlib ``json`` imports in test files (MEMORY.md: always orjson).
- ``mlframe.calibration.post`` exposes a precompiled ``_INCLUDE_RE`` sentinel.
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


def test_postcalibration_include_re_is_compiled() -> None:
    """Postcalibration include re is compiled."""
    pytest.importorskip("sklearn")
    postcalibration = pytest.importorskip("mlframe.calibration.post")
    include_re = getattr(postcalibration, "_INCLUDE_RE", None)
    assert isinstance(include_re, re.Pattern), "mlframe.calibration.post._INCLUDE_RE must be a module-level compiled re.Pattern"
    # Also validate the lru_cache-wrapped compiler is present and returns a Pattern.
    compile_pattern = getattr(postcalibration, "_compile_pattern", None)
    assert callable(compile_pattern), "_compile_pattern helper must exist"
    assert isinstance(compile_pattern("foo"), re.Pattern)
