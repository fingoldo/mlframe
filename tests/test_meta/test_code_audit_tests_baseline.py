"""The test-quality scanners, run over ``tests/`` the way they already run over ``src/``.

``test_code_audit_baseline.py`` roots the audit at ``src/mlframe``, so the checks that exist to catch
defects in TESTS -- a test that asserts on elapsed wall time, one that sleeps and then asserts, one that
reads an attribute it is about to delete -- have never had a subject in this repository. Ported from
pyutilz, which added the same file after four separate rounds of CI-only failures traced to that class.

This matters more than it did a week ago: the per-push leg moved to macOS (see ``.github/workflows/ci.yml``),
and the whole point of a wall-clock assertion is that it encodes the machine it was written on. A runner
with different cores, a cold numba cache and a different BLAS is exactly where those tests stop being
about the code.

Baselined, not enforced at zero. The first scan found 1757, which is what a test suite this size looks
like when these checks have never been pointed at it:

    1014  nondiscriminating_test              the body cannot fail for the reason it claims
     262  source_text_assertion               asserts on code text instead of behaviour
     192  tautological_is_not_none_only_test   the only assertion is "it returned something"
      98  vacuous_empty_pattern_match
      85  wall_clock_assertion                 <- fails on a machine, not on a defect
      69  except_skip_masks_call_under_test
      19  vacuous_assertion
      17  hardcoded_absolute_path_in_test      <- fails off the OS it was written on
       1  sleep_then_assert

Every one is a judgement call about a specific test rather than a mechanical rewrite, so the number is
recorded and the ratchet stops the 1758th from arriving while they are worked through. The two marked
lines are the 102 that bear directly on making a non-Linux leg green.
"""

from __future__ import annotations

from pathlib import Path

import pytest

py_ci_shared_code_audit_meta = pytest.importorskip("py_ci_shared.code_audit_meta")
assert_no_new_code_audit_findings = py_ci_shared_code_audit_meta.assert_no_new_code_audit_findings

TESTS_DIR = Path(__file__).resolve().parent.parent
_BASELINE_PATH = Path(__file__).resolve().parent / "_code_audit_tests_baseline.json"

#: Only the checks whose subject is a TEST. The rest of the audit's 99 scanners are aimed at production
#: code and are already run against ``src/`` by the sibling module; pointing all of them at ``tests/``
#: would bury the ones that belong here under findings nobody is going to act on.
_TEST_QUALITY_CHECKS = [
    "deleted_attribute_read_unconditionally",
    "except_skip_masks_call_under_test",
    "hardcoded_absolute_path_in_test",
    "sleep_then_assert",
    "nondiscriminating_test",
    "source_text_assertion",
    "stale_test_spy_arity",
    "tautological_is_not_none_only_test",
    "unenforced_docstring_invariant",
    "vacuous_assertion",
    "vacuous_empty_pattern_match",
    "wall_clock_assertion",
]


def test_no_new_test_quality_findings_in_the_test_suite(request):
    """Fail on any NEW machine-dependent or non-discriminating test beyond the recorded baseline."""
    assert_no_new_code_audit_findings(
        root=TESTS_DIR,
        baseline_path=_BASELINE_PATH,
        checks=_TEST_QUALITY_CHECKS,
        request=request,
    )
