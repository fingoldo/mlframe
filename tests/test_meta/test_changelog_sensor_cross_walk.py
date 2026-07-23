"""B1 CHANGELOG cross-walk: pin that audit-cycle ``fix`` entries reference a sensor file.

Loose verification across the Round 2026-05-24 audit cycle entries -- every entry that uses ``fix(`` or
introduces a bug-fix MUST mention either a ``test_regression_*.py`` filename, a ``tests/...`` path, or an
explicit sensor identifier. This catches "fixed it but forgot the regression test" entries that the
``feedback_test_every_bug_fix`` memory rule was created to prevent.

Scope: only the audit-cycle section near the top of CHANGELOG.md (delimited by the dated heading). Older
historical entries pre-date the convention and are not enforced.

This is a soft drift signal -- the audit-cycle section can carry a small number of doc-only entries that
genuinely have no associated test (CHANGELOG cosmetic, version bumps, README updates). The threshold
``_MAX_UNCITED_FIX_FRACTION`` allows up to 15% of fix-tagged entries to be doc-only before the test fails.

Implementation note: this test used to hand-roll its own bullet/threshold logic; it now delegates to the
shared engine in ``py_ci_shared.changelog_promise_parity`` (the same engine production_scrapers uses for
its own "CHANGELOG promise resolved elsewhere" check), so the two independently-discovered "a CHANGELOG
claim needs a cross-walk" checks in this ecosystem share one implementation instead of drifting copies.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from py_ci_shared.changelog_promise_parity import (
    assert_changelog_bullets_satisfy_pattern,
    extract_section,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHANGELOG = _REPO_ROOT / "CHANGELOG.md"

# Section header marking the audit-cycle entries this test enforces. Anything before this header is
# considered "newer entries", everything between this and the prior dated heading is the audit-cycle scope.
_AUDIT_SECTION_PATTERN = re.compile(
    r"^##\s+\d{4}-\d{2}-\d{2}.*?(audit cycle|backlog drain|wave[ -]?\d+)",
    re.IGNORECASE | re.MULTILINE,
)

# Patterns that count as a "sensor mention" on a bullet line.
_SENSOR_REFERENCE_PATTERN = re.compile(
    r"(test_regression_[a-zA-Z0-9_]+\.py"
    r"|tests/[a-zA-Z0-9_/]+\.py"
    r"|test_[a-zA-Z0-9_]+\.py"
    r"|tests/(?:test_meta|training|feature_engineering|feature_selection|calibration|inference|models|signal|votenrank|metrics)/[\w/]+"
    r"|sensor[: ]"
    r"|@pytest\.mark\."
    r"|meta-test\b"
    r"|behavioural[- ]equivalence sensor"
    r"|biz_value test"
    r"|verified-already-fixed)",
    re.IGNORECASE,
)

# Patterns that mark a bullet as a "fix" entry (vs doc-only / feature / refactor).
_FIX_BULLET_PATTERN = re.compile(
    r"(fix\([^\)]*\)|\bbug\b|\bregression\b|\bhard-error\b|raise instead of warn|removal|skip removal)",
    re.IGNORECASE,
)

_MAX_UNCITED_FIX_FRACTION = 0.15


def _load_audit_cycle_section() -> str:
    """Extract the top audit-cycle block from CHANGELOG.md -- everything from the first matching dated
    heading down to (but not including) the second matching dated heading.
    """
    content = _CHANGELOG.read_text(encoding="utf-8")
    section = extract_section(content, _AUDIT_SECTION_PATTERN)
    if not section:
        pytest.skip("CHANGELOG.md has no audit-cycle section; convention not yet applicable")
    return section


def test_audit_cycle_section_present():
    """The audit-cycle heading exists -- catches an accidental rewrite that breaks all subsequent scans."""
    section = _load_audit_cycle_section()
    assert "Wave" in section or "wave" in section, "audit-cycle section does not mention Wave* -- scope changed?"
    assert len(section) > 500, f"audit-cycle section suspiciously short ({len(section)} chars)"


def test_each_fix_bullet_mentions_a_sensor_or_meta_test():
    """Every bullet in the audit-cycle section tagged as a fix MUST also mention a sensor file / meta-test /
    behavioural-equivalence sensor / verified-already-fixed disposition.

    Soft threshold: up to ``_MAX_UNCITED_FIX_FRACTION`` of fix bullets may carry no sensor reference and still
    pass (covers genuinely doc-only fixes -- comment scrubs, CHANGELOG corrections). Hard-error only when the
    fraction exceeds the threshold AND there is at least one cited bullet (otherwise the section is too small
    to drift-detect on).
    """
    assert_changelog_bullets_satisfy_pattern(
        _CHANGELOG,
        _FIX_BULLET_PATTERN,
        _SENSOR_REFERENCE_PATTERN,
        section_pattern=_AUDIT_SECTION_PATTERN,
        max_unsatisfied_fraction=_MAX_UNCITED_FIX_FRACTION,
        label="fix bullet",
    )


# NOTE: the former ``test_known_wave17_findings_cite_their_sensors`` was removed
# when CHANGELOG.md was rewritten into a lean, user-facing Keep-a-Changelog file.
# That test hard-pinned specific ``test_regression_w17_*`` sensor FILENAMES to
# appear verbatim in the CHANGELOG; the lean rewrite intentionally moved that
# per-fix engineering detail into git history, and several of those sensor files
# were themselves renamed to drop wave-N tokens. The generic drift signal
# (``test_each_fix_bullet_mentions_a_sensor_or_meta_test``) still applies to any
# future dated audit-cycle section.
