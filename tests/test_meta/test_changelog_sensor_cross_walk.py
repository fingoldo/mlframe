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
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


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
    matches = list(_AUDIT_SECTION_PATTERN.finditer(content))
    if not matches:
        pytest.skip("CHANGELOG.md has no audit-cycle section; convention not yet applicable")
    start = matches[0].start()
    end = matches[1].start() if len(matches) > 1 else len(content)
    return content[start:end]


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
    section = _load_audit_cycle_section()
    bullets = re.findall(r"^- \*\*[^*]+\*\*[^\n]*(?:\n[ \t]+[^\n-][^\n]*)*", section, re.MULTILINE)
    fix_bullets = [b for b in bullets if _FIX_BULLET_PATTERN.search(b)]
    if not fix_bullets:
        pytest.skip("audit-cycle section has no fix bullets; convention can't drift")
    uncited = [b for b in fix_bullets if not _SENSOR_REFERENCE_PATTERN.search(b)]
    cited_n = len(fix_bullets) - len(uncited)
    fraction = len(uncited) / len(fix_bullets)
    if fraction > _MAX_UNCITED_FIX_FRACTION and cited_n > 0:
        bullet_titles = [re.match(r"^- \*\*([^*]+)\*\*", b).group(1) for b in uncited if re.match(r"^- \*\*([^*]+)\*\*", b)]
        raise AssertionError(
            f"{len(uncited)}/{len(fix_bullets)} fix bullets ({fraction:.0%}) lack a sensor / meta-test reference "
            f"(threshold {_MAX_UNCITED_FIX_FRACTION:.0%}). Per ``feedback_test_every_bug_fix`` every fix should "
            f"cite its regression test. Bullets:\n  - " + "\n  - ".join(bullet_titles)
        )


def test_known_wave17_findings_cite_their_sensors():
    """Explicit per-finding cross-walk on the Wave 17 entries.

    Reads the WHOLE CHANGELOG (not just the first ``_load_audit_cycle_section``
    head) so newer ``## YYYY-MM-DD - Wave N`` entries pushed ahead of Wave 17
    don't move the cited sensors out of the soft-scoped scan window. The
    citations must persist anywhere in the file.
    """
    content = _CHANGELOG.read_text(encoding="utf-8")
    required = {
        "A5#4": "test_regression_w17_get_training_configs_memo.py",
        "A5#16": "test_regression_w17_get_training_configs_memo.py",
        "B1 F8": "test_regression_w17_b1_f8_multilabel_chain_order_metamorphic.py",
        "sklearn_matrix marker": "test_sklearn_matrix_marker_invariants.py",
    }
    missing: list[str] = []
    for finding, sensor in required.items():
        if sensor not in content:
            missing.append(f"{finding!r} -> {sensor}")
    assert not missing, (
        "the following Wave 17 entries do not cite the expected sensor file in "
        "CHANGELOG.md -- either the sensor was renamed, the entry was omitted, "
        "or the citation is missing:\n  "
        + "\n  ".join(missing)
    )
