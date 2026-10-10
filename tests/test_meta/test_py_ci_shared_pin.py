"""py-ci-shared is pinned to one full commit SHA everywhere, and the installed package is that commit.

The gate package reaches this repo through four doors: the requirement in requirements-dev.txt, the reusable workflows and
composite actions in ``uses:`` refs, the ``py-ci-shared-ref`` input those workflows fetch configs at, and the pre-commit ``rev:``.
They drifted to five different commits here, several commented ``# v1`` although 19 to 62 commits apart, so CI's lint jobs, its
test job and the local hooks each ran different gate code. A bump is one search-and-replace of the SHA; this fails a bump that
misses a line, a ref that is a tag or branch (which can move under the repo), and a ``# vX.Y.Z`` comment that disagrees.
"""

from __future__ import annotations

import re
from pathlib import Path

from py_ci_shared.git_dependency_pins import assert_installed_includes_pin

REPO_ROOT = Path(__file__).resolve().parents[2]

_SHA = re.compile(r"[0-9a-f]{40}")
_TAG = re.compile(r"#\s*(v\d+\.\d+\.\d+)\b")
_REF_PATTERNS = (
    re.compile(r"uses:\s*fingoldo/py-ci-shared/\.github/[\w./-]+@(?P<ref>[^\s#]+)"),
    re.compile(r"py-ci-shared-ref:\s*['\"]?(?P<ref>[^\s'\"#]+)"),
    re.compile(r"py-ci-shared\s*@\s*git\+https://github\.com/fingoldo/py-ci-shared(?:\.git)?(?:@(?P<ref>[^\s;#'\"]+))?"),
)
_PRECOMMIT_REPO = re.compile(r"-\s*repo:\s*https://github\.com/fingoldo/py-ci-shared(?:\.git)?\s*\n\s*rev:\s*(?P<ref>[^\s#]+)(?P<rest>[^\n]*)")


def _pin_files() -> list[Path]:
    """Every file that can carry a py-ci-shared ref."""
    files = sorted((REPO_ROOT / ".github").rglob("*.yml")) + sorted((REPO_ROOT / ".github").rglob("*.yaml"))
    files += [REPO_ROOT / name for name in ("requirements-dev.txt", "pyproject.toml", ".pre-commit-config.yaml") if (REPO_ROOT / name).exists()]
    return files


def _pins() -> list[tuple[str, str, str]]:
    """``(where, ref, tag comment)`` for every py-ci-shared ref; an unpinned requirement has ref ``""``."""
    found: list[tuple[str, str, str]] = []
    for path in _pin_files():
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for pattern in _REF_PATTERNS:
                m = pattern.search(line)
                if m:
                    tag = _TAG.search(line[m.end() :])
                    found.append((f"{rel}:{lineno}", m.group("ref") or "", tag.group(1) if tag else ""))
                    break
        for m in _PRECOMMIT_REPO.finditer(text):
            tag = _TAG.search(m.group("rest"))
            found.append((f"{rel}:{text.count(chr(10), 0, m.start('ref')) + 1}", m.group("ref"), tag.group(1) if tag else ""))
    return found


def _the_pin() -> str:
    """The one SHA every py-ci-shared ref names; fails on a second SHA, a non-SHA ref or a missing/disagreeing release comment."""
    pins = _pins()
    # The requirement, the pre-commit rev and the workflow refs: fewer means the patterns stopped matching this repo's spellings.
    assert len(pins) >= 10, f"found only {len(pins)} py-ci-shared refs: {pins}"
    bad = [p for p in pins if not _SHA.fullmatch(p[1])]
    assert not bad, "every py-ci-shared ref must be a full 40-character commit SHA, not a tag, branch or nothing:\n  " + "\n  ".join(f"{w}: {r or '(unpinned)'}" for w, r, _ in bad)
    shas = {r for _, r, _ in pins}
    assert len(shas) == 1, "py-ci-shared is pinned to several commits; update every ref to one:\n  " + "\n  ".join(f"{w}: {r[:12]}" for w, r, _ in pins)
    untagged = [w for w, _, t in pins if not t]
    assert not untagged, f"every pin carries the release it is as a `# vX.Y.Z` comment; missing at: {untagged}"
    tags = {t for _, _, t in pins}
    assert len(tags) == 1, f"one SHA commented as several releases: {sorted(tags)}"
    return shas.pop()


def test_every_py_ci_shared_ref_is_one_full_sha_with_one_release_comment():
    """One commit for the requirement, every workflow and action ref, every py-ci-shared-ref input and the pre-commit rev."""
    assert _SHA.fullmatch(_the_pin())


def test_the_installed_py_ci_shared_includes_the_pin():
    """The gate code this interpreter runs contains the pinned commit, so a local run and CI judge with the same rules."""
    assert_installed_includes_pin("py_ci_shared", _the_pin(), dist="py-ci-shared")


_UNRELEASED_MARK = re.compile(r"#\s*v\d+\.\d+\.\d+\+(?P<suffix>\w+)")
_UNRELEASED_ALLOWED: frozenset[str] = frozenset()


def test_the_pin_is_a_released_tag_not_an_unreleased_commit() -> None:
    """A ``# vX.Y.Z+suffix`` comment marks a commit that has no release tag yet; two sessions each pinning their own such commit collide on every file.

    Release py-ci-shared first (tag after its CI is green) and pin the tagged commit; add the suffix to ``_UNRELEASED_ALLOWED`` only for a
    deliberate, short-lived exception.
    """
    marked: dict[str, str] = {}
    for path in _pin_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = _UNRELEASED_MARK.search(line)
            if m and _SHA.search(line):
                marked[f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno}"] = m.group("suffix")
    unreleased = {where: s for where, s in marked.items() if s not in _UNRELEASED_ALLOWED}
    assert not unreleased, f"py-ci-shared is pinned to an unreleased commit (release it, then pin the tag's SHA): {unreleased}"


def test_the_unreleased_marker_matches_the_suffix_form_and_not_a_plain_release() -> None:
    """The marker regex flags ``# v1.22.3+x`` and leaves ``# v1.22.3`` alone, so the test above can fail."""
    assert _UNRELEASED_MARK.search("sha  # v1.22.3+pip_audit_ignore_vulns").group("suffix") == "pip_audit_ignore_vulns"
    assert _UNRELEASED_MARK.search("sha  # v1.22.3") is None
