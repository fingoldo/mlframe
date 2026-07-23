"""Regression tests for audits/full_audit_2026-07-21/x_oss_hygiene_packaging.md findings F2-F7.

F1 (CHANGELOG.md's [Unreleased] section violating its own "lean and user-focused" policy) was
ALREADY resolved by an earlier session's compression pass (before this audit report was even
generated against the pre-compression file) -- confirmed no profiling-narrative bloat pattern
(round-by-round wall-clock numbers, cProfile tottime/cumtime, audit-finding-ID references) remains.

PR1 (add CODE_OF_CONDUCT.md) is explicitly declined per standing user instruction -- F2 is closed
by removing the two dead MANIFEST.in include lines instead. PR2 (mkdocs nav/docs sync CI check),
PR3 (dependency-duplication lint), PR4 (cut a 0.10.0/1.0.0 release) are process proposals with no
reported bug -- deferred. PR5 (blanket eol=lf in .gitattributes) assessed and deferred: forcing it
would trigger a repo-wide line-ending rewrite on the next checkout, exactly the CRLF-mangling risk
class this project's own conventions are set up to avoid -- not worth the blast radius for one
`.sh`-adjacent hygiene nit.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    """Read a repo-relative file as UTF-8 text."""
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# F2: MANIFEST.in no longer references SECURITY.md/CODE_OF_CONDUCT.md, which don't exist
# ---------------------------------------------------------------------------


def test_f2_manifest_no_longer_includes_nonexistent_files():
    """F2 manifest no longer includes nonexistent files."""
    text = _read("MANIFEST.in")
    assert "SECURITY.md" not in text, "F2 REGRESSION: MANIFEST.in must not reference the nonexistent SECURITY.md"
    assert "CODE_OF_CONDUCT.md" not in text, "F2 REGRESSION: MANIFEST.in must not reference the nonexistent CODE_OF_CONDUCT.md"


def test_f2_manifest_referenced_files_all_exist():
    """Every remaining bare `include X` line in MANIFEST.in must reference a real file."""
    text = _read("MANIFEST.in")
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("include ") and not any(c in line for c in "*?["):
            fname = line.removeprefix("include ").strip()
            assert (REPO_ROOT / fname).exists(), f"MANIFEST.in references missing file: {fname}"


# ---------------------------------------------------------------------------
# F3: antropy is no longer declared with two different version floors
# ---------------------------------------------------------------------------


def test_f3_antropy_not_duplicated_in_signal_extra():
    """F3 antropy not duplicated in signal extra."""
    import tomllib

    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        doc = tomllib.load(f)

    core_deps = doc["project"]["dependencies"]
    signal_extra = doc["project"]["optional-dependencies"]["signal"]

    core_antropy = [d for d in core_deps if d.split(">=")[0].strip() == "antropy"]
    extra_antropy = [d for d in signal_extra if d.split(">=")[0].strip() == "antropy"]

    assert len(core_antropy) == 1, "F3 REGRESSION: antropy must remain exactly once in core dependencies"
    assert len(extra_antropy) == 0, "F3 REGRESSION: antropy must not be duplicated inside the 'signal' optional extra"
    assert core_antropy[0] == "antropy>=0.1.4", "F3 REGRESSION: the core antropy floor must be the higher (0.1.4) one, not silently lowered"


# ---------------------------------------------------------------------------
# F4: README's core-install claim no longer silently omits real hard dependencies
# ---------------------------------------------------------------------------


def test_f4_readme_core_install_claim_points_to_pyproject():
    """F4 readme core install claim points to pyproject."""
    text = _read("README.md")
    assert "pyproject.toml" in text.split("The core install pulls")[1][:400], (
        "F4 REGRESSION: the core-install description must point readers to pyproject.toml's "
        "[project.dependencies] instead of re-enumerating (and drifting out of sync with) the list"
    )


# ---------------------------------------------------------------------------
# F5: docs/README.md's index no longer omits the 3 live docs wired into mkdocs.yml's nav
# ---------------------------------------------------------------------------


def test_f5_docs_readme_lists_previously_missing_docs():
    """F5 docs readme lists previously missing docs."""
    text = _read("docs/README.md")
    for missing_doc in ("visualization.md", "SHAP_PROXIED_FS_GAME_THEORY.md", "gallery/index.md"):
        assert missing_doc in text, f"F5 REGRESSION: docs/README.md's index must list {missing_doc}"


# ---------------------------------------------------------------------------
# F6: docs/gallery/index.md's summary is no longer stale vs the real on-disk PNG count
# ---------------------------------------------------------------------------


def test_f6_gallery_index_total_matches_real_png_count():
    """F6 gallery index total matches real png count."""
    import re

    text = _read("docs/gallery/index.md")
    m = re.search(r"Total images:\s*(\d+)", text)
    assert m is not None, "docs/gallery/index.md must state a 'Total images: N' summary line"
    claimed_total = int(m.group(1))

    real_total = len(list((REPO_ROOT / "docs" / "gallery").rglob("*.png")))
    assert claimed_total == real_total, f"F6 REGRESSION: index.md claims {claimed_total} images but {real_total} PNGs exist on disk"


# ---------------------------------------------------------------------------
# F7: the cryptic (AP12)/(AP13) tags are gone from the public doc titles
# ---------------------------------------------------------------------------


def test_f7_no_bare_ap_tags_in_doc_titles():
    """F7 no bare ap tags in doc titles."""
    import re

    for doc in ("docs/calibration_policy.md", "docs/honest_diagnostics_guide.md"):
        first_line = _read(doc).splitlines()[0]
        assert not re.search(r"\(AP\d+\)", first_line), f"F7 REGRESSION: {doc}'s title still carries an unexplained (APnn) tag"
