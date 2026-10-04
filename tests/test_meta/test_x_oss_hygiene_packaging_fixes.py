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

import logging
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    """Read a repo-relative file as UTF-8 text."""
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# F2: MANIFEST.in no longer references SECURITY.md/CODE_OF_CONDUCT.md, which don't exist
# ---------------------------------------------------------------------------


def _process_manifest_includes(manifest: Path, root_files: list[str]) -> tuple[list[str], int]:
    """Feed the bare ``include X`` directives of ``manifest`` to setuptools' own template processor over ``root_files``; return ``(matched files, warning count)``."""
    from setuptools._distutils.filelist import FileList

    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        """Collect the records setuptools' template processor emits for an include that matched nothing."""

        def emit(self, record):
            """Keep ``record``."""
            records.append(record)

    handler = _Collect(level=logging.WARNING)
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    try:
        file_list = FileList()
        file_list.allfiles = list(root_files)
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("include ") and not re.findall(r"[*?\[]", line):
                file_list.process_template_line(line)
    finally:
        root.removeHandler(handler)
        root.setLevel(previous_level)
    return list(file_list.files), len(records)


def _root_files() -> list[str]:
    """Names of the regular files directly under the repository root."""
    return sorted(p.name for p in REPO_ROOT.iterdir() if p.is_file())


def test_manifest_processor_warns_for_a_missing_include_and_matches_an_existing_one(tmp_path):
    """A directive naming a file that is absent produces a warning; one naming a present file is matched without any."""
    manifest = tmp_path / "MANIFEST.in"
    manifest.write_text("include A.md\ninclude NOPE.md\nrecursive-include src *.py\n", encoding="utf-8")
    matched, warnings = _process_manifest_includes(manifest, ["A.md"])
    assert matched == ["A.md"]
    assert warnings == 1


def test_f2_manifest_no_longer_includes_nonexistent_files():
    """MANIFEST.in's include directives match files that exist: the nonexistent SECURITY.md and CODE_OF_CONDUCT.md are not among them."""
    assert not (REPO_ROOT / "SECURITY.md").exists() and not (REPO_ROOT / "CODE_OF_CONDUCT.md").exists()
    matched, warnings = _process_manifest_includes(REPO_ROOT / "MANIFEST.in", _root_files())
    assert matched, "MANIFEST.in declares no bare include directives that match a root file"
    assert warnings == 0, "F2 REGRESSION: MANIFEST.in must not reference nonexistent files"


def test_f2_manifest_referenced_files_all_exist():
    """Every remaining bare `include X` line in MANIFEST.in must reference a real file."""
    matched, warnings = _process_manifest_includes(REPO_ROOT / "MANIFEST.in", _root_files())
    assert matched
    assert warnings == 0, f"MANIFEST.in references {warnings} missing file(s)"


# ---------------------------------------------------------------------------
# F3: antropy is no longer declared with two different version floors
# ---------------------------------------------------------------------------


def test_f3_antropy_not_duplicated_in_signal_extra():
    """F3 antropy not duplicated in signal extra."""
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - repo's own floor is py39, but this test file itself runs under whatever collects it
        import tomli as tomllib  # type: ignore[no-redef]

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
    """The README paragraph that starts "The core install pulls" links readers to pyproject.toml."""
    pointers = re.findall(r"The core install pulls[^`]{0,400}`(pyproject\.toml)`", _read("README.md"))
    assert pointers == ["pyproject.toml"], (
        "F4 REGRESSION: the core-install description must point readers to pyproject.toml's "
        "[project.dependencies] instead of re-enumerating (and drifting out of sync with) the list"
    )


# ---------------------------------------------------------------------------
# F5: docs/README.md's index no longer omits the 3 live docs wired into mkdocs.yml's nav
# ---------------------------------------------------------------------------


def test_f5_docs_readme_lists_previously_missing_docs():
    """docs/README.md's index links to each of the previously missing docs, and every linked doc exists on disk."""
    linked = re.findall(r"\]\(([^)#]+\.md)\)", _read("docs/README.md"))
    assert linked, "docs/README.md links to no markdown docs"
    for missing_doc in ("visualization.md", "SHAP_PROXIED_FS_GAME_THEORY.md", "gallery/index.md"):
        assert missing_doc in linked, f"F5 REGRESSION: docs/README.md's index must list {missing_doc}"
        assert (REPO_ROOT / "docs" / missing_doc).is_file()


# ---------------------------------------------------------------------------
# F6: docs/gallery/index.md's summary is no longer stale vs the real on-disk PNG count
# ---------------------------------------------------------------------------


def test_f6_gallery_index_total_matches_real_png_count():
    """The gallery index's 'Total images: N' summary equals the number of PNGs on disk."""
    claims = re.findall(r"Total images:\s*(\d+)", _read("docs/gallery/index.md"))
    assert len(claims) == 1, "docs/gallery/index.md must state exactly one 'Total images: N' summary line"
    claimed_total = int(claims[0])

    real_total = len(list((REPO_ROOT / "docs" / "gallery").rglob("*.png")))
    assert claimed_total == real_total, f"F6 REGRESSION: index.md claims {claimed_total} images but {real_total} PNGs exist on disk"


# ---------------------------------------------------------------------------
# F7: the cryptic (AP12)/(AP13) tags are gone from the public doc titles
# ---------------------------------------------------------------------------


def test_f7_no_bare_ap_tags_in_doc_titles():
    """The public doc titles carry no unexplained ``(APnn)`` tag, and the tag pattern does match one."""
    assert re.search(r"\(AP\d+\)", "# Title (AP12)")
    for doc in ("docs/calibration_policy.md", "docs/honest_diagnostics_guide.md"):
        first_line = _read(doc).splitlines()[0]
        assert not re.search(r"\(AP\d+\)", first_line), f"F7 REGRESSION: {doc}'s title still carries an unexplained (APnn) tag"
