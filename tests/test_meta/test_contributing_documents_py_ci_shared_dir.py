"""X_CICD_DEPENDENCIES-5 (2026-08-05 audit): pyproject.toml's [tool.ruff] `extend` depends on
$PY_CI_SHARED_DIR, which was never set anywhere in the repo (no workflow, no .pre-commit-config.yaml, no
CONTRIBUTING.md mention) -- confirmed live: `ruff check src/mlframe` with the var unset hard-fails with
"environment variable not found", exactly what CONTRIBUTING.md's own documented dev-setup steps would hit
on the very first `ruff check` a fresh contributor runs. Fixed by adding a clone-and-export step to
CONTRIBUTING.md's dev setup. Meta-test: the setup instructions must mention PY_CI_SHARED_DIR, so this gap
can't silently reopen.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pyproject_ruff_extend_still_depends_on_py_ci_shared_dir():
    """Sanity: this test's premise (the dependency exists) stays true -- if pyproject.toml ever
    stops needing PY_CI_SHARED_DIR, the CONTRIBUTING.md requirement below should be revisited too."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "$PY_CI_SHARED_DIR" in text


def test_contributing_documents_py_ci_shared_dir_setup():
    """CONTRIBUTING.md's dev-setup section must tell a fresh contributor to set PY_CI_SHARED_DIR
    before the first documented `ruff check` command, or that command hard-fails."""
    text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "PY_CI_SHARED_DIR" in text, "CONTRIBUTING.md must document setting PY_CI_SHARED_DIR"

    setup_match = re.search(r"## Development setup\n(.*?)\n## ", text, re.DOTALL)
    assert setup_match is not None, "expected a '## Development setup' section"
    assert "PY_CI_SHARED_DIR" in setup_match.group(
        1
    ), "PY_CI_SHARED_DIR must be set as part of the documented dev-setup steps, before the first ruff invocation"
