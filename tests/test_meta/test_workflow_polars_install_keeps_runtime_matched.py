"""A workflow that swaps polars must swap its native runtime wheel with it.

Since polars 1.37 the native module ships as ``polars-runtime-32`` and has to match ``polars`` exactly. ``pip install --no-deps polars==X``
leaves the runtime of whatever version was installed before, polars then imports without its native module and every use fails with
``NameError: name 'plr' is not defined`` (the first polars-matrix run went red in every leg for this reason).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_PIP_INSTALL = re.compile(r"\bpip\s+install\b[^\n]*")
_BARE_POLARS = re.compile(r"(?<![\w-])polars(?![\w-])")


def offending_lines(text: str) -> list[str]:
    """The ``pip install`` lines that name polars next to ``--no-deps`` without naming polars-runtime."""
    bad = []
    for m in _PIP_INSTALL.finditer(text):
        line = m.group(0)
        if "--no-deps" in line and _BARE_POLARS.search(line) and "polars-runtime" not in line:
            bad.append(line.strip())
    return bad


def test_the_detector_flags_a_no_deps_polars_swap_and_accepts_a_paired_one() -> None:
    """The scan has teeth: the broken form from the first polars-matrix run is flagged, the paired and the plain forms are not."""
    assert offending_lines('pip install --force-reinstall --no-deps "polars==1.36.1"')
    assert not offending_lines('pip install --no-deps "polars==2.0.0" "polars-runtime-32==2.0.0"')
    assert not offending_lines('pip install "polars==1.36.1"')
    assert not offending_lines("pip install --no-deps polars-ds")


def test_no_workflow_swaps_polars_without_its_runtime() -> None:
    """Every workflow and composite action under .github installs polars together with its matching runtime wheel."""
    files = sorted((REPO_ROOT / ".github").rglob("*.y*ml"))
    assert files, "no workflow files found"
    found = {f.relative_to(REPO_ROOT).as_posix(): offending_lines(f.read_text(encoding="utf-8")) for f in files}
    bad = {name: lines for name, lines in found.items() if lines}
    assert not bad, f"pip install --no-deps polars without polars-runtime leaves a mismatched native module: {bad}"
