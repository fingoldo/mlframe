"""A family that fans out to joblib must carry the FE deadline into its workers.

``fe_deadline_passed`` reads a thread-local, and a joblib worker is a different thread or a different process, so a family that dispatches
loses the budget entirely rather than approximately: every check inside the worker reads unset and the enrichment loop runs to completion with
no warning. ``polynom_pair_fe`` works around it by hand, reading the thread-local on the main thread and threading the value through its
payload signature; nothing made the next family do the same.

Today only that one module both dispatches and checks the deadline, so the bug is latent rather than live. This gate is what keeps it that
way: adding a ``Parallel(...)`` to a module that consults the deadline, without carrying it across, fails here.
"""

from __future__ import annotations

import pathlib
import re

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "mlframe" / "feature_selection" / "filters"
_DISPATCHES = re.compile(r"\bParallel\s*\(|\bdelayed\s*\(")
_CONSULTS = re.compile(r"\bfe_deadline_passed\s*\(")
# The ways a module can legitimately carry the deadline across: the shared wrapper, the scope it is built on, or reading the thread-local
# explicitly to pass it down (which is what polynom_pair_fe does, and which the wrapper exists to replace).
_CARRIES = re.compile(r"DeadlineCarrying|fe_deadline_scope|current_fe_deadline|_fe_deadline_state|_state\.deadline")


def _modules_that_dispatch_and_check():
    """Every filters module that both fans out to joblib and consults the FE deadline."""
    out = []
    for path in sorted(_SRC.rglob("*.py")):
        if "_benchmarks" in path.parts or path.name == "_fe_deadline.py":
            continue
        text = path.read_text(encoding="utf-8")
        if _DISPATCHES.search(text) and _CONSULTS.search(text):
            out.append((path, text))
    return out


def test_every_dispatching_family_carries_the_deadline_into_its_workers():
    """A module that fans out and checks the deadline must also publish it on the worker side."""
    offenders = [str(path.relative_to(_SRC)) for path, text in _modules_that_dispatch_and_check() if not _CARRIES.search(text)]
    assert not offenders, (
        "module(s) that dispatch to joblib and consult the FE deadline without carrying it across the worker boundary, so the budget is "
        "absent there rather than approximate: " + ", ".join(offenders) + ". Wrap the payload in "
        "``_fe_deadline.DeadlineCarrying`` (it captures the dispatching thread's deadline and republishes it inside the worker)."
    )


def test_the_detector_sees_the_module_that_motivated_it():
    """Teeth-check: the scan must actually find the known dispatching consumer, or it would pass vacuously."""
    found = {path.name for path, _ in _modules_that_dispatch_and_check()}
    assert "polynom_pair_fe.py" in found, f"the scan no longer sees the dispatching deadline consumer it was written for: {sorted(found)}"
