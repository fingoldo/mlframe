"""Nine baselines hold zero entries, and a new violation must not be silenced by adding one.

The baseline gates in this directory compare current findings against a frozen set and fail on anything
new. Nine of those frozen sets are empty, which makes their gates zero-tolerance: any finding at all is a
new finding. That property is not enforced anywhere -- it is a fact about the files' current contents, and
appending one entry to a JSON file quietly converts a blocking gate into one that tolerates exactly the
violation someone did not want to fix.

The 2026-09-08 review raised the general version of this (baselines warn about drained entries on stderr
and pass regardless, so a regression at a stale `file:line` key is invisible). The general fix has to
contend with `file:line` keys drifting on unrelated edits, which would make every refactor red. For these
nine there is no such tension: the set is empty, the correct size is zero, and it can simply be asserted.
"""

from __future__ import annotations

import orjson
from pathlib import Path

import pytest

from ._scan_guard import assert_scanned_enough

META_DIR = Path(__file__).resolve().parent

# There are 27 baseline files today. The floor is well under that: it separates "the directory was not
# found" from "the directory was scanned", which is the only thing this guard is for.
_MIN_BASELINES = 15

# Measured empty on 2026-09-08. Each is a gate whose violation class the codebase has fully drained; the
# point of the list is that draining is the only acceptable way for one to leave it.
ZERO_TOLERANCE_BASELINES: tuple[str, ...] = (
    "_docstring_baseline.json",
    "_fe_noop_copy_baseline.json",
    "_logger_lazy_baseline.json",
    "_module_level_logging_disable_baseline.json",
    "_numba_config_env_mutation_baseline.json",
    "_readonly_to_numpy_mutation_baseline.json",
    "_tick_isinstance_baseline.json",
    "_unprotected_treeexplainer_baseline.json",
)


def _load(name: str):
    """Parse a baseline file, failing loudly rather than skipping if it is gone."""
    path = META_DIR / name
    assert path.is_file(), f"{name} no longer exists; if the gate was retired, remove it from this list too"
    return orjson.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ZERO_TOLERANCE_BASELINES)
def test_zero_tolerance_baseline_is_still_empty(name: str):
    """Adding an entry here downgrades a blocking gate to one that tolerates the new violation."""
    entries = _load(name)
    assert len(entries) == 0, (
        f"{name} gained {len(entries)} entr(y/ies). This baseline is empty by design: its gate is "
        f"zero-tolerance, and appending to it silences the exact violation that was just introduced "
        f"instead of fixing it. Fix the finding, or -- if the entry is genuinely unfixable -- remove this "
        f"file from ZERO_TOLERANCE_BASELINES with the reason in the commit message, so the downgrade is "
        f"visible in review rather than buried in a JSON diff. Entries: {entries}"
    )


def test_list_covers_every_currently_empty_baseline():
    """A baseline that drains to zero should join the list, or the protection never grows."""
    empty_now = set()
    scanned = 0
    for path in META_DIR.glob("_*baseline*.json"):
        try:
            payload = orjson.loads(path.read_text(encoding="utf-8"))
        except (orjson.JSONDecodeError, OSError):
            continue
        scanned += 1
        if len(payload) == 0:
            empty_now.add(path.name)
    assert_scanned_enough(scanned, "tests/test_meta baseline files", minimum=_MIN_BASELINES)
    assert empty_now, "no empty baselines found at all; this gate cannot run and must not report green"
    unlisted = sorted(empty_now - set(ZERO_TOLERANCE_BASELINES))
    assert not unlisted, (
        f"{len(unlisted)} baseline(s) are now empty but not listed as zero-tolerance. Add them, so the "
        f"drained state is protected rather than merely current:\n" + "\n".join(f"  {n}" for n in unlisted)
    )
