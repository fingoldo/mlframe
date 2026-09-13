"""No new guard that checks a VALUE may ride on an ``assert`` in ``src/mlframe``.

``python -O`` deletes every ``assert``. A guard that only narrows a type for mypy loses nothing when it goes; a guard
that checks a bound, a sum, a membership or a shape loses the only thing enforcing it, and the code after it runs on
a value it was never allowed to see. ``py_ci_shared.value_bearing_asserts`` keeps ``assert x is not None`` and a
bare name legal and flags the rest, ``isinstance`` included: under ``-O`` that vanishes too.

The tree already holds such asserts, recorded in ``_value_bearing_asserts_baseline.json`` as debt keyed
``file::expression`` so a moved line does not churn it. The baseline only shrinks: converting one to an explicit
``raise`` makes its entry stale, and a stale entry fails until it is removed.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from py_ci_shared.value_bearing_asserts import assert_no_value_bearing_asserts, find_value_bearing_asserts

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "mlframe"
_BASELINE_PATH = Path(__file__).resolve().parent / "_value_bearing_asserts_baseline.json"


def regenerate_baseline(path: Path = _BASELINE_PATH) -> None:
    """Rewrite the debt baseline from today's value-bearing asserts, for ``regen_baselines.py``."""
    offenders, _seen = find_value_bearing_asserts(PACKAGE_ROOT)
    keys = sorted({f"{entry.split('  ', 1)[0].rsplit(':', 1)[0]}::{entry.split('  ', 1)[1]}" for entry in offenders})
    path.write_text(orjson.dumps(keys, option=orjson.OPT_INDENT_2).decode("utf-8") + "\n", encoding="utf-8")


def test_no_new_value_bearing_asserts() -> None:
    """No production assert checks a value beyond the recorded debt, and the debt holds no stale entry."""
    # Over 300 asserts exist in src/mlframe; a walk that saw fewer than 100 has stopped reaching the package.
    assert_no_value_bearing_asserts(PACKAGE_ROOT, min_asserts_seen=100, baseline_path=_BASELINE_PATH)
