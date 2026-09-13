"""Drift tracker: the ``_USER_DEFERRED_*`` / ``_GRANDFATHERED`` whitelists in this directory may not grow.

Every meta-test that tolerates known debt keeps it in a named collection. A deferred item costs nothing to add and
a real cleanup costs effort, so left alone the collections only grow. ``py_ci_shared.deferred_drift`` counts their
entries with ``pyutilz.dev.meta_test_utils.count_user_deferred_entries`` and compares with
``_debt_baseline.json``: a new or grown list fails, and so does a list that SHRANK or vanished until the baseline
follows it down. The local copy this replaced only reported a shrink on stderr, which left the old, higher count as
headroom for the list to grow back into.

Refresh after an intentional change::

    pytest tests/test_meta/test_deferred_drift.py --refresh-debt-baseline
"""

from __future__ import annotations

from pathlib import Path

import orjson

from py_ci_shared.deferred_drift import assert_deferred_lists_not_grown

TEST_META_DIR = Path(__file__).resolve().parent
_BASELINE_PATH = TEST_META_DIR / "_debt_baseline.json"


def regenerate_baseline(path: Path = _BASELINE_PATH) -> None:
    """Rewrite the debt baseline from today's counts, for ``regen_baselines.py``."""
    from pyutilz.dev.meta_test_utils import count_user_deferred_entries

    path.write_text(
        orjson.dumps(count_user_deferred_entries(TEST_META_DIR), option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8") + "\n",
        encoding="utf-8",
    )


def test_user_deferred_lists_havent_grown() -> None:
    """No deferred-debt whitelist grew, appeared, shrank or vanished without the baseline following it."""
    assert_deferred_lists_not_grown(TEST_META_DIR, _BASELINE_PATH)
