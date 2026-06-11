"""Wave 9.1 loop-iter-17 regression: RFECV path column order MUST NOT
depend on PYTHONHASHSEED.

Pre-fix at ``_mrmr_fit_impl.py:864``:
  ``list(set(X.columns) - set(X.columns[selected_vars]))``

``set`` iteration order is randomized by Python's string hashing,
which uses ``PYTHONHASHSEED``. Concrete demo: 5 distinct orderings
observed across seeds 0-4 for a 10-column input. That order is fed
into RFECV's CatBoost, whose tie-broken feature importances produce
different ``support_`` across runs that differ only in their hash seed.

Effect: ``MRMR(random_seed=42, run_additional_rfecv_minutes=5)`` violates
the "same seed -> identical support_" contract. Different processes
(different OS-randomized PYTHONHASHSEED) silently disagree. Breaks
stability_iters audits, blocks bit-exact regression tests.

Severity: medium (opt-in path, but documented stability contract broken).

Fix: order-preserving list comprehension that scans ``X.columns`` in its
canonical order and skips names in the selected set:
  ``_sel = set(X.columns[selected_vars].tolist())``
  ``temp_columns = [c for c in X.columns if c not in _sel]``
"""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest


def _column_order_via_post_fix_pattern(cols, selected):
    """Mirrors the iter-17 fix exactly."""
    X_cols = pd.Index(cols)
    _sel = set(X_cols[selected].tolist())
    return [c for c in X_cols if c not in _sel]


def _column_order_via_pre_fix_pattern(cols, selected):
    """Mirrors the pre-fix pattern - used by the baseline test."""
    X_cols = pd.Index(cols)
    return list(set(X_cols.tolist()) - set(X_cols[selected].tolist()))


def test_post_fix_pattern_is_deterministic_across_hash_seeds():
    """The fix uses an order-preserving filter that doesn't depend on
    ``set`` iteration order. Run the same operation in subprocesses with
    different PYTHONHASHSEED values; outputs must be identical.
    """
    cols = [f"feat_{w}" for w in
             ("alpha", "bravo", "charlie", "delta", "echo",
              "foxtrot", "golf", "hotel", "india", "juliet")]
    selected = [0, 1]
    script = (
        "import sys, pandas as pd\n"
        f"cols = {cols!r}\n"
        f"selected = {selected!r}\n"
        "X_cols = pd.Index(cols)\n"
        "_sel = set(X_cols[selected].tolist())\n"
        "temp = [c for c in X_cols if c not in _sel]\n"
        "sys.stdout.write('|'.join(temp))\n"
    )
    outs = []
    for seed in (0, 1, 2, 3, 4):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(seed)
        r = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=30,
        )
        outs.append(r.stdout.strip())
    assert len(set(outs)) == 1, (
        f"post-fix column order varied across PYTHONHASHSEED: "
        f"got {len(set(outs))} distinct orderings: {outs}"
    )


def test_baseline_pre_fix_pattern_is_nondeterministic():
    """Confirms the pre-fix pattern's non-determinism so the iter-17 fix
    is necessary. If this ever stops being true (CPython removes
    randomized hashing), the fix can be revisited.
    """
    cols = [f"feat_{w}" for w in
             ("alpha", "bravo", "charlie", "delta", "echo",
              "foxtrot", "golf", "hotel", "india", "juliet")]
    selected = [0, 1]
    script = (
        "import sys, pandas as pd\n"
        f"cols = {cols!r}\n"
        f"selected = {selected!r}\n"
        "X_cols = pd.Index(cols)\n"
        "temp = list(set(X_cols.tolist()) - set(X_cols[selected].tolist()))\n"
        "sys.stdout.write('|'.join(temp))\n"
    )
    outs = []
    for seed in (0, 1, 2, 3, 4):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = str(seed)
        r = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, text=True, timeout=30,
        )
        outs.append(r.stdout.strip())
    # Pre-fix: at least 2 distinct orderings expected.
    assert len(set(outs)) >= 2, (
        f"pre-fix pattern was expected to be non-deterministic; "
        f"got only {len(set(outs))} unique orderings: {outs}. If CPython "
        f"changed set iteration semantics, revisit the iter-17 fix."
    )


def test_post_fix_preserves_canonical_column_order():
    """The fix preserves the original column order from ``X.columns``,
    minus the selected names.
    """
    cols = ["a", "b", "c", "d", "e", "f"]
    selected = [1, 3]  # drop 'b' and 'd'
    result = _column_order_via_post_fix_pattern(cols, selected)
    assert result == ["a", "c", "e", "f"]
