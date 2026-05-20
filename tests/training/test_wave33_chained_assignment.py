"""Wave-33 sensor: cleaning.py:93 unreliable _is_view gate -> unconditional copy.

The wave-33 audit found 1 residual P1 in a codebase that already swept
this bug class. The cleaning.py site previously gated a value-refresh
on an unreliable ``values.base`` probe:

Pre-fix:
    vals = sub_df[col].values if col in sub_df else None
    is_view = vals is not None and getattr(vals, "base", None) is not None
    if not is_view:
        sub_df[col] = df.loc[analyse_mask, col]

Problem:
- pandas <2 (copy-on-write OFF default): values.base could be None even
  for a view -> gate fired and SettingWithCopyWarning emitted.
- pandas >=2 (copy-on-write ON default in 2.1+): values always returns
  a fresh ndarray -> base is ALWAYS non-None -> gate INVERTS:
  the refresh NEVER fires -> sub_df[col] stays stale -> value_counts
  reflects pre-mutation state.

Post-fix: unconditional ``sub_df.copy()`` + ``sub_df[col] = df.loc[...]``.
"""
from __future__ import annotations

import pathlib

import mlframe as _mlframe


def test_cleaning_is_view_heuristic_removed():
    """The unreliable ``values.base`` heuristic must be gone."""
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "preprocessing" / "cleaning.py"
    ).read_text(encoding="utf-8")
    # Pre-fix shape MUST be gone:
    assert "is_view = vals is not None and getattr(vals, \"base\", None) is not None" not in src, (
        "Wave 33 P1 regression: the unreliable values.base heuristic "
        "reappeared in cleaning.py; under pandas 2.x CoW the gate "
        "inverts and the refresh silently skips."
    )
    assert "if not is_view:\n            sub_df[col] = df.loc" not in src
    # Post-fix marker:
    assert "Wave 33 P1 fix" in src
    assert "the gate INVERTS" in src


def test_cleaning_unconditional_copy_pattern():
    """The post-fix unconditional sub_df = sub_df.copy() must be present."""
    src = (
        pathlib.Path(_mlframe.__file__).resolve().parent
        / "preprocessing" / "cleaning.py"
    ).read_text(encoding="utf-8")
    assert "sub_df = sub_df.copy() if col in sub_df else sub_df" in src, (
        "Wave 33 P1 regression: defensive copy before sub_df[col] = "
        "assignment is gone; SettingWithCopyWarning will reappear "
        "under pandas <2."
    )
