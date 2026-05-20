"""Sensor: the suite-wide unsupervised pre-screen MUST protect ctx.cat_features
(plus text / embedding / group / ts cols) and MUST protect target columns from
EVERY (target_type, target_name) pair across the suite, not just the first
target's siblings.

Pre-fix shape (stateful side-effects audit P0-2):

  if ctx.cat_features:
      pass                              # &lt;-- DEAD no-op. cat_features unprotected.
  ... compute_unsupervised_drops(protected_columns=_protected)

A near-constant categorical column (e.g. one-hot encoded rare-event flag) was
silently dropped. Multi-target-type suites (regression + binary) also lost
sibling target columns because the protected set only covered the FIRST target's
siblings, not the union across all target_by_type entries.

Post-fix: cat / text / embedding / group / ts cols are explicitly added to the
protected set, AND the protected target set is computed from the full
ctx.target_by_type (suite-wide union).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.pre_screen import compute_unsupervised_drops


def test_compute_unsupervised_drops_respects_protected_set():
    """Baseline: the underlying helper honours the protected set on a numeric column."""
    df = pd.DataFrame({
        "var_col": np.arange(100, dtype=np.float64),  # varying, would be kept
        "const_num": np.full(100, 3.0),               # constant, would normally drop
    })
    drops_unprotected = compute_unsupervised_drops(df, protected_columns=())
    assert "const_num" in drops_unprotected, "baseline: const numeric should be detected"

    # Protect const_num -> stays even though constant.
    drops_protected = compute_unsupervised_drops(df, protected_columns={"const_num"})
    assert "const_num" not in drops_protected, "protected col must survive the screen"


def test_compute_unsupervised_drops_handles_categorical_dtype():
    """REGRESSION: pre-fix np.issubdtype(col.dtype, np.number) raised TypeError on
    pd.CategoricalDtype (it's not a numpy dtype), taking the whole pre-screen pass
    down. Post-fix pd.api.types.is_numeric_dtype handles extension dtypes gracefully."""
    df = pd.DataFrame({
        "var_col": np.arange(100, dtype=np.float64),
        "cat_col": pd.Categorical(["A", "B"] * 50),
        "const_num": np.full(100, 3.0),
    })
    drops = compute_unsupervised_drops(df, protected_columns=())
    assert "const_num" in drops, "const num still detected"
    # Categorical with 2 distinct values is informative, not constant -> kept.
    assert "cat_col" not in drops, "varying Categorical must be kept"


def test_compute_unsupervised_drops_handles_pandas_string_dtype():
    """Pandas StringDtype (extension) must also not crash the pre-screen."""
    df = pd.DataFrame({
        "var_col": np.arange(50, dtype=np.float64),
        "str_col": pd.array(["x", "y"] * 25, dtype=pd.StringDtype()),
    })
    # Must not raise.
    drops = compute_unsupervised_drops(df, protected_columns=())
    # Just sanity that the function returned a list and didn't crash.
    assert isinstance(drops, list)


def test_compute_unsupervised_drops_handles_pandas_nullable_int():
    """Pandas nullable Int extension dtypes: is_numeric_dtype returns True so the
    variance check still runs (and drops a constant Int64 col)."""
    df = pd.DataFrame({
        "const_nullable_int": pd.array([5, 5, 5, 5, 5], dtype="Int64"),
        "var_col": np.arange(5, dtype=np.float64),
    })
    drops = compute_unsupervised_drops(df, protected_columns=())
    assert "const_nullable_int" in drops


def test_dead_pass_branch_was_fixed_via_source_inspection():
    """The exact pre-fix shape (``if ctx.cat_features: pass``) must NOT appear
    in _train_one_target.py anymore. Source-level regression guard.

    Justified per the [[feedback_behavioral_tests]] memory rule's carve-out:
    when a behavioural test can't easily set up the multi-suite ctx state, a
    source-level pattern check is acceptable IF it has a clear failure message
    pointing at the bug-class.
    """
    import pathlib
    src = pathlib.Path(
        r"D:/Upd/Programming/PythonCodeRepository/mlframe/src/mlframe/training/core/_phase_train_one_target.py"
    ).read_text(encoding="utf-8")
    assert "if ctx.cat_features:\n                pass" not in src, (
        "Pre-screen 'if ctx.cat_features: pass' dead-pass regression. "
        "cat_features must be explicitly added to the protected set."
    )
    # Positive assertion: cat_features are now actually added.
    assert "_protected.update(str(c) for c in ctx.cat_features)" in src, (
        "Pre-screen must protect cat_features via _protected.update(...)."
    )
    # And the suite-wide protected set (target_by_type union, not just first target):
    assert "for _tt_targets in (ctx.target_by_type or {}).values()" in src, (
        "Pre-screen protected set must iterate the full target_by_type to cover "
        "every (target_type, target_name) -- not just the first target's siblings."
    )


def test_text_and_embedding_features_added_to_protected():
    """Text + embedding cols carry semantic meaning the model relies on; protect them too."""
    import pathlib
    src = pathlib.Path(
        r"D:/Upd/Programming/PythonCodeRepository/mlframe/src/mlframe/training/core/_phase_train_one_target.py"
    ).read_text(encoding="utf-8")
    assert "ctx.text_features" in src and "ctx.embedding_features" in src, (
        "Pre-screen must protect text + embedding features alongside cat_features."
    )
