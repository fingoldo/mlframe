"""biz_val tests for ``mlframe.feature_engineering.categorical`` --
``compute_countaggs`` + ``get_countaggs_names``.

Per CLAUDE.md: each test asserts a SYNTHETIC measurable WIN that
locks in the categorical-aggregate contract. Naming:
``test_biz_val_categorical_<fn>_<scenario>``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")


def _make_cat_series(n=200, n_unique=5, seed=42):
    """Categorical series with ``n_unique`` distinct values."""
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice([f"cat_{i}" for i in range(n_unique)], size=n), name="cat")


# ---------------------------------------------------------------------------
# compute_countaggs
# ---------------------------------------------------------------------------


def test_biz_val_categorical_compute_countaggs_returns_fixed_length_vector():
    """``compute_countaggs`` must produce a fixed-length feature
    vector regardless of input series length. This is the headline
    "arbitrary categorical -> fixed numeric vector" contract."""
    from mlframe.feature_engineering.categorical import (
        compute_countaggs,
        get_countaggs_names,
    )

    arr_short = _make_cat_series(n=50, n_unique=5)
    arr_long = _make_cat_series(n=2000, n_unique=5)
    out_short = compute_countaggs(arr_short)
    out_long = compute_countaggs(arr_long)
    names = get_countaggs_names()
    assert len(out_short) == len(out_long), f"output dim must be invariant to input length; short={len(out_short)}, long={len(out_long)}"
    # Names length must match output length (contract for downstream
    # column-naming).
    assert len(names) == len(out_short), f"get_countaggs_names() must match compute_countaggs output length; got names={len(names)}, output={len(out_short)}"


@pytest.mark.parametrize("n_unique", [2, 5, 10, 50])
def test_biz_val_categorical_compute_countaggs_handles_cardinality(n_unique):
    """Must handle low / medium / high cardinality without changing
    output shape. Catches regressions where the aggregator grows
    unbounded with unique count."""
    from mlframe.feature_engineering.categorical import (
        compute_countaggs,
        get_countaggs_names,
    )

    arr = _make_cat_series(n=500, n_unique=n_unique)
    out = compute_countaggs(arr)
    names = get_countaggs_names()
    assert len(out) == len(names), f"cardinality={n_unique}: output len {len(out)} != names len {len(names)}"


def test_biz_val_categorical_compute_countaggs_top_n_changes_output_shape():
    """``counts_top_n=3`` must produce MORE features than ``=1`` --
    each extra top-N entry contributes count + value pair."""
    from mlframe.feature_engineering.categorical import get_countaggs_names

    names_n1 = get_countaggs_names(counts_top_n=1)
    names_n3 = get_countaggs_names(counts_top_n=3)
    assert len(names_n3) > len(names_n1), f"counts_top_n=3 ({len(names_n3)}) must produce more names than =1 ({len(names_n1)})"


def test_biz_val_categorical_compute_countaggs_normalize_changes_values():
    """``counts_normalize=True/False`` must produce DIFFERENT
    aggregate values (normalisation divides counts by n).

    The output may contain a MIX of numeric counts AND string top-
    values; element-wise equality check tolerates that."""
    from mlframe.feature_engineering.categorical import compute_countaggs

    arr = _make_cat_series(n=300, n_unique=5)
    out_norm = list(compute_countaggs(arr, counts_normalize=True))
    out_raw = list(compute_countaggs(arr, counts_normalize=False))
    # Both must be same length but differ in at least one numeric position.
    assert len(out_norm) == len(out_raw)
    different = False
    for a, b in zip(out_norm, out_raw):
        try:
            if abs(float(a) - float(b)) > 1e-6:
                different = True
                break
        except (TypeError, ValueError):
            # Non-numeric (top value); compare via ==
            if a != b:
                different = True
                break
    assert different, "counts_normalize flag had zero effect on output"


def test_biz_val_categorical_compute_countaggs_singleton_series_smoke():
    """Edge case: single-value series must not crash. Catches
    regressions in the unique-count == 1 branch."""
    from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names

    arr = pd.Series(["only_value"] * 100, name="cat")
    out = compute_countaggs(arr)
    # Behavioural: single-unique-value series must NOT crash AND the value tuple length must match the names
    # returned by the symmetric get_countaggs_names() call. is_not_None alone passed even when the function
    # silently emitted a value tuple whose length disagreed with the canonical name list (a real bug class - the
    # downstream tabular concat aligns by position, so a mismatch silently misaligns columns).
    assert out is not None, "compute_countaggs returned None on singleton-unique input"
    assert len(out) > 0, "compute_countaggs returned empty output on singleton-unique input"
    names = get_countaggs_names()
    assert len(names) == len(out), (
        f"name list length ({len(names)}) must match value list length ({len(out)}) for singleton input - "
        f"misalignment silently misaligns the downstream tabular concat"
    )


def test_biz_val_categorical_compute_countaggs_top_n_zero_smoke():
    """``counts_top_n=0`` must complete cleanly (no top-N features included). Catches regression in the zero-branch.

    The corresponding ``get_countaggs_names()`` invocation must yield zero entries with top-N suffix conventions
    (`_top_value_*` / `_top_count_*`) when top-N is disabled. A bare ``out is not None`` check would pass even if
    the function silently emitted top-N positions in the value list despite the disable flags."""
    from mlframe.feature_engineering.categorical import compute_countaggs, get_countaggs_names

    arr = _make_cat_series(n=200, n_unique=5)
    out = compute_countaggs(
        arr,
        counts_top_n=0,
        counts_return_top_counts=False,
        counts_return_top_values=False,
    )
    assert out is not None
    assert len(out) > 0
    names = get_countaggs_names(
        counts_top_n=0,
        counts_return_top_counts=False,
        counts_return_top_values=False,
    )
    assert len(names) == len(out), (
        f"name list length ({len(names)}) must match value list length ({len(out)}); disabled top-N flags can't "
        "silently leak top-N positions into the value tuple"
    )
    assert not any(("top_value_" in nm) or ("top_count_" in nm) or ("top_n_" in nm) for nm in names), (
        f"counts_top_n=0 + disabled returns must yield no top-N feature names; got names={names[:20]}"
    )


def test_biz_val_categorical_get_countaggs_names_no_duplicates():
    """``get_countaggs_names()`` must return unique names -- duplicates
    would create columns that collide in a downstream DataFrame."""
    from mlframe.feature_engineering.categorical import get_countaggs_names

    names = get_countaggs_names()
    assert len(names) == len(set(names)), f"duplicate names in get_countaggs_names(); names={names}, unique count={len(set(names))}"
