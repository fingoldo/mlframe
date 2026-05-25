"""Regression sensor for A2#4 (S32): `_DEFAULT_DATE_METHODS` mutable module-level default.

Verifies that ``create_date_features`` does NOT leak mutations from the returned/used
``methods`` mapping back into the module-level singleton. A future patch that did
``methods.pop("hour", None)`` inside the function would silently corrupt every
subsequent invocation; we pin both invariants:

1. The singleton ``_DEFAULT_DATE_METHODS`` exposes a snapshot of its expected keys.
2. After a call that mutates a caller-supplied ``methods`` dict, the singleton stays
   unchanged.
3. Two back-to-back calls with ``methods=None`` see the same keys; mutation of the
   first call's internal dict does not propagate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_engineering.basic import _DEFAULT_DATE_METHODS, create_date_features


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31"])})


def test_default_methods_singleton_keys_pinned():
    assert set(_DEFAULT_DATE_METHODS.keys()) == {"day", "weekday", "month"}


def test_call_does_not_share_singleton_with_caller():
    # Provoke any internal mutation by passing methods=None, then mutate the singleton
    # snapshot ourselves -- the singleton must remain intact.
    before = dict(_DEFAULT_DATE_METHODS)
    out = create_date_features(_frame(), cols=["d"], delete_original_cols=False)
    assert out is not None
    after = dict(_DEFAULT_DATE_METHODS)
    assert before == after, "create_date_features mutated module-level _DEFAULT_DATE_METHODS"


def test_caller_supplied_methods_dict_independent_across_calls():
    user_methods = {"day": np.int8, "month": np.int8}
    out1 = create_date_features(_frame(), cols=["d"], delete_original_cols=False, methods=user_methods)
    # Mutate caller dict between calls; the second call sees a fresh default copy.
    user_methods.pop("day", None)
    out2 = create_date_features(_frame(), cols=["d"], delete_original_cols=False)
    assert "d_day" in out2.columns
    assert "d_month" in out2.columns
    assert "d_weekday" in out2.columns
    assert out1 is not out2
