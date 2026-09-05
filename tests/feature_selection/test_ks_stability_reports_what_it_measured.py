"""A drift guard must not report a clean bill on columns it never inspected.

`ks_stability_filter` derives its column list, when none is given, as a DOUBLE filter: shared name AND
pandas-numeric dtype. That empties on a rename, a prefixed test frame, a pipeline emitting engineered names
on one side only, or columns arriving as object/string or a non-pandas-numeric extension dtype. The
per-column loop then ran zero times, `rows` stayed empty, and every caller read "no unstable features" --
a fully drifted feature set indistinguishable from a clean one, with no log line and no exception.

The narrower shape sat one level in: a column whose finite values are empty on either side was recorded
`stable=True`, which reads as "checked and fine" rather than "could not be judged".

`stable` stays boolean, because callers use it directly as a mask -- `drop_noninformative_vs_reference`
does `report.loc[report["stable"], "column"]` -- so the distinction is carried by a separate `measured`
column instead of by making `stable` tri-state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import ks_stability_filter


def _frames(n: int = 400, shift: float = 0.0):
    """Train/test frames whose numeric column differs by `shift`."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    test = pd.DataFrame({"a": rng.normal(size=n) + shift, "b": rng.normal(size=n)})
    return train, test


def test_a_rename_between_the_frames_raises_rather_than_reporting_clean():
    """The accident the derivation cannot survive: no column name is shared, so nothing can be screened."""
    train, test = _frames()
    test = test.rename(columns={"a": "test_a", "b": "test_b"})
    with pytest.raises(ValueError, match="no shared numeric columns"):
        ks_stability_filter(train, test)


def test_non_numeric_columns_alone_raise_rather_than_reporting_clean():
    """Shared names but no numeric dtype among them: the second half of the double filter."""
    train = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    test = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    with pytest.raises(ValueError, match="no shared numeric columns"):
        ks_stability_filter(train, test)


def test_the_message_says_what_it_saw():
    """A reader has to be able to tell a rename from an all-categorical frame without re-running anything."""
    train, test = _frames()
    test = test.rename(columns={"a": "test_a", "b": "test_b"})
    with pytest.raises(ValueError) as exc:
        ks_stability_filter(train, test)
    message = str(exc.value)
    assert "2 column(s)" in message
    assert "0 shared" in message


def test_an_explicitly_empty_column_list_is_left_alone():
    """Passing an empty list is the caller's own decision, not the derivation misfiring."""
    train, test = _frames()
    report = ks_stability_filter(train, test, feature_cols=[])
    assert len(report) == 0


def test_a_column_with_no_finite_values_is_marked_unmeasured():
    """`stable=True` there means "not judged"; without the flag it was indistinguishable from "checked and fine"."""
    train, test = _frames()
    train["c"] = np.nan
    test["c"] = np.nan
    report = ks_stability_filter(train, test)
    row = report.loc[report["column"] == "c"].iloc[0]
    assert bool(row["stable"]) is True, "an unjudged column must not be proposed for dropping"
    assert bool(row["measured"]) is False, "an all-NaN column was reported as if its distributions had been compared"


def test_a_measured_column_says_so():
    """The flag has to distinguish, so the ordinary case must carry True."""
    train, test = _frames()
    report = ks_stability_filter(train, test)
    assert bool(report.loc[report["column"] == "a"].iloc[0]["measured"]) is True


def test_stable_stays_a_usable_boolean_mask():
    """`drop_noninformative_vs_reference` indexes with it directly, so its dtype is part of the contract."""
    train, test = _frames()
    train["c"] = np.nan
    test["c"] = np.nan
    report = ks_stability_filter(train, test)
    kept = report.loc[report["stable"], "column"].tolist()
    assert "c" in kept


def test_a_real_shift_is_still_flagged_unstable():
    """Guards the screen itself: the rewrite must not have made everything look stable."""
    train, test = _frames(shift=1.0)
    report = ks_stability_filter(train, test)
    row = report.loc[report["column"] == "a"].iloc[0]
    assert bool(row["measured"]) is True
    assert bool(row["stable"]) is False, "a one-sigma shift should read as unstable"


def test_the_multi_split_path_also_reports_measured():
    """The majority-vote branch builds its own row dict and must carry the same flag."""
    train, test = _frames()
    report = ks_stability_filter(train, test, n_splits=3, split_frac=0.5, random_state=0)
    assert bool(report.loc[report["column"] == "a"].iloc[0]["measured"]) is True
