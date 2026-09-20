"""A composite spec the suite cannot build a base column for must not reach training.

Discovery may add base columns to its OWN frame (the engineered per-group causal bases are on by default), and those
columns never reach the suite's split frames. The column builder answers a name it cannot find with an all-NaN column,
so such a spec was trained on a NaN target while its gate verdicts had been measured on a column the trainer never sees.
The bases are functions of past ``y``, which a predict frame does not carry, so they are screening-only by construction.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from mlframe.training.core._misc_helpers import _build_full_column_from_splits
from mlframe.training.core._phase_composite_discovery import _drop_specs_whose_bases_the_suite_cannot_materialise


def _spec(name: str, base: str, extra: tuple = ()) -> SimpleNamespace:
    """A duck-typed stand-in carrying only the spec fields the base check reads."""
    return SimpleNamespace(name=name, transform_name="linear_residual", base_column=base, extra_base_columns=extra)


def _frames() -> tuple:
    """Train/val/test frames carrying a real base column but not the engineered one."""
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "real_base": [0.5, 0.6, 0.7]})
    return df, df, df


def test_spec_on_a_discovery_only_base_is_dropped_with_a_reason():
    """The engineered base exists only inside discovery, so its spec is dropped and recorded as a failure."""
    disc = SimpleNamespace(specs_=[_spec("keep", "real_base"), _spec("drop", "y__gcausal_lag1")])
    dropped = _drop_specs_whose_bases_the_suite_cannot_materialise(disc, _frames(), "y")
    assert [s.name for s in disc.specs_] == ["keep"], "only the spec whose base the suite can build may survive"
    assert len(dropped) == 1 and dropped[0]["name"] == "drop"
    assert dropped[0]["rejected"] is True
    assert "y__gcausal_lag1" in dropped[0]["reason"]


def test_a_missing_extra_base_drops_the_spec_too():
    """Multi-base specs need every column: a missing extra base is as unbuildable as a missing primary."""
    disc = SimpleNamespace(specs_=[_spec("multi", "real_base", extra=("y__gcausal_expmean",))])
    dropped = _drop_specs_whose_bases_the_suite_cannot_materialise(disc, _frames(), "y")
    assert disc.specs_ == [] and len(dropped) == 1
    assert "y__gcausal_expmean" in dropped[0]["reason"]


def test_specs_survive_when_the_column_check_has_nothing_to_check_against():
    """With no readable frames the check cannot prove absence, so it keeps the specs rather than dropping them blindly."""
    disc = SimpleNamespace(specs_=[_spec("keep", "y__gcausal_lag1")])
    assert _drop_specs_whose_bases_the_suite_cannot_materialise(disc, (None, None, None), "y") == []
    assert [s.name for s in disc.specs_] == ["keep"]


def test_building_a_column_no_split_carries_warns_instead_of_returning_nan_quietly(caplog):
    """An all-NaN column becomes an all-NaN target downstream, so the builder names the column it could not find."""
    df, _, _ = _frames()
    idx = np.arange(3)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._misc_helpers"):
        out = _build_full_column_from_splits("y__gcausal_lag1", df, None, None, idx, None, None, n_total=3)
    assert np.all(np.isnan(out))
    assert [r for r in caplog.records if "y__gcausal_lag1" in r.getMessage()], "the missing column must be named in a warning"


def test_a_column_present_in_a_split_is_built_without_warning(caplog):
    """The normal path is unchanged: values land at their split's row positions and nothing is logged."""
    df, _, _ = _frames()
    idx = np.arange(3)
    with caplog.at_level(logging.WARNING, logger="mlframe.training.core._misc_helpers"):
        out = _build_full_column_from_splits("real_base", df, None, None, idx, None, None, n_total=3)
    np.testing.assert_allclose(out, [0.5, 0.6, 0.7])
    assert not [r for r in caplog.records if "real_base" in r.getMessage()]
