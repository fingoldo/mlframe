"""transform()'s identity fast-path must prove the columns are the SAME ones (mrmr_audit_2026-09-14 PERIPHERY-1).

The fast-path returns the caller's frame unchanged when the selected-count equals ``X.shape[1]``. The bare
width check above it is deliberately skipped for named frames so the actionable column-name ``RuntimeError``
further down can fire -- but the fast-path returned before that check ran, so a same-width frame with
different or merely reordered columns came back unchanged with every column silently mis-mapped.

``transform`` is a plain function bound onto MRMR, so these exercise the branch directly with a fitted-shaped
stub rather than paying a full fit (which engineers recipes and never reaches the fast-path at all).
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_validate_transform import _identity_fastpath_is_safe, transform


class _FittedStub:
    """The minimal fitted surface ``transform`` reads on the identity fast-path."""

    def __init__(self, names):
        self.feature_names_in_ = np.asarray(list(names), dtype=object)
        self.n_features_in_ = len(names)
        self.support_ = np.arange(len(names))  # every input column selected
        self._engineered_recipes_ = []  # no recipes -> fast-path is eligible

    def _append_engineered(self, base_out, X, recipes):
        """With no recipes the real implementation is a passthrough; reached only off the fast-path."""
        assert not recipes
        return base_out


@pytest.fixture
def frame():
    """A three-column numeric frame matching the stub's fitted column set."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=64) for c in ("a", "b", "c")})


def test_identity_fastpath_returns_the_frame_when_columns_match_exactly(frame):
    """The optimisation itself must survive: same names, same order -> the very same object back."""
    out = transform(_FittedStub(["a", "b", "c"]), frame)
    assert out is frame


def test_renamed_columns_do_not_slip_through_the_identity_fastpath(frame):
    """Same width, one column renamed: must NOT be handed back as if it were the fitted frame."""
    renamed = frame.rename(columns={"a": "zzz"})
    with pytest.raises(Exception) as excinfo:
        transform(_FittedStub(["a", "b", "c"]), renamed)
    assert "zzz" not in str(excinfo.value) or "a" in str(excinfo.value)


def test_reordered_columns_do_not_slip_through_the_identity_fastpath(frame):
    """Same names, different order: returning X unchanged would disagree with get_feature_names_out()."""
    reordered = frame[["c", "b", "a"]]
    out = transform(_FittedStub(["a", "b", "c"]), reordered)
    assert out is not reordered, "the fast-path returned a reordered frame unchanged"
    assert list(out.columns) == ["a", "b", "c"]


def test_safety_probe_rejects_a_reorder_and_accepts_an_exact_match(frame):
    """Unit-level teeth for the predicate the fast-path now consults."""
    stub = _FittedStub(["a", "b", "c"])
    assert _identity_fastpath_is_safe(stub, frame) is True
    assert _identity_fastpath_is_safe(stub, frame[["c", "b", "a"]]) is False
    assert _identity_fastpath_is_safe(stub, frame.rename(columns={"a": "zzz"})) is False


def test_an_unnamed_ndarray_still_takes_the_fastpath(frame):
    """ndarrays carry no names and their width was already validated upstream -- keep them eligible."""
    arr = frame.to_numpy()
    out = transform(_FittedStub(["a", "b", "c"]), arr)
    assert out is arr


def test_polars_reorder_is_rejected_too(frame):
    """The named-frame guard must cover polars, not just pandas."""
    pl = pytest.importorskip("polars")
    pf = pl.from_pandas(frame)
    stub = _FittedStub(["a", "b", "c"])
    assert _identity_fastpath_is_safe(stub, pf) is True
    assert _identity_fastpath_is_safe(stub, pf.select(["c", "b", "a"])) is False
