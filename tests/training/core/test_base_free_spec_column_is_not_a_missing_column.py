"""A unary spec's empty base column is not a missing column.

A production run logged "column '' is in none of the train/val/test frames; returning all-NaN. A composite target built
on it cannot be trained." right after composite discovery shipped a ``yqclip`` spec, which is base-free by construction
and trained fine. The warning is true of a base a spec needs and does not have, so it must not fire for a spec that
never had one -- otherwise the real case is lost among the false ones.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.training.core._misc_helpers import _build_full_column_from_splits


def _frames():
    """Two split frames carrying one real column, and their row indices in the full row space."""
    train = pd.DataFrame({"budget_amount": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"budget_amount": [4.0, 5.0]})
    return train, test, np.arange(3), np.arange(3, 5)


def test_base_free_sentinel_returns_all_nan_without_warning(caplog):
    """The base-free sentinel is the unary spec's own contract: all-NaN, quietly."""
    train, test, tr_idx, te_idx = _frames()
    with caplog.at_level(logging.WARNING):
        out = _build_full_column_from_splits("", train, None, test, tr_idx, None, te_idx, n_total=5)
    assert out.shape == (5,) and np.isnan(out).all()
    assert not [r for r in caplog.records if "is in none of the train/val/test frames" in r.getMessage()]


def test_a_genuinely_missing_base_column_still_warns(caplog):
    """The warning must survive for the case it was written for."""
    from mlframe.utils.log_throttle import reset_throttle_counts

    reset_throttle_counts("build_full_column_missing_everywhere")
    train, test, tr_idx, te_idx = _frames()
    with caplog.at_level(logging.WARNING):
        out = _build_full_column_from_splits("no_such_column", train, None, test, tr_idx, None, te_idx, n_total=5)
    assert np.isnan(out).all()
    assert [r for r in caplog.records if "is in none of the train/val/test frames" in r.getMessage()]


def test_a_present_column_is_still_assembled():
    """The ordinary path is untouched."""
    train, test, tr_idx, te_idx = _frames()
    out = _build_full_column_from_splits("budget_amount", train, None, test, tr_idx, None, te_idx, n_total=5)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_unary_transforms_accept_the_all_nan_base():
    """Why the placeholder is safe: a base-free transform ignores the base it is handed."""
    from mlframe.training.composite.transforms import get_transform

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    base = np.full(5, np.nan)
    for name in ("y_quantile_clip", "log_y", "cbrt_y"):
        transform = get_transform(name)
        assert transform.requires_base is False
        assert np.asarray(transform.domain_check(y, base)).all()
