"""The RFECV ID-like warning distinguishes row-index / hash columns from legitimate count-like integers via monotonicity, range and unique fraction."""
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers.rfecv import RFECV
from mlframe.feature_selection.wrappers.rfecv._validate import _sanitize_X_inputs

_LOGGER = "mlframe.feature_selection.wrappers.rfecv"
_N = 400


def _msgs(caplog, col):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"c": col(rng), "x1": rng.normal(size=_N)})
    sel = RFECV(estimator=LogisticRegression(), cv=3, verbose=0, random_state=0, drop_id_like_sequences=False)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _sanitize_X_inputs(sel, df, rng.integers(0, 2, _N))
    return [r.getMessage() for r in caplog.records if "cardinality > 0.5*n" in r.getMessage()]


def _countlike(rng, frac):
    k = int(_N * frac)
    base = rng.permutation(np.arange(2 * _N))[:k]
    return np.concatenate([base, rng.choice(base, _N - k)])[rng.permutation(_N)]


def test_monotonic_autoincrement_int_warns_with_details(caplog):
    msgs = _msgs(caplog, lambda rng: np.arange(_N))
    assert len(msgs) == 1 and "monotonic" in msgs[0] and "nunique=400" in msgs[0] and "n=400" in msgs[0] and "int64" in msgs[0]


def test_random_large_range_int_hash_warns(caplog):
    msgs = _msgs(caplog, lambda rng: rng.choice(10**7, _N, replace=False))
    assert len(msgs) == 1 and "hash-like" in msgs[0]


def test_non_monotonic_countlike_int_with_unique_frac_0_6_does_not_warn(caplog):
    assert _msgs(caplog, lambda rng: _countlike(rng, 0.6)) == []


def test_non_monotonic_int_with_unique_frac_above_0_9_warns(caplog):
    msgs = _msgs(caplog, lambda rng: rng.permutation(_N))
    assert len(msgs) == 1 and "high-cardinality" in msgs[0]


def test_integer_valued_float_id_warns(caplog):
    assert len(_msgs(caplog, lambda rng: rng.choice(10**7, _N, replace=False).astype(float))) == 1


def test_continuous_float_does_not_warn(caplog):
    assert _msgs(caplog, lambda rng: rng.normal(size=_N)) == []
