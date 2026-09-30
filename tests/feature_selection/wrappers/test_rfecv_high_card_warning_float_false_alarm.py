"""RFECV's "cardinality > 0.5*n looks like ID / hash" warning must not fire for genuinely continuous float columns."""
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.wrappers.rfecv import RFECV
from mlframe.feature_selection.wrappers.rfecv._validate import _sanitize_X_inputs

_LOGGER = "mlframe.feature_selection.wrappers.rfecv"


def _warned_columns(caplog, df, y):
    sel = RFECV(estimator=LogisticRegression(), cv=3, verbose=0, random_state=0)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        _sanitize_X_inputs(sel, df, y)
    return [r.getMessage() for r in caplog.records if "cardinality > 0.5*n" in r.getMessage()]


def _frame(col):
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({"c": col(rng, n), "x1": rng.normal(size=n)}), rng.integers(0, 2, n)


def test_continuous_float_column_does_not_trigger_id_warning(caplog):
    df, y = _frame(lambda rng, n: rng.normal(size=(n, 5)).mean(axis=1))
    assert _warned_columns(caplog, df, y) == []


def test_high_card_int_id_column_triggers_warning(caplog):
    df, y = _frame(lambda rng, n: rng.choice(10**6, n, replace=False))
    msgs = _warned_columns(caplog, df, y)
    assert len(msgs) == 1 and "'c'" in msgs[0]


def test_integer_valued_float_id_column_triggers_warning(caplog):
    df, y = _frame(lambda rng, n: rng.choice(10**6, n, replace=False).astype(float))
    msgs = _warned_columns(caplog, df, y)
    assert len(msgs) == 1 and "'c'" in msgs[0]
