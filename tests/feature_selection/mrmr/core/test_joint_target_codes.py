"""A multi-output y reaches the FE stages as one joint class per row (it was raveled into n * k labels against n rows)."""

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._y_encoding import joint_target_codes


def test_joint_codes_one_class_per_distinct_combination():
    y = np.array([[0, 1], [0, 1], [1, 0], [1, 1], [0, 0], [1, 0]])
    c = joint_target_codes(y)
    assert c.shape == (6,) and c.dtype == np.int64
    assert c[0] == c[1] and c[2] == c[5] and len({c[0], c[2], c[3], c[4]}) == 4


def test_joint_codes_continuous_column_is_binned_not_singletons():
    rng = np.random.default_rng(0)
    y = np.column_stack([rng.integers(0, 2, 2000), rng.standard_normal(2000)])
    assert len(np.unique(joint_target_codes(y))) < 200


def test_single_column_unchanged():
    y = np.array([3, 1, 3, 2, 1])
    np.testing.assert_array_equal(joint_target_codes(y), joint_target_codes(y.reshape(-1, 1)))


def test_joint_multioutput_fit_keeps_triplet_fe_alive(caplog):
    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(0)
    n = 160
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(6)})
    y = np.column_stack([(X["f0"] > 0).astype(int), (X["f1"] > 0).astype(int)])
    with caplog.at_level(logging.WARNING):
        MRMR(verbose=0, random_state=0, n_jobs=1, fit_cache_max=0, multioutput_strategy="joint", fe_hybrid_orth_triplet_enable=True).fit(X, y)
    assert not [r for r in caplog.records if "boolean index did not match" in r.getMessage()]
