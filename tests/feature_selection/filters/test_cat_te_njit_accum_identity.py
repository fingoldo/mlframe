"""_compute_target_encoding njit accumulator is bit-identical to the per-row loop; the real target-encoding path
runs + is deterministic through an MRMR fit."""

import numpy as np
import pandas as pd
from mlframe.feature_selection.filters._cat_target_encoding_and_weighted import _cell_sum_cnt_njit


def test_cell_sum_cnt_njit_bit_identical():
    rng = np.random.default_rng(0)
    n, nu = 40000, 300
    classes = rng.integers(0, nu, n).astype(np.int64)
    y = rng.standard_normal(n)
    cs = np.zeros(nu)
    cc = np.zeros(nu)
    for row in range(n):
        c = int(classes[row])
        cs[c] += y[row]
        cc[c] += 1.0
    gs, gc = _cell_sum_cnt_njit(classes, y, nu)
    assert np.array_equal(gs, cs) and np.array_equal(gc, cc)


def test_kfold_te_fit_runs_and_deterministic():
    from mlframe.feature_selection.filters import MRMR

    rng = np.random.default_rng(1)
    n = 6000
    X = pd.DataFrame({f"cat{i}": rng.integers(0, 15, n).astype(float) for i in range(3)})
    X["num"] = rng.standard_normal(n)
    y = ((X["cat0"] == X["cat1"]) * 0.8 + X["num"] * 0.4 + rng.standard_normal(n) * 0.3 > 0.4).astype(int).to_numpy()

    def fit():
        MRMR._FIT_CACHE.clear()
        m = MRMR(
            fe_ntop_features=4,
            n_jobs=1,
            verbose=0,
            random_seed=3,
            fe_kfold_te_enable=True,
            fe_kfold_te_cols=["cat0", "cat1", "cat2"],
            skip_retraining_on_same_content=False,
        )
        m.fit(X, y)
        return tuple(np.asarray(m.support_).ravel().tolist())

    a = fit()
    b = fit()
    assert a == b
