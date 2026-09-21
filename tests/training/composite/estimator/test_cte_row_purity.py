"""Row purity: a deployed composite's prediction for a row depends on that row, not on the batch or the thread it came in.

``test_predict_batching_invariance.py`` covers chunking with a real inner (to BLAS round-off), the warm-up contract of the
recurrent transforms and NaN-base locality. This module adds what that cannot: chunk invariance to 1e-14 relative with a
per-row inner, where cross-row coupling in the wrapper would show up orders of magnitude above that (the only differences
left are the 1-2 ulp of a transform's own BLAS dot, ``base @ alphas`` or ``weights @ knots``, whose reduction order follows
the batch); concurrent predicts on one fitted wrapper; and the ``lag_predict`` component the cross-target ensemble ships,
whose missing-lag fill came from the batch.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import NotFittedError

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform, list_transforms
from mlframe.training.core._phase_composite_post_lag_predict import _LagPredictDeployableModel

_POINTWISE = [n for n in list_transforms() if not get_transform(n).recurrent]


class _RowwiseInner(BaseEstimator, RegressorMixin):
    """Predicts a fixed elementwise function of one feature: no reduction of its own across rows."""

    def fit(self, X, y):
        """Remember the T mean and spread; the prediction is ``mean + 0.1 * spread * feat``."""
        t = np.asarray(y, dtype=np.float64)
        self.mean_, self.spread_ = float(np.mean(t)), float(np.std(t)) or 1.0
        return self

    def predict(self, X):
        """Elementwise in ``feat``."""
        return self.mean_ + 0.1 * self.spread_ * np.asarray(X["feat"], dtype=np.float64)


def _fitted(name: str):
    """A wrapper over the row-wise inner, fitted on 900 rows, and a 200-row continuation."""
    rng = np.random.default_rng(0)
    n = 1100
    base = np.linspace(1.0, 20.0, n) + rng.normal(0.0, 0.05, n)
    X = pd.DataFrame({"base": base, "base2": rng.uniform(1.0, 5.0, n), "feat": rng.normal(size=n), "grp": rng.integers(0, 3, n)})
    y = 0.7 * base + 0.3 * X["base2"].to_numpy() + 0.4 * X["feat"].to_numpy() + rng.normal(0.0, 0.1, n) + 2.0
    t = get_transform(name)
    kw: dict = {"base_column": "base"}
    if t.requires_groups:
        kw["group_column"] = "grp"
    if t.n_bases > 1:
        kw["base_columns"] = ("base", "base2")
    est = CompositeTargetEstimator(base_estimator=_RowwiseInner(), transform_name=name, **kw).fit(X.iloc[:900], y[:900])
    return est, X.iloc[900:].reset_index(drop=True)


@pytest.mark.parametrize("size", [1, 7, 50])
@pytest.mark.parametrize("name", _POINTWISE)
def test_a_pointwise_prediction_does_not_depend_on_the_batch(name: str, size: int):
    """With an elementwise inner, a pointwise transform's chunked prediction equals the whole-batch one to 1e-14 relative."""
    est, cont = _fitted(name)
    full = np.asarray(est.predict(cont))
    chunked = np.concatenate([np.asarray(est.predict(cont.iloc[i : i + size])) for i in range(0, len(cont), size)])
    np.testing.assert_allclose(chunked, full, rtol=1e-14, atol=0.0, err_msg=f"{name}: a row's prediction depends on its batch")


def test_concurrent_predicts_count_every_call_and_keep_their_own_shrink_flags():
    """8 threads x 150 predicts: no lost counter increment, and each thread reads the shrink flags of its own batch.

    The switch interval is cut to 1 us so the interpreter swaps threads inside the counter read-modify-write and between a
    predict and the read of its flags; at the default 5 ms the races exist but rarely fire.
    """
    import sys

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # restored in the finally below
    est, cont = _fitted("linear_residual")
    before = int(est.runtime_stats_.get("predict_calls", 0))
    errors: list[str] = []
    barrier = threading.Barrier(8)

    def _worker(k: int) -> None:
        batch = cont.iloc[: 10 + 5 * k]
        barrier.wait()
        for _ in range(150):
            est.predict(batch)
            info = est.soft_shrink_info_
            if info["n_rows"] != len(batch):
                errors.append(f"thread {k}: shrink info describes {info['n_rows']} rows, its batch has {len(batch)}")
                return

    threads = [threading.Thread(target=_worker, args=(k,)) for k in range(8)]
    try:
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    finally:
        sys.setswitchinterval(old)
    assert not errors, errors[:3]
    assert int(est.runtime_stats_["predict_calls"]) - before == 1200


def test_the_lag_component_fills_a_missing_lag_from_train_not_from_its_batch():
    """A NaN lag gets the train median, the same in any batch; before, it got the median of the batch being predicted."""
    train = pd.DataFrame({"lag": np.arange(100.0)})
    model = _LagPredictDeployableModel("lag").fit(train)
    batch_a = pd.DataFrame({"lag": [np.nan, 1000.0, 2000.0]})
    batch_b = pd.DataFrame({"lag": [np.nan, -5.0]})
    single = pd.DataFrame({"lag": [np.nan]})
    fills = [model.predict(b)[0] for b in (batch_a, batch_b, single)]
    assert fills == [float(np.median(np.arange(100.0)))] * 3


def test_an_unfitted_lag_component_refuses_to_invent_a_fill():
    """Unfitted, a missing lag raises instead of borrowing the batch median (a one-row NaN batch used to return 0.0)."""
    model = _LagPredictDeployableModel("lag")
    np.testing.assert_array_equal(model.predict(pd.DataFrame({"lag": [1.0, 2.0]})), [1.0, 2.0])
    with pytest.raises(NotFittedError):
        model.predict(pd.DataFrame({"lag": [np.nan]}))
