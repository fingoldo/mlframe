"""Predictions must not depend on how the rows are batched, beyond what a recurrence inherently needs.

No test compared ``predict(X)[i]`` with the same row predicted in a smaller batch, so the batch-state defects - a
recurrence cold-starting at every batch, a NaN base corrupting its neighbours - were invisible. Measured with the
estimator on a 200-row continuation:

* pointwise transforms: chunked and one-batch predictions agree to BLAS round-off (the inner model's matrix product
  reduces in a different order per batch size, about 1e-14), so they are compared to 1e-12 relative, not bit-for-bit;
* recurrent transforms cold-start per batch (EWMA 8.4, volatility-normalized 26 y-units off), and become exact once
  the caller prepends a warm-up prefix of the preceding rows - the serving contract for a recurrence;
* ``frac_diff`` has power-law memory, so a bounded prefix only approaches the one-batch answer, and the centred
  rolling quantile reads rows AFTER the one predicted, so its last rows in a chunk differ whatever the prefix: both
  are pinned below as known limits, not as contracts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from mlframe.training.composite import CompositeTargetEstimator
from mlframe.training.composite.transforms import get_transform, list_transforms

_POINTWISE = [n for n in list_transforms() if not get_transform(n).recurrent]
_RECURRENT = [n for n in list_transforms() if get_transform(n).recurrent]
# Power-law memory (frac_diff) and a centred, look-ahead window: a bounded prefix cannot make these exact.
_UNBOUNDED_MEMORY = {"frac_diff", "frac_diff_grouped", "rolling_quantile_ratio_centered"}


def _frame(n: int = 1400, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """A positive, time-ordered frame with a second base and a group key."""
    rng = np.random.default_rng(seed)
    base = np.linspace(1.0, 20.0, n) + rng.normal(0.0, 0.05, n)
    base2 = rng.uniform(1.0, 5.0, n)
    feat = rng.normal(size=n)
    grp = rng.integers(0, 3, n)
    y = 0.7 * base + 0.3 * base2 + 0.4 * feat + rng.normal(0.0, 0.1, n) + 2.0
    return pd.DataFrame({"base": base, "base2": base2, "feat": feat, "grp": grp}), y


def _fitted(name: str, inner=None):
    """A wrapper fitted on the first 900 rows, and the continuation it predicts."""
    X, y = _frame()
    kw: dict = {"base_column": "base"}
    if get_transform(name).requires_groups:
        kw["group_column"] = "grp"
    if "multi" in name:
        kw["base_columns"] = ("base", "base2")
    est = CompositeTargetEstimator(base_estimator=inner if inner is not None else LinearRegression(), transform_name=name, **kw)
    est.fit(X.iloc[:900], y[:900])
    return est, X.iloc[900:].reset_index(drop=True)


@pytest.mark.parametrize("size", [1, 7, 50])
@pytest.mark.parametrize("name", _POINTWISE)
def test_pointwise_predictions_do_not_depend_on_the_batch(name: str, size: int):
    """Row i predicted in a chunk of ``size`` equals row i predicted in the whole continuation (to BLAS round-off)."""
    est, cont = _fitted(name)
    cont = cont.iloc[:200]
    full = np.asarray(est.predict(cont))
    chunked = np.concatenate([np.asarray(est.predict(cont.iloc[i : i + size])) for i in range(0, len(cont), size)])
    np.testing.assert_allclose(chunked, full, rtol=0, atol=1e-12 * max(1.0, float(np.max(np.abs(full)))))


@pytest.mark.parametrize("name", [n for n in _RECURRENT if n not in _UNBOUNDED_MEMORY])
def test_a_recurrent_prediction_is_exact_given_a_warm_up_prefix(name: str):
    """Each 20-row chunk, predicted with the 300 preceding rows prepended, equals the one-batch prediction of those rows."""
    est, cont = _fitted(name)
    full = np.asarray(est.predict(cont))
    for start in range(300, 480, 20):
        window = np.asarray(est.predict(cont.iloc[start - 300 : start + 20]))[-20:]
        np.testing.assert_allclose(window, full[start : start + 20], rtol=0, atol=1e-6)


@pytest.mark.parametrize("name", sorted(_UNBOUNDED_MEMORY & set(_RECURRENT)))
def test_known_limit_unbounded_memory_recurrences_only_approach_the_one_batch_answer(name: str):
    """Known limit, not a contract: a longer prefix brings the chunk closer, but never all the way.

    ``frac_diff`` weights decay as a power law, so any bounded prefix truncates its memory; the centred rolling quantile
    uses rows after the one being predicted, which a chunk ending at that row does not have.
    """
    est, cont = _fitted(name)
    full = np.asarray(est.predict(cont))

    def worst(prefix: int) -> float:
        """Largest chunk-vs-batch deviation over a few chunks with this much prefix."""
        return max(
            float(np.max(np.abs(np.asarray(est.predict(cont.iloc[s - prefix : s + 20]))[-20:] - full[s : s + 20])))
            for s in range(300, 480, 60)
        )

    assert worst(300) <= worst(0) + 1e-12, f"{name}: a longer warm-up must not make the chunk worse"


@pytest.mark.parametrize("name", _RECURRENT)
def test_a_nan_base_in_a_recurrent_batch_stays_contained(name: str):
    """One missing base must never turn into NaN predictions, and a causal recurrence must leave earlier rows alone."""
    est, cont = _fitted(name, inner=make_pipeline(SimpleImputer(), LinearRegression()))
    cont = cont.iloc[:200].copy()
    clean = np.asarray(est.predict(cont))
    with_nan = cont.copy()
    with_nan.loc[100, "base"] = np.nan
    pred = np.asarray(est.predict(with_nan))
    assert np.all(np.isfinite(pred)), f"{name}: a NaN base produced {int((~np.isfinite(pred)).sum())} non-finite predictions"
    if name != "rolling_quantile_ratio_centered":  # a centred window reads the rows after it, by construction
        np.testing.assert_allclose(pred[:100], clean[:100], rtol=0, atol=1e-12, err_msg=f"{name}: a NaN base changed earlier rows")
