"""CPX36 identity regression: batched-stack predict must be BIT-IDENTICAL to per-perturbation predict.

Three FE transformers (counterfactual_substitution, adversarial_flip, fisher_weighted_residual)
replaced a per-feature loop of full LightGBM predict calls with ONE predict over a vertically-
stacked perturbation matrix. For tree models predict is per-row independent, so the stacked
predict equals the per-perturbation predict EXACTLY. These tests pin that identity against the
frozen pre-optimization baseline snapshot under _benchmarks/_cpx36_baseline/ — they FAIL if the
current (batched) code diverges by even one ULP from the original per-perturbation code.
"""

from __future__ import annotations

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from mlframe.feature_engineering.transformer.counterfactual_substitution import (
    compute_counterfactual_substitution_features as cfact_new,
)
from mlframe.feature_engineering.transformer.adversarial_flip import (
    compute_adversarial_flip_features as advflip_new,
)
from mlframe.feature_engineering.transformer.fisher_weighted_residual import (
    compute_fisher_weighted_residual_features as fisher_new,
)
from mlframe.feature_engineering._benchmarks._cpx36_baseline.counterfactual_substitution_old import (
    compute_counterfactual_substitution_features as cfact_old,
)
from mlframe.feature_engineering._benchmarks._cpx36_baseline.adversarial_flip_old import (
    compute_adversarial_flip_features as advflip_old,
)
from mlframe.feature_engineering._benchmarks._cpx36_baseline.fisher_weighted_residual_old import (
    compute_fisher_weighted_residual_features as fisher_old,
)


def _data(task: str, n_train=600, n_query=120, d=12, seed=0):
    rng = np.random.default_rng(seed)
    Xt = rng.standard_normal((n_train, d)).astype(np.float32)
    Xq = rng.standard_normal((n_query, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    yc = Xt @ w + 0.1 * rng.standard_normal(n_train).astype(np.float32)
    y = (yc > np.median(yc)).astype(np.float32) if task == "binary" else yc
    return Xt, y, Xq


@pytest.mark.parametrize("task", ["regression", "binary"])
@pytest.mark.parametrize(
    "old_fn,new_fn",
    [(cfact_old, cfact_new), (advflip_old, advflip_new), (fisher_old, fisher_new)],
    ids=["counterfactual", "adversarial_flip", "fisher"],
)
def test_batched_predict_bit_identical(old_fn, new_fn, task):
    Xt, y, Xq = _data(task)
    df_old = old_fn(Xt, y, Xq, seed=7, task=task)
    df_new = new_fn(Xt, y, Xq, seed=7, task=task)
    assert df_old.columns == df_new.columns
    a = df_old.to_numpy()
    b = df_new.to_numpy()
    assert a.shape == b.shape
    assert np.array_equal(a, b), f"batched predict diverged from per-perturbation: max|d|={np.max(np.abs(a - b)):.3e}"
