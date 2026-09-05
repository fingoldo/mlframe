"""The residual-band transformer cluster must band rows on out-of-fold, not in-sample, predictions.

`_fit_baseline_predict` was copy-pasted into ten modules of this package under the same name. Some were
later fixed to return inner-KFold(3) out-of-fold predictions and the rest were not, because the fix was
propagated by hand. An in-sample prediction is close to `y_t` almost by construction -- the model was just
fit on these exact rows -- so the residual it implies understates the true one, and every band, quantile
and top-K-hardest selection derived from it is drawn on a distorted signal.

Measured on n=400, p=6, y = x0 + 0.5*x1 + 0.3*noise with LGBM(50, depth 3): mean |residual| is 0.2092
in-sample against 0.2968 out-of-fold, and 244 of 400 rows land in a different quintile band.

The five modules that shared a signature now call one shared implementation, so the tests below check both
that each is honest and that they agree exactly -- the second is what stops the cluster drifting apart
again.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("sklearn")

# The five modules whose `_fit_baseline_predict` shares one signature and now one body.
SHARED = [
    "bidir_residual_band",
    "hard_row_attention",
    "multi_temp_residual_band",
    "prediction_band_attention",
    "signed_residual_band",
]


def _baseline_of(module_name: str):
    """Return one module's `_fit_baseline_predict`."""
    mod = importlib.import_module(f"mlframe.feature_engineering.transformer.{module_name}")
    return mod._fit_baseline_predict


def _pure_noise(n: int = 300, p: int = 8, seed: int = 0):
    """A target independent of the features, where any residual shrinkage IS memorisation."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng.normal(size=n)


def _in_sample_residual(X, y):
    """Mean |residual| of a single fit-and-predict-on-the-same-rows baseline."""
    import lightgbm as lgb

    m = lgb.LGBMRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=0, verbose=-1, n_jobs=-1)
    m.fit(X, y)
    return float(np.mean(np.abs(y - np.asarray(m.predict(X)))))


@pytest.mark.parametrize("module_name", SHARED)
def test_the_baseline_does_not_memorise_its_own_rows(module_name: str):
    """On a target independent of X, an honest baseline cannot beat predicting nothing; an in-sample one does."""
    X, y = _pure_noise()
    oof = _baseline_of(module_name)(X, y, task="regression", seed=0)
    oof_residual = float(np.mean(np.abs(y - oof)))
    in_sample_residual = _in_sample_residual(X, y)
    assert oof_residual > in_sample_residual * 1.15, (
        f"{module_name}: mean |residual| {oof_residual:.4f} against an in-sample {in_sample_residual:.4f} -- "
        "the baseline is fitting and predicting on the same rows"
    )


@pytest.mark.parametrize("module_name", SHARED[1:])
def test_every_shared_copy_returns_exactly_the_same_predictions(module_name: str):
    """Equality, not similarity: these call one implementation, and drift between them is the bug class."""
    X, y = _pure_noise(seed=3)
    reference = _baseline_of(SHARED[0])(X, y, task="regression", seed=5)
    got = _baseline_of(module_name)(X, y, task="regression", seed=5)
    assert np.array_equal(got, reference), f"{module_name} no longer agrees with {SHARED[0]}"


def test_the_quintile_knn_baseline_is_also_out_of_fold():
    """`y_quintile_baseline_knn` took a separate Xall to predict on, but always passed the train matrix.

    It is checked apart from the five above because it runs a deeper baseline (100 iterations, depth 5), so
    its predictions are its own -- what must hold is only that they are not fitted on the rows they score.
    """
    from mlframe.feature_engineering.transformer.y_quintile_baseline_knn import _fit_baseline_predict

    X, y = _pure_noise()
    oof_residual = float(np.mean(np.abs(y - _fit_baseline_predict(X, y, task="regression", seed=0))))
    assert oof_residual > _in_sample_residual(X, y) * 1.15, f"mean |residual| {oof_residual:.4f} still looks fitted on its own rows"


def test_the_binary_task_returns_positive_class_probabilities():
    """The binary branch must come back in [0, 1], not as raw margins."""
    from mlframe.feature_engineering.transformer._baseline_oof import fit_baseline_predict_oof

    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + 0.4 * rng.normal(size=300) > 0).astype(np.int32)
    out = fit_baseline_predict_oof(X, y, "binary", 0)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_a_signal_bearing_target_is_still_predicted_well():
    """Guards the fix itself: honest predictions must not be honest-and-useless."""
    from mlframe.feature_engineering.transformer._baseline_oof import fit_baseline_predict_oof

    rng = np.random.default_rng(2)
    X = rng.normal(size=(400, 6))
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.normal(size=400)
    oof = fit_baseline_predict_oof(X, y, "regression", 0)
    assert float(np.corrcoef(oof, y)[0, 1]) > 0.9, "the out-of-fold baseline lost the signal it is supposed to model"


def test_too_few_rows_fall_back_rather_than_raising():
    """Below three rows there is no honest split; the caller must still get an array back."""
    from mlframe.feature_engineering.transformer._baseline_oof import fit_baseline_predict_oof

    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = np.array([0.5, 1.5])
    out = fit_baseline_predict_oof(X, y, "regression", 0)
    assert out.shape == (2,)
