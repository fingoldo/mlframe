"""The linear screening family must factorise each fold once and solve per spec, matching a float64 Ridge.

``SimpleImputer(mean) + Ridge(alpha=1)`` was refit for every spec although only the target differs, re-imputing the
fold and re-factorising the same regularised Gram matrix each time: 1836 ms for 32 targets on 13.3k x 60, against 83 ms
with one factorisation. The solve runs in float64, so it agrees with a float64 sklearn fit rather than with the float32
arithmetic the screening block's dtype used to force.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from mlframe.training.composite.discovery import _ridge_shared_fold
from mlframe.training.composite.discovery._ridge_shared_fold import fit_ridge_on_shared_fold
from mlframe.training.composite.discovery._screening_tiny_perbin import _tiny_cv_rmse_y_scale
from mlframe.training.composite.transforms import get_transform


def _matrix(n: int = 2000, f: int = 12, nan_frac: float = 0.02, seed: int = 0) -> np.ndarray:
    """A float32 feature block with scattered missing cells, like the screening matrix."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, f)).astype(np.float32)
    x[rng.random(x.shape) < nan_frac] = np.nan
    return x


def _sklearn_float64(x_fit, t, x_eval):
    """The linear family exactly as built today, fed float64 so the comparison is about the method, not the dtype."""
    model = Pipeline([("imp", SimpleImputer(strategy="mean")), ("ridge", Ridge(alpha=1.0))])
    return model.fit(x_fit.astype(np.float64), t).predict(x_eval.astype(np.float64))


def test_predictions_match_a_float64_ridge_pipeline():
    """Same imputation, same penalty, same intercept handling: predictions agree to float64 round-off."""
    x = _matrix()
    rng = np.random.default_rng(1)
    rows = np.arange(1500)
    _ridge_shared_fold._CACHE.clear()
    for k in (0.5, 2.0, -3.0):
        t = k * np.nan_to_num(x[rows, 0]) + rng.normal(size=rows.size)
        ours = fit_ridge_on_shared_fold(x, rows, t).predict(x[1500:])
        ref = _sklearn_float64(x[rows], t, x[1500:])
        np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-7)


def test_an_all_missing_column_is_handled_like_the_imputer():
    """SimpleImputer drops a column with no observed value; filling it with 0 must give the same predictions."""
    x = _matrix()
    x[:, 3] = np.nan
    rows = np.arange(1500)
    t = np.nan_to_num(x[rows, 0]) + 0.1
    _ridge_shared_fold._CACHE.clear()
    ours = fit_ridge_on_shared_fold(x, rows, t).predict(x[1500:])
    np.testing.assert_allclose(ours, _sklearn_float64(x[rows], t, x[1500:]), rtol=0, atol=1e-7)


def test_each_fold_is_factorised_once_across_specs(monkeypatch):
    """Three specs over three folds need three factorisations, not nine."""
    import scipy.linalg

    calls = {"n": 0}
    real = scipy.linalg.cho_factor

    def counting(*args, **kwargs):
        """Count every factorisation."""
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "cho_factor", counting)
    rng = np.random.default_rng(2)
    base = rng.uniform(10.0, 50.0, 1500)
    x = rng.normal(size=(1500, 6))
    y = 1.5 * base + 3.0 * x[:, 0] + rng.normal(size=1500)
    _ridge_shared_fold._CACHE.clear()
    for name, params in (("linear_residual", {"alpha": 1.5, "beta": 0.0}), ("diff", {}), ("additive_residual", {})):
        _tiny_cv_rmse_y_scale(
            y, base, get_transform(name), params, x,
            family="linear", n_estimators=10, num_leaves=7, learning_rate=0.1, cv_folds=3, random_state=0, n_jobs=1,
        )
    assert calls["n"] == 3, f"expected one factorisation per fold across specs, got {calls['n']}"


def test_the_linear_family_scores_match_the_float64_sklearn_path(monkeypatch):
    """A spec's CV-RMSE through the shared factorisation equals the one a float64 sklearn pipeline gives it."""
    from mlframe.training.composite.discovery import _screening_tiny_perbin as perbin

    rng = np.random.default_rng(3)
    base = rng.uniform(10.0, 50.0, 1500)
    x = rng.normal(size=(1500, 6)).astype(np.float32)
    y = 1.5 * base + 3.0 * x[:, 0] + rng.normal(size=1500)

    def score():
        """The diff spec's linear-family CV-RMSE."""
        return float(_tiny_cv_rmse_y_scale(
            y, base, get_transform("diff"), {}, x,
            family="linear", n_estimators=10, num_leaves=7, learning_rate=0.1, cv_folds=3, random_state=0, n_jobs=1,
        ))

    _ridge_shared_fold._CACHE.clear()
    shared = score()

    def sklearn_fit(x_all, rows, target):
        """The per-spec float64 pipeline, standing in for the shared factorisation."""
        model = Pipeline([("imp", SimpleImputer(strategy="mean")), ("ridge", Ridge(alpha=1.0))])
        fitted = model.fit(np.asarray(x_all[rows], dtype=np.float64), target)

        class _AsFloat64:
            """Predict through the pipeline on a float64 view, as the shared model does."""

            def predict(self, x_eval):
                """Delegate to the fitted pipeline."""
                return fitted.predict(np.asarray(x_eval, dtype=np.float64))

        return _AsFloat64()

    monkeypatch.setattr(perbin, "fit_ridge_on_shared_fold", sklearn_fit)
    assert shared == pytest.approx(score(), rel=1e-9)
