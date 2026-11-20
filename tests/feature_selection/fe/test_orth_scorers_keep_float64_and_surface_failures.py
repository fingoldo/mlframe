"""dCor/HSIC/KSG/Lasso/ElasticNet score large-offset columns in double; HSIC zeroes non-finite scores loudly; Lasso surfaces non-convergence."""

import logging
import warnings

import numpy as np

from mlframe.feature_selection.filters import _orthogonal_dcor_fe as dcor
from mlframe.feature_selection.filters import _orthogonal_elasticnet_fe as enet
from mlframe.feature_selection.filters import _orthogonal_hsic_fe as hsic
from mlframe.feature_selection.filters import _orthogonal_ksg_mi_fe as ksg
from mlframe.feature_selection.filters import _orthogonal_lasso_fe as lasso


def _offset_problem(n: int = 400):
    """x = 1e6 + 0.01 * s with y = s + noise: float32 collapses x to a constant (resolution 0.0625 at 1e6)."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=n)
    x = 1e6 + 0.01 * s
    y = s + 0.1 * rng.normal(size=n)
    return x.reshape(-1, 1), y


def test_dcor_batch_scores_large_offset_column_in_double():
    """A column whose signal sits below float32 resolution at its offset still gets a high dCor."""
    X, y = _offset_problem()
    assert dcor._dcor_batch(X, y)[0] > 0.5


def test_hsic_batch_scores_large_offset_column_in_double():
    """Same offset column: HSIC must see the signal."""
    X, y = _offset_problem()
    assert hsic._hsic_batch(X, y)[0] > 1e-3


def test_ksg_batch_scores_large_offset_column_in_double():
    """Same offset column against a binary target: KSG MI must be clearly positive."""
    X, y = _offset_problem()
    assert ksg._ksg_mi_batch(X, (y > 0).astype(int))[0] > 0.2


def test_lasso_coefs_see_large_offset_column_in_double():
    """Standardising in float32 turns the column into a constant (coef 0); in double it carries the signal."""
    X, y = _offset_problem()
    assert lasso._fit_lasso_abs_coefs(X, y, alpha=0.01, standardize=True)[0] > 0.3


def test_elasticnet_coefs_see_large_offset_column_in_double():
    """Same for ElasticNet."""
    X, y = _offset_problem()
    assert enet._fit_elasticnet_abs_coefs(X, y, alpha=0.01, l1_ratio=0.5, standardize=True)[0] > 0.3


def test_hsic_batch_zeroes_nonfinite_column_scores_and_warns(caplog):
    """A NaN column must score 0.0 (not NaN, which would turn every floor into NaN) and log a WARNING."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    X[5, 0] = np.nan
    y = X[:, 1] + 0.1 * rng.normal(size=200)
    with caplog.at_level(logging.WARNING, logger=hsic.logger.name):
        out = hsic._hsic_batch(X, y)
    assert np.isfinite(out).all() and out[0] == 0.0 and out[1] > 0.0
    assert any("non-finite" in r.getMessage() for r in caplog.records)


def test_lasso_nonconvergence_logs_warning(monkeypatch, caplog):
    """A ConvergenceWarning from the solver must reach the log instead of being silenced."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso

    real_fit = Lasso.fit

    def noisy_fit(self, X, y, *a, **k):
        """Fit, then emit the solver's ConvergenceWarning."""
        warnings.warn("no converge", ConvergenceWarning, stacklevel=2)
        return real_fit(self, X, y, *a, **k)

    monkeypatch.setattr(Lasso, "fit", noisy_fit)
    X, y = _offset_problem()
    with caplog.at_level(logging.WARNING, logger=lasso.logger.name):
        lasso._fit_lasso_abs_coefs(X, y, alpha=0.01, standardize=True)
    assert any("did not converge" in r.getMessage() for r in caplog.records)
