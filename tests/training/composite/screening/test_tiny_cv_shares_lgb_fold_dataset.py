"""The y-scale tiny CV must bin each fold once and share it across specs, with the scores unchanged.

Every spec on one base trains its tiny LightGBM model on the same fold rows; only the label differs. The sklearn
wrapper re-binned those rows on every fit - 328 ms of a roughly 0.9 s fit at the tiny-model defaults. The fold dataset
is now built once and its label swapped, which gives bit-identical predictions and 1.32x on the fits.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery import _lgb_shared_fold
from mlframe.training.composite.discovery import _screening_tiny_perbin as perbin
from mlframe.training.composite.discovery._screening_tiny_perbin import _tiny_cv_rmse_y_scale
from mlframe.training.composite.transforms import get_transform

pytest.importorskip("lightgbm")


def _data(n: int = 1500, seed: int = 0):
    """A base-driven target with a few extra features."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(10.0, 50.0, n)
    x = rng.normal(size=(n, 6))
    y = 1.5 * base + 3.0 * x[:, 0] + rng.normal(size=n)
    return y, base, x


def _score(y, base, x, transform_name: str, params: dict) -> float:
    """One spec's y-scale CV-RMSE at small tiny-model settings."""
    return float(_tiny_cv_rmse_y_scale(
        y, base, get_transform(transform_name), params, x,
        family="lightgbm", n_estimators=30, num_leaves=15, learning_rate=0.1, cv_folds=3, random_state=0, n_jobs=1,
    ))


_SPECS = [("linear_residual", {"alpha": 1.5, "beta": 0.0}), ("diff", {}), ("additive_residual", {})]


def test_scores_are_bit_identical_to_the_per_fit_path(monkeypatch):
    """Sharing the binned fold is a speed change only: every spec's score must equal the per-fit one exactly."""
    y, base, x = _data()
    _lgb_shared_fold._CACHE.clear()
    shared = [_score(y, base, x, name, dict(p)) for name, p in _SPECS]

    import lightgbm as lgb

    def sklearn_fit(x_all, rows, target, *, params, n_estimators):
        """The per-fit reference: the sklearn wrapper exactly as ``_build_tiny_model('lgb', ...)`` builds it."""
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, num_leaves=params["num_leaves"], learning_rate=params["learning_rate"],
            random_state=params["seed"], n_jobs=params["num_threads"], verbose=-1, force_col_wise=True,
        )
        return model.fit(x_all[rows], target)

    monkeypatch.setattr(perbin, "fit_on_shared_fold", sklearn_fit)
    per_fit = [_score(y, base, x, name, dict(p)) for name, p in _SPECS]
    assert shared == per_fit, f"shared-fold scores {shared} differ from per-fit scores {per_fit}"


def test_each_fold_is_binned_once_across_specs(monkeypatch):
    """Three specs over three folds need three dataset builds, not nine."""
    y, base, x = _data()
    _lgb_shared_fold._CACHE.clear()
    builds = {"n": 0}
    import lightgbm as lgb

    real_dataset = lgb.Dataset

    def counting_dataset(*args, **kwargs):
        """Count every dataset the shared path builds."""
        builds["n"] += 1
        return real_dataset(*args, **kwargs)

    monkeypatch.setattr(lgb, "Dataset", counting_dataset)
    for name, p in _SPECS:
        _score(y, base, x, name, dict(p))
    assert builds["n"] == 3, f"expected one dataset per fold across specs, got {builds['n']}"


def test_a_freed_matrix_never_hits_a_stale_dataset():
    """A cache keyed on id() alone could serve a dead matrix's dataset to a new one at the same address."""
    y, base, x = _data()
    _lgb_shared_fold._CACHE.clear()
    first = _score(y, base, x.copy(), "diff", {})
    other = x.copy()
    other[:, 0] = -other[:, 0]  # a different matrix with the same shape
    second = _score(y, base, other, "diff", {})
    fresh = _score(y, base, other.copy(), "diff", {})
    assert second == fresh and second != first
