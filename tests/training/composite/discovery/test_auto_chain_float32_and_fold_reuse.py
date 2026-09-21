"""Auto-chain must keep a float32 block as float32 and stop re-slicing each fold's rows per candidate.

Every base upcast its tiny-sample matrix to float64 in parallel threads, and each of the roughly twelve candidates
fancy-index-copied both fold slices although the shared-fold LightGBM cache needs the train rows only on its first call.
LightGBM bins its input and float32 -> float64 is exact, so the block now stays float32 for LightGBM; a built fold hands
back its holdout slice and takes no train slice. Every CV score, chain and MI gain is unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest

import mlframe.training.composite.discovery._auto_chain as auto_chain
from mlframe.training.composite.discovery._auto_chain import discover_chains
from mlframe.training.composite.discovery._lgb_fold_cache import LgbFoldCache

pytest.importorskip("lightgbm")

_REAL_CV = auto_chain._y_scale_cv_rmse


def _heavy_tail(n: int = 1500, seed: int = 0):
    """A base-linear target with a cubed feature residual a tail-compressing chain can model."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, n)
    x0, x1 = rng.normal(size=n), rng.normal(size=n)
    y = 2.0 * base + (0.8 * x0 + 0.4 * x1) ** 3 + 0.05 * rng.normal(size=n)
    return y, base, np.column_stack([x0, x1])


def _scores(monkeypatch, x):
    """Every candidate's CV score and the returned chains for ``x``."""
    rows = []

    def recording(tf, **kw):
        """Record each candidate's score."""
        out = _REAL_CV(tf, **kw)
        rows.append((getattr(tf, "name", "raw"), out[0]))
        return out

    monkeypatch.setattr(auto_chain, "_y_scale_cv_rmse", recording)
    y, base, _ = _heavy_tail()
    chains = discover_chains(y=y, base=base, x_matrix=x, cv_folds=3, n_estimators=30, num_leaves=8, random_state=0, top_k=3, compute_mi_gain=True)
    return rows, [(c.name, c.rmse, c.mi_gain) for c in chains]


def test_float32_and_float64_blocks_give_identical_chains(monkeypatch):
    """Keeping float32 is a memory change only: every candidate score and every chain equals the float64 run."""
    _, _, x = _heavy_tail()
    rows32, chains32 = _scores(monkeypatch, x.astype(np.float32))
    rows64, chains64 = _scores(monkeypatch, x.astype(np.float32).astype(np.float64))
    assert rows32 == rows64
    assert any(name.startswith("chain_") for name, _ in rows32), "chain candidates must be scored for this to mean anything"
    assert chains32 == chains64


def test_a_built_fold_predicts_without_its_train_rows():
    """Once the fold is built, ``x_tr=None`` trains on the cached binned rows and gives the same predictions."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 4))
    t = 2.0 * x[:, 0] + rng.normal(size=600)
    tr, va = np.arange(400), np.arange(400, 600)
    mask = np.ones(400, dtype=bool)
    cache = LgbFoldCache(n_estimators=20, num_leaves=7, learning_rate=0.1, random_state=0, n_jobs=1)
    first = cache.fit_predict(0, x[tr], t[tr], mask, x[va])
    assert cache.has_fold(0)
    np.testing.assert_array_equal(cache.holdout(0), x[va])
    again = cache.fit_predict(0, None, t[tr], mask, cache.holdout(0))
    np.testing.assert_array_equal(first, again)


def test_a_masked_candidate_on_a_built_fold_still_learns():
    """The train-row count comes from the mask, so a masked candidate on a built fold subsets correctly."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(600, 4))
    t = 3.0 * x[:, 0] + rng.normal(size=600)
    tr, va = np.arange(400), np.arange(400, 600)
    cache = LgbFoldCache(n_estimators=30, num_leaves=7, learning_rate=0.1, random_state=0, n_jobs=1)
    cache.fit_predict(0, x[tr], t[tr], np.ones(400, dtype=bool), x[va])
    mask = np.ones(400, dtype=bool)
    mask[::3] = False
    preds = cache.fit_predict(0, None, t[tr], mask, cache.holdout(0))
    assert np.corrcoef(preds, t[va])[0, 1] > 0.9
