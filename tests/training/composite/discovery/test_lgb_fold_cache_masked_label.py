"""A masked candidate in the shared-fold LightGBM cache must train on its own label, not on the placeholder.

``LgbFoldCache`` builds one binned dataset per fold with an all-zeros placeholder label and trains each candidate on a
row ``subset`` of it with that candidate's target. A lazy subset constructs itself from the parent at train time and
takes the parent's label with it, so setting the label before construction was silently ignored: every candidate whose
domain mask removed a row trained on zeros and predicted a constant, while ``get_label()`` still reported the label
that had been set. Auto-chain discovery scores its log/ratio candidates through exactly this path.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery._lgb_fold_cache import LgbFoldCache

pytest.importorskip("lightgbm")


def _data(n: int = 3000, seed: int = 0):
    """A feature block with a strong, learnable target."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    y = 3.0 * x[:, 0] + rng.normal(size=n)
    return x, y


def _cache() -> LgbFoldCache:
    """A small single-threaded cache."""
    return LgbFoldCache(n_estimators=40, num_leaves=15, learning_rate=0.1, random_state=0, n_jobs=1)


def test_a_masked_candidate_learns_its_target():
    """With rows masked out, predictions must track the target, not collapse to a constant."""
    x, y = _data()
    mask = np.ones(len(y), dtype=bool)
    mask[::3] = False
    preds = _cache().fit_predict(0, x, y, mask, x)
    assert np.std(preds) > 0.5 * np.std(y), f"masked candidate predicted a near-constant (std={np.std(preds):.4f})"
    assert np.corrcoef(preds, y)[0, 1] > 0.9


def test_masked_and_unmasked_candidates_agree_closely():
    """Dropping a third of the rows must not turn a good fit into a useless one."""
    x, y = _data()
    cache = _cache()
    full = cache.fit_predict(0, x, y, np.ones(len(y), dtype=bool), x)
    mask = np.ones(len(y), dtype=bool)
    mask[::3] = False
    masked = cache.fit_predict(0, x, y, mask, x)
    rmse_full = float(np.sqrt(np.mean((full - y) ** 2)))
    rmse_masked = float(np.sqrt(np.mean((masked - y) ** 2)))
    assert rmse_masked < 1.5 * rmse_full, f"masked RMSE {rmse_masked:.3f} vs full {rmse_full:.3f}"


def test_two_candidates_on_one_fold_do_not_share_a_label():
    """The fold dataset is reused across candidates; each must see its own target, masked or not."""
    x, y = _data()
    cache = _cache()
    mask = np.ones(len(y), dtype=bool)
    mask[::4] = False
    first = cache.fit_predict(0, x, y, mask, x)
    second = cache.fit_predict(0, x, -y, mask, x)
    assert np.corrcoef(first, second)[0, 1] < -0.9, "the second candidate must learn -y, not reuse the first's label"
