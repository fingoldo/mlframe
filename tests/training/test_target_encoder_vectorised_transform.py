"""Regression sensor: LeakageSafeEncoder.transform uses the vectorised factorize+gather path, bit-identical to the per-row loop.

The per-row `for i, c in enumerate(cats)` loop in `_encode_with_full_train_stat` was a 10M-iteration Python hotspot at scoring
time (~20s for target_mean, ~40s for woe at n=10M). The vectorised path computes the encoding once over the unique categories
and gathers it back. These tests pin (a) bit-identity vectorised-vs-per-row across methods/edge cases and (b) that `transform`
actually routes through the vectorised helper (a spy catches a regression that reverts to the full-length per-row loop).
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.feature_handling.target_encoders import LeakageSafeEncoder


def _fit(method, n=4000, n_cat=50, seed=1):
    rng = np.random.default_rng(seed)
    pool = np.array([f"c{i}" for i in range(n_cat)], dtype=object)
    Xtr = pool[rng.integers(0, n_cat, n)]
    y = (rng.random(n) < 0.3).astype(np.int64)
    enc = LeakageSafeEncoder(method=method, cv=3)
    enc.fit_transform(Xtr, y)
    tpool = np.array([f"c{i}" for i in range(n_cat + 3)], dtype=object)  # +3 unseen
    Xte = tpool[rng.integers(0, n_cat + 3, n)]
    return enc, Xte


@pytest.mark.parametrize("method", ["target_mean", "woe", "target_james_stein", "target_m_estimate"])
@pytest.mark.parametrize("n_cat", [1, 5, 50])
def test_vectorised_transform_bit_identical_to_per_row(method, n_cat):
    enc, Xte = _fit(method, n_cat=n_cat)
    from mlframe.training.feature_handling.target_encoders import _categorical_to_string_array

    cats = _categorical_to_string_array(Xte)
    vec = enc.transform(Xte)
    per_row = enc._encode_per_row(cats)
    assert np.array_equal(vec, per_row, equal_nan=True), f"{method} n_cat={n_cat}: vectorised != per-row"


def test_transform_routes_through_vectorised_not_full_per_row(monkeypatch):
    enc, Xte = _fit("woe")
    seen = {}
    orig = enc._encode_per_row

    def spy(cats):
        seen.setdefault("lens", []).append(len(cats))
        return orig(cats)

    monkeypatch.setattr(enc, "_encode_per_row", spy)
    out = enc.transform(Xte)
    assert out.shape[0] == len(Xte)
    # Per-row is invoked only over the (small) unique set, never the full 4000-row array.
    assert seen["lens"], "_encode_per_row was not called at all"
    assert max(seen["lens"]) < len(Xte), f"per-row loop ran over full array len {max(seen['lens'])}; vectorisation regressed"
