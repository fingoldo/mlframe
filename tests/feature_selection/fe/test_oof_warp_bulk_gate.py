"""A warp that wins only through the extreme tails of its column is rejected; a warp that wins in the bulk is kept."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._oof_warp_fe import hybrid_oof_warp_fe, warp_candidates


def _frame(n=10000, seed=0):
    """``tail``: noise in the bulk, 5% huge values that decide a squared term of y. ``wave``: a sine of the column decides y everywhere."""
    rng = np.random.default_rng(seed)
    tail = rng.normal(size=n)
    out = rng.random(n) < 0.05
    tail[out] = rng.choice([-1.0, 1.0], out.sum()) * rng.uniform(10, 30, out.sum())
    wave = rng.uniform(0, 6 * np.pi, n)
    y = np.sin(wave) + 0.02 * tail**2 + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"tail": tail, "wave": wave}), y


def test_candidates_carry_the_bulk_gain_next_to_the_full_gain():
    """The tail column has a significant full-range gain and none in the bulk; the wave column has both."""
    X, y = _frame()
    cands = {c["col"]: c for c in warp_candidates(X, y, ["tail", "wave"])}
    assert cands["tail"]["gain"] > 2.0 * cands["tail"]["se"]
    assert cands["tail"]["bulk_gain"] <= 2.0 * cands["tail"]["bulk_se"]
    assert cands["wave"]["bulk_gain"] > 2.0 * cands["wave"]["bulk_se"]


def test_stage_keeps_the_wave_warp_and_rejects_the_tail_warp():
    """Only the column whose structure lives in the bulk is warped."""
    X, y = _frame()
    _, appended, _, _ = hybrid_oof_warp_fe(X, y)
    assert appended == ["oofwarp(wave)"]
