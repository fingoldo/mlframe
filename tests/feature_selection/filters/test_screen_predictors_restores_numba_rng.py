"""A seeded ``screen_predictors`` call must move numba's global stream off its own seed on exit, on every call.

The exit re-seed draws an entropy-derived seed and is best-effort: a seed numba rejects is logged at debug and the stream is
left where the screen put it. A raw 64-bit ``os.urandom`` draw is >= 2**63 about half the time, which numba's int64 seed
argument cannot take, so the restore silently did nothing on roughly half of all seeded screens.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from mlframe.feature_selection.filters import _screen_predictors as sp


def _tiny_screen(random_seed: Optional[int]) -> None:
    """Run the real screening loop on a small categorical problem."""
    rng = np.random.default_rng(0)
    n = 60
    factors_data = rng.integers(0, 3, size=(n, 4)).astype(np.int32)
    targets_data = rng.integers(0, 2, size=(n, 1)).astype(np.int32)
    sp.screen_predictors(
        factors_data=factors_data,
        factors_nbins=np.array([3, 3, 3, 3], dtype=np.int32),
        factors_names=[f"f{i}" for i in range(4)],
        targets_data=targets_data,
        targets_nbins=np.array([2], dtype=np.int32),
        y=np.array([0], dtype=np.int32),
        full_npermutations=2,
        baseline_npermutations=2,
        n_workers=1,
        verbose=0,
        random_seed=random_seed,
    )


def test_the_exit_reseed_is_accepted_even_when_the_entropy_draw_is_at_the_top_of_its_range(monkeypatch):
    """Entropy bytes of all ones are the worst case: unmasked they read as 2**64 - 1, a value numba refuses."""
    real_seed = sp.set_numba_random_seed
    calls: list[tuple[int, str]] = []

    def spy(seed):
        """Record each seed and whether numba accepted it, then behave exactly like the real call."""
        try:
            real_seed(seed)
        except Exception as exc:
            calls.append((int(seed), type(exc).__name__))
            raise
        calls.append((int(seed), "ok"))

    monkeypatch.setattr(sp, "set_numba_random_seed", spy)
    monkeypatch.setattr(os, "urandom", lambda k: b"\xff" * k)

    _tiny_screen(random_seed=1234)

    assert calls[0] == (1234, "ok"), f"the screen must seed numba for its own determinism first: {calls}"
    assert len(calls) == 2, f"expected one entry seed and one exit re-seed: {calls}"
    restore_seed, outcome = calls[1]
    assert outcome == "ok", f"numba rejected the exit re-seed {restore_seed} ({outcome}); the stream stayed on the screen's own seed"
    assert restore_seed != 1234
    assert 0 <= restore_seed < 2**32, f"restore seed {restore_seed} is outside what numpy's legacy seeding takes under NUMBA_DISABLE_JIT=1"


def test_an_unseeded_screen_never_touches_numba(monkeypatch):
    """Without random_seed there is nothing to restore, so neither the entry seed nor the exit re-seed may fire."""
    calls: list[int] = []
    monkeypatch.setattr(sp, "set_numba_random_seed", lambda seed: calls.append(int(seed)))

    _tiny_screen(random_seed=None)

    assert calls == []
