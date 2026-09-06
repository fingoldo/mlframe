"""The stratified splitter must not reposition numba's global RNG stream for the rest of the thread.

`_iterative_stratification_njit` calls `np.random.seed(seed)` inside `@njit` scope. That sets *numba's*
per-thread global stream, not numpy's, and numba exposes no portable `get_state` -- so without an explicit
restore, every later njit kernel on that thread draws from wherever the splitter's draws happened to leave
the stream, and nothing in the caller can see it happen.

No current consumer is actually shifted (the numba-RNG users in the package re-seed per row or per
permutation before drawing), which is why this went unnoticed. The invariant is what matters: the next
unseeded numba draw added anywhere would silently inherit the splitter's position.

`feature_selection/filters/screen.py` is the codebase's own worked example of the hazard, and its
entropy-derived save/restore now lives in `mlframe.utils.rng_scope` so both sites share one implementation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from mlframe.utils.rng_scope import numba_rng_scope


@njit(cache=True)
def _numba_draws(n: int) -> np.ndarray:
    """Draw from numba's global stream without seeding it, so the caller's position is what shows up."""
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.random.random()
    return out


@njit(cache=True)
def _seed_numba(seed: int) -> None:
    """Put numba's global stream at a known position."""
    np.random.seed(seed)


def _draws_after_seed(seed: int, body) -> np.ndarray:
    """Seed numba, run `body`, then draw -- so the draws expose where `body` left the stream."""
    _seed_numba(seed)
    body()
    return _numba_draws(6)


def test_the_scope_moves_the_stream_off_where_the_block_left_it():
    """After the scope, the stream must not be sitting where the block's own draws left it.

    Compared against a deterministic reference -- the position an UNSCOPED block leaves -- rather than
    against a second scoped run. The restore is entropy-derived, so two scoped runs differ for reasons
    that have nothing to do with the fix, and an assertion resting on that is measuring the entropy
    source rather than the scope.
    """
    def _unscoped():
        """The pre-fix shape: seed numba, consume draws, leave the stream there."""
        _seed_numba(4242)
        _numba_draws(50)

    def _scoped():
        """The same block under the scope."""
        with numba_rng_scope(4242):
            _numba_draws(50)

    unrestored = _draws_after_seed(7, _unscoped)
    scoped = _draws_after_seed(7, _scoped)
    assert not np.array_equal(scoped, unrestored), (
        "the draws after the scoped block match the draws after the unscoped one exactly, so the scope "
        "left numba's stream at the position the block's own draws determined"
    )


def test_without_the_scope_the_block_dictates_what_follows():
    """The bug, shown directly: an unscoped seeded block pins every later draw on the thread."""
    def _unscoped_block():
        """The pre-fix shape: seed numba and consume draws, with no restore."""
        _seed_numba(4242)
        _numba_draws(50)

    first = _draws_after_seed(7, _unscoped_block)
    second = _draws_after_seed(7, _unscoped_block)
    assert np.array_equal(first, second), (
        "an unscoped seeded block no longer determines the following draws; this test has lost the " "contrast it exists to draw"
    )


def test_every_restore_seed_is_one_numba_can_accept():
    """The restore is best-effort, so a seed numba rejects makes the scope silently do nothing.

    numba types its seed argument as int64: a full 64-bit entropy draw is >= 2**63 about half the time and
    raises OverflowError on the way in. That landed in a debug log and left the stream exactly where the
    guarded block put it -- the scope failing, invisibly, on roughly half its calls. `screen.py` drew its
    numba and cupy restoration seeds the same way and had the same defect.
    """
    from pyutilz.data.numbalib import set_numba_random_seed

    from mlframe.utils import rng_scope

    seeds = {rng_scope._fresh_seed() for _ in range(2000)}
    assert len(seeds) > 1900, "the restore seeds are not varying; the entropy source is broken"
    worst = max(seeds)
    assert worst < 2**63, f"a restore seed reached {worst}, which numba's int64 seed argument cannot accept"
    set_numba_random_seed(worst)  # the real gate: numba must take the largest value this can produce


def test_an_unseeded_scope_touches_nothing():
    """`screen.py`'s rule: a run that was never seeded must not re-seed a stream it did not set."""
    _seed_numba(11)
    baseline = _numba_draws(6)

    _seed_numba(11)
    with numba_rng_scope(None):
        pass
    assert np.array_equal(_numba_draws(6), baseline), "an unseeded scope moved the stream"


def test_the_splitter_call_site_is_scoped():
    """End-to-end: a real split must not decide what the next unseeded njit kernel draws."""
    from mlframe.training._split_helpers import _stratified_split_3way

    rng = np.random.RandomState(0)
    n = 400
    y = np.column_stack([rng.randint(0, 2, n), rng.randint(0, 2, n)]).astype(np.int8)

    def _split():
        """One stratified split, seeded identically each time."""
        _stratified_split_3way(np.arange(n), test_size=0.2, val_size=0.2, stratify_y=y, random_state=5)

    first = _draws_after_seed(3, _split)
    second = _draws_after_seed(3, _split)
    assert not np.array_equal(first, second), "a stratified split left numba's stream at a position derived from its own draws"
