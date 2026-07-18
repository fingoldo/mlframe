"""Regression: ``_engineered_continuous_`` fit-time scratch is pruned after each FE round.

Pre-fix, ``self._engineered_continuous_[col]`` (a full-length float64 array per engineered column,
written in ``_step_score.py``) accumulated across EVERY FE round and was freed only ONCE at fit end
(``_fit_impl_core.py``, "FIT-TIME SCRATCH" delete). On a multi-round fit where most of a round's
engineered candidates do NOT survive the next mRMR redundancy screen, this held onto n_rows*8 bytes
per rejected candidate for the rest of the fit -- unbounded growth proportional to the TOTAL number
of engineered candidates ever produced, not the number actually still reachable.

The fix prunes ``_engineered_continuous_`` right after each round's ``screen_predictors`` redundancy
filter finalises ``selected_vars``, dropping entries for engineered columns that did not survive (see
``_prune_engineered_continuous_store`` in ``_mrmr_fit_impl/_helpers.py`` for the safety argument: the
FE operand pool only widens beyond ``selected_vars`` on the very first FE step, before any engineered
column exists, so a column absent from a fresh ``selected_vars`` can never be read as an operand again).

TEST STRATEGY NOTE: an end-to-end MRMR.fit on small synthetic fixtures (n<=5000, up to 10 candidate
pairs, fe_max_steps up to 3) was probed empirically and consistently showed screen_predictors's
forward-selection re-screen KEEPING every just-engineered candidate (0 drops) -- its stopping
criterion is lenient enough that a handful of genuinely-distinct nonlinear candidates each still
clear the incremental-gain bar. Production-scale fits (hundreds of engineered candidates, many
genuinely near-duplicate/redundant) are where real drops occur; that scale is not practical for a
fast unit test. So this module proves the fix two ways: (1) DIRECT, deterministic tests of
``_prune_engineered_continuous_store`` against controlled before/after states (the primary proof the
memory win is real and correctly scoped), and (2) an END-TO-END wiring test on a real multi-round FE
fit proving the hook fires with the real ``(cols, selected_vars)`` shapes without crashing, and that
final selection is BYTE-IDENTICAL to a baseline run with pruning forced to a no-op (the safety proof
that matters regardless of whether any given fit happens to trigger a drop).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl import _helpers as _h
from mlframe.feature_selection.filters.mrmr import MRMR

# ---------------------------------------------------------------------------
# DIRECT: _prune_engineered_continuous_store against controlled before/after state.
# ---------------------------------------------------------------------------


class _FakeMRMRState:
    """Minimal stand-in exposing only what the pruner reads: ``_engineered_continuous_``."""

    def __init__(self, store: dict):
        self._engineered_continuous_ = store


def test_prune_drops_entries_not_in_selected_vars():
    """Prune drops entries not in selected vars."""
    n = 500
    cols = ["raw_a", "raw_b", "eng_survivor", "eng_dropped_1", "eng_dropped_2"]
    store = {
        "eng_survivor": np.zeros(n, dtype=np.float64),
        "eng_dropped_1": np.zeros(n, dtype=np.float64),
        "eng_dropped_2": np.zeros(n, dtype=np.float64),
    }
    inst = _FakeMRMRState(dict(store))
    # Only "eng_survivor" (index 2) is in the fresh selected_vars; raw_a/raw_b are also selected
    # but never appear in the store (only engineered columns are ever written there).
    selected_vars = [0, 1, 2]
    n_dropped = _h._prune_engineered_continuous_store(inst, cols, selected_vars)
    assert n_dropped == 2
    assert set(inst._engineered_continuous_.keys()) == {"eng_survivor"}


def test_prune_keeps_everything_when_all_survive():
    """Prune keeps everything when all survive."""
    n = 200
    cols = ["raw_a", "eng_1", "eng_2"]
    store = {"eng_1": np.zeros(n), "eng_2": np.zeros(n)}
    inst = _FakeMRMRState(dict(store))
    selected_vars = [0, 1, 2]  # both engineered columns survive
    n_dropped = _h._prune_engineered_continuous_store(inst, cols, selected_vars)
    assert n_dropped == 0
    assert set(inst._engineered_continuous_.keys()) == {"eng_1", "eng_2"}


def test_prune_drops_everything_when_none_survive():
    """Prune drops everything when none survive."""
    n = 200
    cols = ["raw_a", "eng_1", "eng_2"]
    store = {"eng_1": np.zeros(n), "eng_2": np.zeros(n)}
    inst = _FakeMRMRState(dict(store))
    selected_vars = [0]  # neither engineered column survives this round's screen
    n_dropped = _h._prune_engineered_continuous_store(inst, cols, selected_vars)
    assert n_dropped == 2
    assert inst._engineered_continuous_ == {}


def test_prune_is_noop_on_empty_or_absent_store():
    """Prune is noop on empty or absent store."""
    inst_empty = _FakeMRMRState({})
    assert _h._prune_engineered_continuous_store(inst_empty, ["a", "b"], [0, 1]) == 0

    class _NoStore:
        """Groups tests covering NoStore."""
        pass

    assert _h._prune_engineered_continuous_store(_NoStore(), ["a", "b"], [0, 1]) == 0


def test_prune_bounds_peak_store_size_across_simulated_rounds():
    """Simulates the accumulate-then-prune pattern across several FE rounds: without pruning the
    store would grow monotonically (total candidates ever seen); with pruning it stays bounded by
    the number of CURRENTLY-reachable engineered columns. This is the direct proof of the memory win
    the audit finding targets, independent of whether any single real fit happens to trigger a drop."""
    n = 300
    cols = ["raw_a", "raw_b"]
    store: dict = {}
    inst = _FakeMRMRState(store)
    total_ever_created = 0
    peak_with_pruning = 0
    rng = np.random.default_rng(0)

    for round_idx in range(5):
        # Each round "engineers" 4 new candidates; append to cols/store (mirrors _step_score.py writes).
        new_names = [f"round{round_idx}_cand{k}" for k in range(4)]
        for nm in new_names:
            cols.append(nm)
            store[nm] = rng.normal(size=n)
            total_ever_created += 1
        # This round's screen keeps only the FIRST candidate of this round plus everything from prior
        # rounds' survivors (simulate a round that admits few new features but never revisits old ones).
        survivors_this_round = [cols.index(nm) for nm in new_names[:1]]
        prior_engineered_survivors = [cols.index(k) for k in list(store.keys()) if k not in new_names][:1]
        selected_vars = [0, 1, *survivors_this_round, *prior_engineered_survivors]
        _h._prune_engineered_continuous_store(inst, cols, selected_vars)
        peak_with_pruning = max(peak_with_pruning, len(store))

    assert total_ever_created == 20, "fixture sanity: 5 rounds x 4 candidates"
    assert (
        peak_with_pruning < total_ever_created
    ), f"pruning must bound the store below the cumulative candidate count: peak={peak_with_pruning} vs total_ever_created={total_ever_created}"
    # With pruning each round keeps at most ~2 entries alive (1 new + 1 carried survivor).
    assert peak_with_pruning <= 3, f"expected pruning to keep the store small across rounds; peak={peak_with_pruning}"


# ---------------------------------------------------------------------------
# END-TO-END: wiring fires on a real multi-round FE fit; selection unchanged vs no-pruning baseline.
# ---------------------------------------------------------------------------


def _canonical_composite_fixture(seed: int = 0, n: int = 3000):
    """The proven canonical fixture from test_mrmr_fe_composite_feedforward.py: y = a**2/b + f/5 +
    log(c)*sin(d). Reliably engineers multiple pair candidates (+ a within-step additive-fusion
    composite) in round 1, giving the confirm-rescreen a non-empty ``_engineered_continuous_`` to
    prune against."""
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    b = rng.random(n)
    c = rng.random(n)
    d = rng.random(n)
    e = rng.random(n)
    f = rng.random(n)
    y = a**2 / b + f / 5.0 + np.log(c) * np.sin(d)
    df = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e})
    return df, pd.Series(y, name="y")


@pytest.mark.slow
def test_prune_hook_fires_during_real_multi_round_fit():
    """The prune hook must actually execute (with a non-empty store and well-formed args) during a
    real fit that engineers pair candidates -- proves the wiring is live, not dead code."""
    df, y = _canonical_composite_fixture()
    calls: list[tuple[int, int]] = []  # (len(cols) at call time, len(store) at call time)
    orig = _h._prune_engineered_continuous_store

    def spy(instance, cols, selected_vars):
        """Helper that spy."""
        calls.append((len(cols), len(getattr(instance, "_engineered_continuous_", None) or {})))
        return orig(instance, cols, selected_vars)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_h, "_prune_engineered_continuous_store", spy)
        m = MRMR(verbose=0, fe_max_steps=1, n_workers=1, fit_cache_max=0, random_seed=42)
        m.fit(df, y)

    assert calls, "expected the prune hook to run at least once during a multi-round FE fit"
    assert any(
        store_len > 0 for _cols_len, store_len in calls
    ), f"expected at least one prune-hook call to see a populated _engineered_continuous_ store; calls={calls}"


@pytest.mark.slow
def test_selection_unchanged_vs_no_pruning_baseline():
    """Pruning must be a pure memory optimisation: byte-identical final selection vs a baseline run
    with the pruner forced to a no-op (the pre-fix accumulate-everything behaviour)."""
    df, y = _canonical_composite_fixture()

    with pytest.MonkeyPatch.context() as mp:
        m_pruned = MRMR(verbose=0, fe_max_steps=1, n_workers=1, fit_cache_max=0, random_seed=42)
        m_pruned.fit(df, y)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_h, "_prune_engineered_continuous_store", lambda *a, **k: 0)
        m_baseline = MRMR(verbose=0, fe_max_steps=1, n_workers=1, fit_cache_max=0, random_seed=42)
        m_baseline.fit(df, y)

    assert list(m_pruned.get_feature_names_out()) == list(m_baseline.get_feature_names_out()), "pruning must not change the final selection"
    assert list(m_pruned.support_) == list(m_baseline.support_)


@pytest.mark.slow
def test_engineered_continuous_store_freed_at_fit_end():
    """Post-fit, the scratch store must not be a populated instance attribute (pre-existing contract,
    unaffected by mid-fit pruning -- the final delete still fires)."""
    df, y = _canonical_composite_fixture(seed=5)
    m = MRMR(verbose=0, fe_max_steps=1, n_workers=1, fit_cache_max=0, random_seed=5)
    m.fit(df, y)
    assert not getattr(m, "_engineered_continuous_", None), "engineered-continuous scratch must be empty/absent after fit"
