"""MRMR-level pins of the binning reuse across targets: a target-independent ``nbins_strategy`` bins a column once for every target of a suite, a supervised one bins it again for each target.

The unit-level pins live in ``tests/feature_selection/discretization/test_unsupervised_edges_reused_across_targets.py`` (``per_feature_edges``) and ``test_discretize_col_cache.py``. These run the whole MRMR fit: a second fit on the same columns with another target must compute no column edges for an unsupervised strategy, and must compute them again for a supervised one.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins_columns as cols
from mlframe.feature_selection.filters.mrmr import MRMR


def _frame(n: int = 800, seed: int = 0):
    """Eight numeric columns; two targets of different types on the same rows (a binary one and a regression one)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"x{i}" for i in range(8)])
    y_bin = (X["x0"] + 0.5 * rng.standard_normal(n) > 0).astype(np.int8)
    y_reg = (X["x1"] * X["x2"] + 0.3 * rng.standard_normal(n)).astype(np.float64)
    return X, y_bin, y_reg


def _count_uncached(monkeypatch) -> list:
    """The column-edge computations that really run (a memo hit does not reach the uncached builder), each recorded as a copy of the column."""
    calls: list = []
    real = cols._compute_col_edges_uncached

    def counting(col, *args, **kwargs):
        """Record one real computation (a copy of the column) and delegate."""
        calls.append(np.array(col, copy=True))
        return real(col, *args, **kwargs)

    monkeypatch.setattr(cols, "_compute_col_edges_uncached", counting)
    return calls


def _fit(X, y, strategy: str):
    """One MRMR fit with the strategy and no feature engineering, so only the binning of the raw columns is counted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=0, random_seed=0, nbins_strategy=strategy, fe_max_steps=0).fit(X, y)


@pytest.mark.parametrize("strategy", ["quantile_spacing", "freedman_diaconis", "sturges", "uniform"])
def test_unsupervised_strategy_bins_each_column_once_across_targets(monkeypatch, strategy):
    """The first target bins the raw columns; the second target (another type) computes no column edge again."""
    X, y_bin, y_reg = _frame()
    cols.clear_edge_memo()
    calls = _count_uncached(monkeypatch)
    _fit(X, y_bin, strategy)
    first = len(calls)
    calls.clear()
    _fit(X, y_reg, strategy)
    # The target itself is one of the binned columns of a fit (its own edges depend on the target, so it is recomputed); the pin is on the FEATURE columns.
    target = np.asarray(y_reg, dtype=np.float64)
    feature_recomputed = [c for c in calls if not np.allclose(c, target, atol=1e-6)]  # MRMR casts the target (float32), so compare with a tolerance
    assert first > 0, "the first fit must bin the columns"
    assert len(feature_recomputed) == 0, f"{strategy}: the second target re-bound {len(feature_recomputed)} feature column(s) that were binned for the first target"


def test_supervised_strategy_bins_again_for_another_target(monkeypatch):
    """Contrast: ``mdlp`` depends on the target, so the second target recomputes the edges (the pin is that it does NOT reuse them)."""
    X, y_bin, y_reg = _frame()
    cols.clear_edge_memo()
    calls = _count_uncached(monkeypatch)
    _fit(X, y_bin, "mdlp")
    first = len(calls)
    calls.clear()
    _fit(X, y_reg, "mdlp")
    assert first > 0
    assert len(calls) > 0, "a supervised strategy must recompute its edges for another target"


# KNOWN GAP, NOT IMPLEMENTED ON PURPOSE (decision 2026-11-23, tracker row MRMR-COMPOSITE-BIN-ONCE).
# Found fix: give the child fit the parent's column codes sliced by the child's rows, for target-independent strategies (their edges do not depend on y, so the same rows and the same edges give the same
# codes). Supervised strategies keep re-binning: the child target is T = y - beta*u, not y, so their edges differ.
# Why it is not implemented: the child fit runs only with fe_target_mix_candidates=True, which is OFF by default, and the measurement of 2026-11-23 (PLAN-05, three fixtures, nine seeds) found no value
# in that search: no change in what is selected, one +0.005 R2, at 1.1-2.8x the wall. The fix would touch the fit path (codes handed from parent to child, cache keys by row subset) to speed up a mode that
# nobody should switch on. strict=True: the day the fix lands the test XPASSes and fails the run, which forces removing this marker.
@pytest.mark.xfail(strict=True, reason="child fit of the target mix re-bins its row subset; fix not implemented because the candidate search showed no value (PLAN-05, MRMR-COMPOSITE-BIN-ONCE)")
@pytest.mark.parametrize("strategy", ["quantile_spacing", "freedman_diaconis", "sturges", "uniform"])
def test_composite_child_fit_adds_no_column_binning(monkeypatch, strategy):
    """The target-mix child fit (``fe_target_mix_candidates=True``) fits on a row subset of the parent. It should reuse the parent's codes (the same rows, the same edges), not bin the columns again.

    Pinned for target-independent strategies only; marked xfail (strict) because the fix is deliberately not implemented, see the comment above.
    """
    X, _y_bin, y_reg = _frame(n=1500)
    cols.clear_edge_memo()
    calls = _count_uncached(monkeypatch)
    _fit_with_mix(X, y_reg, mix=False, strategy=strategy)
    main_only = len(calls)
    calls.clear()
    cols.clear_edge_memo()
    fits = _count_fit_calls(monkeypatch)
    _fit_with_mix(X, y_reg, mix=True, strategy=strategy)
    with_child = len(calls)
    # Non-vacuous: the child fit must actually have run, otherwise zero extra binning proves nothing.
    assert len(fits) == 2, f"expected the main fit and one composite child fit, got {len(fits)} MRMR.fit call(s)"
    assert main_only > 0
    assert with_child == main_only, f"the composite child re-bound {with_child - main_only} column(s) the main fit had already binned ({main_only} main, {with_child} with the child)"


def _count_fit_calls(monkeypatch) -> list:
    """Record every ``MRMR.fit`` call (the main fit and each child fit of the target mix)."""
    calls: list = []
    real = MRMR.fit

    def counting(self, *args, **kwargs):
        """Record one fit and delegate."""
        calls.append(1)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(MRMR, "fit", counting)
    return calls


def _fit_with_mix(X, y, mix: bool, strategy: str = "quantile_spacing"):
    """One MRMR fit with the target-mix second pass on or off (the child fit runs only when both candidates and enable are on)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return MRMR(verbose=0, random_seed=0, nbins_strategy=strategy, fe_max_steps=1, fe_target_mix_enable=mix, fe_target_mix_candidates=mix).fit(X, y)


def test_default_nbins_strategy_is_quantile_spacing():
    """The default is the owner's choice of 2026-11-23: a strategy that depends on the column only (a suite of targets bins once) and whose bin count grows with N (see ``qs_nbins``)."""
    assert MRMR().nbins_strategy == "quantile_spacing"


@pytest.mark.parametrize("steps", [2, 3])
def test_no_column_content_is_binned_twice_within_one_fit_at_several_fe_steps(monkeypatch, steps):
    """With ``fe_max_steps > 1`` a column the first step already binned must not be binned again by a later step (the edge memo keys on content): every computed column is distinct in content."""
    X, _y_bin, y_reg = _frame(n=1500)
    cols.clear_edge_memo()
    calls = _count_uncached(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        MRMR(verbose=0, random_seed=0, fe_max_steps=steps, fe_target_mix_enable=False).fit(X, y_reg)
    assert len(calls) > 0
    keys = [hash(np.ascontiguousarray(c).tobytes()) for c in calls]
    repeated = len(keys) - len(set(keys))
    assert repeated == 0, f"fe_max_steps={steps}: {repeated} of {len(keys)} column binnings repeated the content of an earlier one"
