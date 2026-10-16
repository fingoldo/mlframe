"""The pair family gate runs exactly when its validation candidates fit the cell budget; a larger family is decided by the table floor alone."""

import numpy as np
import pandas as pd

import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit as emit
import mlframe.feature_selection.filters._feature_engineering_pairs._pairs_emit_extval as extval


def _frame(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"f{i}": rng.standard_normal(n) for i in range(8)})
    y = ((X["f0"] * X["f1"] + X["f2"] / (1 + X["f3"].abs()) + 0.3 * rng.standard_normal(n)) > 0).astype(np.int64)
    return X, y


def _fit_calls(monkeypatch, max_cells):
    calls = {"exact": [], "floor": []}
    real_exact, real_floor = emit._apply_pair_family_gate, emit._apply_pair_family_floor

    def exact(pair_batches, valid_pairs_perf, classes_y, n_bins, alpha=0.01):
        calls["exact"].append(sum(int(np.asarray(c).shape[1]) for _, c in pair_batches))
        return real_exact(pair_batches, valid_pairs_perf, classes_y, n_bins, alpha)

    def floor(valid_pairs_perf, classes_y, n_bins, n_rows, k_total):
        calls["floor"].append(int(k_total))
        return real_floor(valid_pairs_perf, classes_y, n_bins, n_rows, k_total)

    monkeypatch.setattr(emit, "_apply_pair_family_gate", exact)
    monkeypatch.setattr(emit, "_apply_pair_family_floor", floor)
    monkeypatch.setattr(extval, "_FAMILY_EXACT_MAX_CELLS", max_cells)
    from mlframe.feature_selection.filters.mrmr import MRMR

    X, y = _frame()
    MRMR(verbose=0, random_state=0, n_jobs=1, fit_cache_max=0).fit(X, y)
    return calls


def test_small_family_takes_the_exact_gate(monkeypatch):
    calls = _fit_calls(monkeypatch, 10**12)
    assert calls["exact"] and not calls["floor"]


def test_family_over_the_budget_is_decided_by_the_floor_and_never_built(monkeypatch):
    calls = _fit_calls(monkeypatch, 0)
    assert calls["floor"] and not calls["exact"]
    assert any(k > 0 for k in calls["floor"]), "at least one pair must have had validation candidates gated by the floor"
