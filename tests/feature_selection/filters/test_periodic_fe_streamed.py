"""Modular FE scored in RAM-budget chunks of source columns returns the same winners and scores as the one-pass pipeline."""

import numpy as np
import pandas as pd
import pytest

import mlframe.feature_selection.filters._feature_engineering_mem_budget as budget
from mlframe.feature_selection.filters._periodic_fe import hybrid_modular_fe


def _frame(n=4000, seed=0):
    """Frame."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"c{i}": rng.integers(0, 10_000, n).astype(np.float64) for i in range(6)})
    y = ((X["c2"] % 7) < 3).astype(np.int64).to_numpy().copy()
    flip = rng.random(n) < 0.1
    y[flip] = 1 - y[flip]
    return X, y


@pytest.mark.parametrize("cols_per_chunk", [1, 2])
def test_streamed_matches_one_pass(monkeypatch, cols_per_chunk):
    """Streamed matches one pass."""
    X, y = _frame()
    full_aug, full_scores = hybrid_modular_fe(X, y, top_k=4)
    per_src = len(X) * 5 * 3 * 3 * 4
    real = budget._can_hoist_shared_buffer
    monkeypatch.setattr(
        budget, "_can_hoist_shared_buffer", lambda nb, n_workers=1: (False, nb, 0) if nb > cols_per_chunk * per_src else real(nb, n_workers=n_workers)
    )
    st_aug, st_scores = hybrid_modular_fe(X, y, top_k=4)
    appended = [c for c in full_aug.columns if c not in X.columns]
    assert "c2__mod_7" in appended or "c2__modsin_7" in appended or "c2__modcos_7" in appended
    assert [c for c in st_aug.columns if c not in X.columns] == appended
    pd.testing.assert_frame_equal(st_aug[appended], full_aug[appended])
    a = full_scores.set_index("engineered_col").sort_index()
    b = st_scores.set_index("engineered_col").sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_residue_that_equals_the_source_column_is_not_a_candidate(monkeypatch):
    """A column inside [0, period) has itself as residue, so its ``mod`` candidate (an exact copy of the raw column) is not emitted; ``sin`` / ``cos`` are."""
    from mlframe.feature_selection.filters._periodic_fe import engineered_name_modular, generate_modular_features

    monkeypatch.setenv("MLFRAME_MODULAR_DROP_COPIES", "1")

    rng = np.random.default_rng(0)
    inside = rng.integers(0, 2, 300).astype(float)
    wraps = rng.uniform(0, 50, 300)
    eng = generate_modular_features(pd.DataFrame({"a": inside, "b": wraps}), periods=(7.0,))
    assert engineered_name_modular("a", 7.0, "mod") not in eng.columns
    assert engineered_name_modular("a", 7.0, "sin") in eng.columns and engineered_name_modular("a", 7.0, "cos") in eng.columns
    assert engineered_name_modular("b", 7.0, "mod") in eng.columns


def test_gate_first_handles_source_columns_whose_names_contain_double_underscores(monkeypatch):
    """An engineered source column such as ``x__T1`` is mapped to its candidates by construction, not by splitting the candidate name; a periodic one is still found."""
    rng = np.random.default_rng(0)
    n = 4000
    hour = rng.uniform(0, 200, n)
    X = pd.DataFrame({"x__T1": hour, "noise": rng.normal(size=n)})
    y = (np.sin(2 * np.pi * (hour % 24) / 24) + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    X_aug, _ = hybrid_modular_fe(X, y, periods=(24.0,), top_k=3)
    assert any(c.startswith("x__T1__mod") for c in X_aug.columns if c not in X.columns)


def test_a_failed_candidate_allocation_is_retried_in_smaller_chunks_with_the_same_winners(monkeypatch):
    """A MemoryError on a large chunk halves the chunk instead of dropping the modular columns; the winners do not change."""
    import mlframe.feature_selection.filters._periodic_fe as pf

    X, y = _frame()
    ref_aug, ref_scores = hybrid_modular_fe(X, y, top_k=4)
    real = pf._hybrid_modular_fe_gate_first
    seen = []

    def flaky(X_, y_, src_cols, chunk, *a, **k):
        """Fail every call whose chunk exceeds two source columns."""
        seen.append(chunk)
        if chunk > 2:
            raise MemoryError("Unable to allocate")
        return real(X_, y_, src_cols, chunk, *a, **k)

    monkeypatch.setattr(pf, "_hybrid_modular_fe_gate_first", flaky)
    aug, scores = hybrid_modular_fe(X, y, top_k=4)
    assert seen[0] > 2 and seen[-1] <= 2
    assert [c for c in aug.columns if c not in X.columns] == [c for c in ref_aug.columns if c not in X.columns]
    pd.testing.assert_frame_equal(scores.set_index("engineered_col").sort_index(), ref_scores.set_index("engineered_col").sort_index())


def test_the_chunk_is_capped_by_a_fixed_candidate_budget_not_only_by_free_memory(monkeypatch):
    """With plenty of free memory the chunk still stays within the fixed candidate-matrix budget."""
    import mlframe.feature_selection.filters._periodic_fe as pf

    X, y = _frame()
    monkeypatch.setattr(pf, "_MODULAR_CHUNK_BYTES", len(X) * 5 * 3 * 3 * 4 * 2)
    seen = []
    real = pf._hybrid_modular_fe_gate_first
    monkeypatch.setattr(pf, "_hybrid_modular_fe_gate_first", lambda X_, y_, s, chunk, *a, **k: (seen.append(chunk), real(X_, y_, s, chunk, *a, **k))[1])
    hybrid_modular_fe(X, y, top_k=4)
    assert seen and seen[0] <= 2
