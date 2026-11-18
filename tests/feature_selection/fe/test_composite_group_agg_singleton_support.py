"""The composite group aggregates are not generated when the conditioning support leaves every candidate a conditional MI of exactly 0."""

import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _composite_group_agg_fe as cga


def _frame(n: int = 4000, n_num: int = 8) -> "tuple[pd.DataFrame, np.ndarray]":
    """Three group columns, ``n_num`` continuous columns and a target that depends on the first column's mean within the (g0, g1) cell."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame({f"g{j}": rng.integers(0, card, n) for j, card in enumerate((4, 3, 5))})
    shift = rng.standard_normal((4, 3))
    for j in range(n_num):
        X[f"x{j}"] = rng.standard_normal(n).astype(np.float32)
    X["x0"] += (2.0 * shift[X["g0"].to_numpy(), X["g1"].to_numpy()]).astype(np.float32)
    y = (shift[X["g0"].to_numpy(), X["g1"].to_numpy()] + 0.3 * rng.standard_normal(n) > 0).astype(np.int64)
    return X, y


def _run(X, y, **kw):
    """The family with explicit key sets and numeric columns."""
    return cga.hybrid_composite_group_agg_fe(X, y, group_col_sets=[("g0", "g1"), ("g1", "g2")], **kw)


def test_singleton_support_skips_the_generation_and_says_so(monkeypatch, caplog):
    """Eight continuous conditioning columns make every row its own cell: nothing is generated, nothing is appended, and the skip is logged as a warning."""
    X, y = _frame()
    num = [f"x{j}" for j in range(8)]
    calls = []
    real = cga.generate_composite_group_agg_features
    monkeypatch.setattr(cga, "generate_composite_group_agg_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    with caplog.at_level(logging.WARNING):
        X_aug, appended, recipes, scores = _run(X, y, num_cols=num)
    assert calls == [] and appended == [] and recipes == [] and X_aug is X
    assert list(scores.columns) == ["engineered_col", "source_col", "cmi", "source_mi", "uplift"] and len(scores) == 0
    assert any("every candidate's conditional MI is 0" in r.getMessage() for r in caplog.records)
    # the unshortened path reaches the same verdict: every score is exactly 0 and nothing passes the threshold
    monkeypatch.setattr(cga, "conditioning_leaves_no_candidate", lambda *a, **k: False)
    _x, appended_full, _r, scores_full = _run(X, y, num_cols=num)
    assert calls == [1] and appended_full == [] and len(scores_full) > 0 and float(scores_full["cmi"].abs().max()) == 0.0


def test_a_support_that_leaves_room_changes_nothing(monkeypatch):
    """One conditioning column (ten cells): the check answers no, and the winners, the scores and the appended columns equal those without the check."""
    X, y = _frame()
    kw = dict(num_cols=["x0"], min_cmi=1e-6, min_uplift=-1e9)  # a positive threshold (the check is live) and no uplift gate, so the planted effect is kept
    got = _run(X, y, **kw)
    monkeypatch.setattr(cga, "conditioning_leaves_no_candidate", lambda *a, **k: False)
    want = _run(X, y, **kw)
    assert got[1] == want[1] and got[1], "the planted group effect must be found"
    pd.testing.assert_frame_equal(got[3], want[3])
    pd.testing.assert_frame_equal(got[0], want[0])


def test_a_non_positive_threshold_still_generates(monkeypatch):
    """With ``min_cmi <= 0`` a zero score passes, so the candidates are generated even under a singleton support (the refit of the winners calls the family this way)."""
    X, y = _frame()
    calls = []
    real = cga.generate_composite_group_agg_features
    monkeypatch.setattr(cga, "generate_composite_group_agg_features", lambda *a, **k: calls.append(1) or real(*a, **k))
    _x, appended, _r, _s = _run(X, y, num_cols=[f"x{j}" for j in range(8)], min_cmi=-1e9, min_uplift=-1e9, top_k=3)
    assert calls == [1] and len(appended) == 3
