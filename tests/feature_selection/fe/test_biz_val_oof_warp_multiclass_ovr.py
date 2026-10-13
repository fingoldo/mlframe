"""Out-of-fold warp on a multiclass target whose class order is not monotone in the signal: one-vs-rest warps keep the MI the rank-scaled warp loses."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._oof_warp_fe import hybrid_oof_warp_fe, warp_candidates


def _non_monotone_class_case(n: int = 20000, seed: int = 0):
    """``a`` in U(0, 1): class 1 on the middle third; the two outer thirds split 50/50 between classes 0 and 2.

    The rank of the classes (0 < 1 < 2) puts class 1 between the two outer classes, so the rank-scaled warp averages the outer mixture onto the middle and the warp of ``a``
    carries less information about the class than ``a`` itself. The indicator of class 0 or 2 is a clean interval-plus-mixture warp that the raw column cannot resolve."""
    r = np.random.default_rng(seed)
    a = r.random(n)
    outer = np.where(r.random(n) < 0.5, 0, 2)
    y = np.where((a > 1 / 3) & (a < 2 / 3), 1, outer)
    return pd.DataFrame({"a": a}), y


def _best_ovr_mi(X: pd.DataFrame, y: np.ndarray) -> float:
    """Largest held-out MI among the one-vs-rest warps of column ``a``."""
    return max(c["mi_warp"] for c in warp_candidates(X, y, ["a"], multiclass="ovr"))


def test_biz_val_ovr_warp_keeps_mi_rank_warp_loses_on_non_monotone_classes():
    """The best one-vs-rest warp keeps at least 0.55 nats while the rank warp falls below the raw column's MI."""
    X, y = _non_monotone_class_case()
    rank = warp_candidates(X, y, ["a"])
    assert len(rank) == 1
    mi_raw, mi_rank = rank[0]["mi_raw"], rank[0]["mi_warp"]
    mi_ovr = _best_ovr_mi(X, y)

    assert mi_rank < mi_raw, (mi_rank, mi_raw)
    assert mi_ovr >= 0.55, mi_ovr
    assert mi_ovr - mi_rank >= 0.08, (mi_ovr, mi_rank)


def test_biz_val_ovr_warp_is_accepted_and_rank_warp_is_rejected_on_non_monotone_classes():
    """The operator accepts one-vs-rest warp columns and rejects the rank warp of the same column."""
    X, y = _non_monotone_class_case()
    _, appended_rank, _, _ = hybrid_oof_warp_fe(X, y, num_cols=["a"], top_k=5, multiclass="rank")
    _, appended_ovr, recipes_ovr, enc_ovr = hybrid_oof_warp_fe(X, y, num_cols=["a"], top_k=5, multiclass="ovr")

    assert appended_rank == []
    assert appended_ovr and all(name.startswith("oofwarp(a,ovr") for name in appended_ovr)
    assert list(enc_ovr.columns) == appended_ovr
    assert all(r.kind == "oof_warp1d" for r in recipes_ovr)


def test_ovr_is_the_rank_warp_on_a_binary_target():
    """Two classes have a single rank order, so the one-vs-rest mode must not change the candidates."""
    X, y = _non_monotone_class_case()
    y_bin = (y == 2).astype(np.int64)
    rank = warp_candidates(X, y_bin, ["a"])
    ovr = warp_candidates(X, y_bin, ["a"], multiclass="ovr")
    assert [c["cls"] for c in ovr] == [None]
    assert ovr[0]["gain"] == pytest.approx(rank[0]["gain"])


def test_unknown_multiclass_mode_is_rejected():
    """An unknown multiclass mode raises instead of silently falling back to the rank warp."""
    X, y = _non_monotone_class_case(n=1000)
    with pytest.raises(ValueError, match="multiclass"):
        warp_candidates(X, y, ["a"], multiclass="softmax")
