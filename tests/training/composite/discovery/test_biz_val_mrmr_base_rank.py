"""Unit + biz_value tests for MRMR base-candidate ranking in composite discovery.

Covers ``_mrmr_base_rank.mrmr_rank_bases`` (the pure greedy selector) and its
opt-in hook in ``_auto_base`` behind ``base_ranking_criterion="mrmr"``:

- Greedy MRMR picks the most-relevant candidate first, then a diverse one even
  when a redundant candidate has marginally higher relevance; ``beta=0`` reduces
  to pure relevance ordering; degenerate pools (1 candidate, all-identical,
  ``k>=n``) behave.
- Flag OFF -> ``_auto_base`` output is byte-identical to the legacy MI path (a
  pin that FAILS if the MRMR branch ever leaks into the default).
- biz_value: with 3 near-duplicate strong bases + 1 independent slightly-weaker
  base, MRMR's top-K includes the independent base (diversity) while pure MI
  fills with duplicates, giving lower ensemble correlation AND better held-out
  RMSE by a quantitative margin.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.composite.discovery._auto_base import _auto_base
from mlframe.training.composite.discovery._mrmr_base_rank import mrmr_rank_bases
from mlframe.training._composite_target_discovery_config import (
    CompositeTargetDiscoveryConfig,
)


# --------------------------------------------------------------------------- #
# 1. Unit: the pure greedy selector.
# --------------------------------------------------------------------------- #
def test_mrmr_picks_relevant_first_then_diverse_second():
    """b has marginally higher relevance than c but is highly redundant with
    the top pick a; MRMR must pick the diverse c second, not the redundant b."""
    names = ["a", "b", "c"]
    rel = [1.0, 0.95, 0.60]
    # a~b strongly redundant; a~c and b~c nearly independent.
    red = np.array(
        [
            [0.0, 0.90, 0.05],
            [0.90, 0.0, 0.04],
            [0.05, 0.04, 0.0],
        ]
    )
    order = mrmr_rank_bases(names, rel, red, 3, beta=1.0)
    assert order[0] == "a", "max-relevance candidate must lead"
    assert order[1] == "c", f"diverse candidate must be picked 2nd, got {order}"


def test_mrmr_beta_zero_is_pure_relevance_order():
    """beta=0 kills the redundancy term -> ordering is exactly by relevance."""
    names = ["a", "b", "c"]
    rel = [1.0, 0.95, 0.60]
    red = np.array(
        [
            [0.0, 0.90, 0.05],
            [0.90, 0.0, 0.04],
            [0.05, 0.04, 0.0],
        ]
    )
    assert mrmr_rank_bases(names, rel, red, 3, beta=0.0) == ["a", "b", "c"]


def test_mrmr_callable_redundancy_matches_matrix():
    """The callable redundancy source must produce the same order as the
    equivalent matrix (the ``_auto_base`` hook passes a memoised callable)."""
    names = ["a", "b", "c"]
    rel = [1.0, 0.95, 0.60]
    red = np.array(
        [
            [0.0, 0.90, 0.05],
            [0.90, 0.0, 0.04],
            [0.05, 0.04, 0.0],
        ]
    )
    via_fn = mrmr_rank_bases(names, rel, lambda i, j: red[i, j], 3, beta=1.0)
    assert via_fn == mrmr_rank_bases(names, rel, red, 3, beta=1.0)


def test_mrmr_degenerate_pools():
    """1 candidate, all-identical redundancy, and k>=n all behave."""
    assert mrmr_rank_bases(["only"], [0.5], np.zeros((1, 1)), 3) == ["only"]
    assert mrmr_rank_bases([], [], np.zeros((0, 0)), 3) == []
    assert mrmr_rank_bases(["a", "b"], [0.4, 0.4], np.ones((2, 2)), 0) == []
    # All-identical relevance + redundancy: still returns min(k, n) items,
    # ties broken by original order.
    ident = mrmr_rank_bases(
        ["a", "b", "c"],
        [0.7, 0.7, 0.7],
        np.ones((3, 3)),
        5,
        beta=1.0,
    )
    assert ident == ["a", "b", "c"]


def test_mrmr_validates_shapes():
    """Mrmr validates shapes."""
    with pytest.raises(ValueError):
        mrmr_rank_bases(["a", "b"], [0.1], np.zeros((2, 2)), 2)
    with pytest.raises(ValueError):
        mrmr_rank_bases(["a", "b"], [0.1, 0.2], np.zeros((3, 3)), 2)


# --------------------------------------------------------------------------- #
# 2. ``_auto_base`` integration harness (minimal stub ``self``).
# --------------------------------------------------------------------------- #
def _make_self(config, X: np.ndarray, feature_names: list[str]):
    """Make self."""
    name_to_col = {n: i for i, n in enumerate(feature_names)}

    def _build_feature_matrix(df, cols, idx):
        """Build feature matrix."""
        if not cols:
            return np.zeros((idx.size, 0), dtype=np.float64)
        return np.column_stack([X[idx, name_to_col[c]] for c in cols])

    return SimpleNamespace(
        config=config,
        _build_feature_matrix=_build_feature_matrix,
        _hint_strengths_pct=None,
    )


def _base_config(**overrides):
    """Base config."""
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        auto_base_top_k=2,
        auto_base_demote_time_index=False,
        auto_base_demote_spatial_coords=False,
        auto_base_structural_boost=False,
        auto_base_null_perms=0,
        auto_base_dedup_corr_threshold=1.0,  # isolate MRMR from crude dedup
        base_max_abs_corr_with_y=1.0,  # keep near-copy exclusion out
        mi_estimator="bin",
        mi_nbins=16,
        mi_sample_n=None,
        random_state=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _redundant_plus_independent(n: int, rng: np.random.Generator):
    """3 near-duplicate strong bases (share ``sig1``) + 1 independent base
    (``sig2``, slightly weaker relevance). y depends on BOTH signals."""
    sig1 = rng.normal(size=n)
    sig2 = rng.normal(size=n)
    y = 3.0 * sig1 + 1.5 * sig2 + rng.normal(size=n) * 0.3
    dup1 = sig1 + rng.normal(size=n) * 0.4
    dup2 = sig1 + rng.normal(size=n) * 0.4
    dup3 = sig1 + rng.normal(size=n) * 0.4
    indep = sig2 + rng.normal(size=n) * 0.4
    names = ["dup1", "dup2", "dup3", "indep"]
    X = np.column_stack([dup1, dup2, dup3, indep]).astype(np.float64)
    return names, X, y.astype(np.float64)


def _holdout_rmse(X: np.ndarray, cols: list[str], names: list[str], y: np.ndarray) -> float:
    """OLS on the first half, RMSE on the held-out second half."""
    idx = {n: i for i, n in enumerate(names)}
    A = np.column_stack([X[:, idx[c]] for c in cols])
    A = np.column_stack([A, np.ones(len(A))])
    h = len(y) // 2
    coef, *_ = np.linalg.lstsq(A[:h], y[:h], rcond=None)
    pred = A[h:] @ coef
    return float(np.sqrt(np.mean((y[h:] - pred) ** 2)))


def _mean_abs_corr(X: np.ndarray, cols: list[str], names: list[str]) -> float:
    """Mean abs corr."""
    idx = {n: i for i, n in enumerate(names)}
    cs = [X[:, idx[c]] for c in cols]
    vals = []
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)):
            vals.append(abs(np.corrcoef(cs[i], cs[j])[0, 1]))
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# 3. Byte-identity pin: flag OFF == legacy MI path.
# --------------------------------------------------------------------------- #
def test_auto_base_default_byte_identical_to_mi_and_mrmr_differs():
    """Default (no criterion) == explicit "mi", and MRMR yields a DIFFERENT
    top-K on the redundant scenario -- so if the MRMR branch ever leaked into
    the default, ``default == mi`` would still hold but ``mrmr != mi`` would
    make ``default == mrmr`` (this test would then fail)."""
    rng = np.random.default_rng(20260702)
    names, X, y = _redundant_plus_independent(4000, rng)
    train_idx = np.arange(len(y))

    cfg_default = _base_config()  # no base_ranking_criterion attribute set
    top_default = _auto_base(_make_self(cfg_default, X, names), None, names, y, train_idx)

    cfg_mi = _base_config(base_ranking_criterion="mi")
    top_mi = _auto_base(_make_self(cfg_mi, X, names), None, names, y, train_idx)

    cfg_mrmr = _base_config(base_ranking_criterion="mrmr")
    top_mrmr = _auto_base(_make_self(cfg_mrmr, X, names), None, names, y, train_idx)

    assert top_default == top_mi, "default must be byte-identical to the MI path"
    assert top_mrmr != top_mi, "MRMR must diverge from MI on the redundant scenario"


# --------------------------------------------------------------------------- #
# 4. biz_value: MRMR shortlist is more diverse AND predicts better.
# --------------------------------------------------------------------------- #
def test_biz_val_mrmr_shortlist_diverse_and_lower_rmse():
    """Pure MI fills the top-2 with sig1 duplicates; MRMR swaps in the
    independent sig2 base. Assert: MRMR top-2 contains ``indep``, MI does NOT;
    MRMR shortlist has lower mean pairwise corr AND lower held-out RMSE."""
    rng = np.random.default_rng(7)
    names, X, y = _redundant_plus_independent(6000, rng)
    train_idx = np.arange(len(y))

    top_mi = _auto_base(
        _make_self(_base_config(base_ranking_criterion="mi"), X, names),
        None,
        names,
        y,
        train_idx,
    )
    top_mrmr = _auto_base(
        _make_self(_base_config(base_ranking_criterion="mrmr"), X, names),
        None,
        names,
        y,
        train_idx,
    )

    assert "indep" not in top_mi, f"MI top-K should fill with dups, got {top_mi}"
    assert "indep" in top_mrmr, f"MRMR must surface the independent base, got {top_mrmr}"

    corr_mi = _mean_abs_corr(X, top_mi, names)
    corr_mrmr = _mean_abs_corr(X, top_mrmr, names)
    assert corr_mrmr < corr_mi - 0.3, f"MRMR shortlist must be less redundant: mi={corr_mi:.3f} mrmr={corr_mrmr:.3f}"

    rmse_mi = _holdout_rmse(X, top_mi, names, y)
    rmse_mrmr = _holdout_rmse(X, top_mrmr, names, y)
    # sig2 carries ~1.5x of y's variance the dups cannot explain; adding it
    # measurably cuts held-out error. Floor 15% below the measured ~40% gain.
    assert rmse_mrmr < 0.85 * rmse_mi, f"MRMR shortlist must predict better: mi={rmse_mi:.4f} mrmr={rmse_mrmr:.4f}"
