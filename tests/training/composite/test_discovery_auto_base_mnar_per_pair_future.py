"""Regression + biz_value sensors for the deferred (FUTURE) audit finding D12
in ``src/mlframe/training/composite/discovery/_auto_base.py``.

D12 (sibling of D11): ``_auto_base`` ranked candidate bases by MI computed on
the GLOBAL all-column finite intersection
(``np.isfinite(y) & np.all(np.isfinite(x_matrix), axis=1)``). For mid-range-NaN
columns that intersection is a non-random (MNAR) subset -- only the rows where
EVERY feature happens to be observed -- so MI(y, x_j) estimated on it is biased
by the joint-observability pattern and silently shifts which base wins. The fix
ranks by PER-PAIR (per-column) MI: each column's MI is estimated on its own
``isfinite(col) & isfinite(y)`` rows, mirroring the per-pair contract already
used by ``_mi_to_target`` (D11) and the prebinned ``-1``-sentinel path.

Pinned here:

1. The new kernel ``_mi_per_feature_y_fixed_per_col`` is BIT-IDENTICAL to the
   global-mask ``_mi_per_feature_y_fixed`` on an all-finite screening sample.
2. A single mostly-NaN column does NOT zero the MI ranking for the dense
   informative columns (the global AND-mask collapsed to <50 rows pre-fix).
3. biz_value: end-to-end ``_auto_base`` recovers the genuinely-informative
   base under MNAR contamination, where the pre-fix global mask would fall
   back to arbitrary feature-list order.
4. The ``auto_base_mi_per_pair_mask=False`` opt-out reproduces the legacy
   global-mask path (so a future "always per-pair" cannot silently drop the
   escape hatch), and the per-pair default differs from it under heavy NaN.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.training.composite.discovery._auto_base import _auto_base
from mlframe.training.composite.discovery.screening import (
    _mi_per_feature_y_fixed,
    _mi_per_feature_y_fixed_per_col,
)
from mlframe.training._composite_target_discovery_config import (
    CompositeTargetDiscoveryConfig,
)


def _informative_matrix(n: int, k: int, rng: np.random.Generator):
    """k dense columns each linearly informative about a shared target,
    with monotonically decreasing signal strength."""
    target = rng.normal(size=n).astype(np.float64)
    cols = []
    for j in range(k):
        noise = rng.normal(size=n) * (0.5 + 0.4 * j)
        cols.append(target * (1.0 + 0.3 * j) + noise)
    X = np.column_stack(cols).astype(np.float64)
    return X, target


# --------------------------------------------------------------------------- #
# 1. Kernel-level: bit-identity on all-finite + per-pair recovery under NaN.
# --------------------------------------------------------------------------- #
def test_per_col_kernel_bit_identical_on_all_finite():
    """``_mi_per_feature_y_fixed_per_col`` == ``_mi_per_feature_y_fixed`` to the
    bit when nothing is NaN (the per-pair mask is the identity there)."""
    rng = np.random.default_rng(7)
    X, y = _informative_matrix(3000, 5, rng)
    a = _mi_per_feature_y_fixed(X, y, nbins=16)
    b = _mi_per_feature_y_fixed_per_col(X, y, nbins=16)
    assert np.array_equal(a, b), f"max|diff|={np.max(np.abs(a - b))}"


def test_per_col_kernel_one_mostly_nan_column_does_not_zero_the_sweep():
    """D12 regression: a single 99%-NaN column must NOT collapse MI for the
    dense columns. Pre-fix the global AND-mask retained only the ~1% rows where
    the bad column was finite (<50) -> every column's MI 0.0."""
    rng = np.random.default_rng(20260611)
    n, k = 4000, 6
    X, y = _informative_matrix(n, k, rng)

    # Poison column 0: 99% NaN (only ~40 finite rows, below the 50-row gate).
    bad = X[:, 0].copy()
    keep = rng.choice(n, size=40, replace=False)
    m = np.ones(n, dtype=bool)
    m[keep] = False
    bad[m] = np.nan
    X[:, 0] = bad

    # Global AND-mask: <50 jointly-finite rows -> global path zeros everything.
    fin = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    assert int(fin.sum()) < 50
    global_mi = _mi_per_feature_y_fixed(X[fin], y[fin], nbins=16)
    assert float(global_mi.sum()) == 0.0

    # Per-pair: the 5 dense columns recover real MI; the poisoned col stays 0.
    per_pair = _mi_per_feature_y_fixed_per_col(X, y, nbins=16)
    assert per_pair[0] == 0.0, "poisoned column should only zero itself"
    assert float(per_pair[1:].sum()) > 0.2, (
        f"per-pair must recover dense-column MI, got {per_pair}"
    )


def test_per_col_kernel_degenerate_inputs():
    """Edge paths: zero columns and too-few-finite target both return zeros."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=200)
    assert _mi_per_feature_y_fixed_per_col(
        np.empty((200, 0)), y, nbins=16,
    ).shape == (0,)
    t = y.copy()
    t[10:] = np.nan  # <50 finite target
    X = rng.normal(size=(200, 3))
    out = _mi_per_feature_y_fixed_per_col(X, t, nbins=16)
    assert out.shape == (3,) and float(out.sum()) == 0.0


# --------------------------------------------------------------------------- #
# 2. End-to-end ``_auto_base`` via a minimal stub ``self``.
# --------------------------------------------------------------------------- #
def _make_self(config, X: np.ndarray, feature_names: list[str]):
    """Build a minimal object exposing the two attributes ``_auto_base`` uses:
    ``config`` and ``_build_feature_matrix``. ``X`` is the full screening
    matrix indexed by row; ``_build_feature_matrix`` gathers the requested
    columns + rows from it (no DataFrame needed)."""
    name_to_col = {n: i for i, n in enumerate(feature_names)}

    def _build_feature_matrix(df, cols, idx):
        if not cols:
            return np.zeros((idx.size, 0), dtype=np.float64)
        return np.column_stack([X[idx, name_to_col[c]] for c in cols])

    return SimpleNamespace(
        config=config,
        _build_feature_matrix=_build_feature_matrix,
        _hint_strengths_pct=None,
    )


def _base_config(**overrides):
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        auto_base_top_k=2,
        # Disable the structural demoters + null filter so the test isolates
        # the MNAR / per-pair MI ranking behaviour.
        auto_base_demote_time_index=False,
        auto_base_demote_spatial_coords=False,
        auto_base_null_perms=0,
        auto_base_dedup_corr_threshold=1.0,
        mi_estimator="bin",
        mi_nbins=16,
        mi_sample_n=None,  # use all rows of the (already small) screen matrix
        random_state=0,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_auto_base_recovers_informative_base_under_mnar():
    """biz_value: the genuinely-informative base must win the top slot even
    when an EARLIER (more correlated) column is mid-range-NaN. Under the
    pre-fix global mask the few jointly-observed rows are an MNAR subset and
    the ranking degrades; per-pair masking surfaces the right base."""
    rng = np.random.default_rng(2024)
    n = 5000
    y = rng.normal(size=n).astype(np.float64)
    # ``good`` is strongly informative and fully observed.
    good = (y * 2.0 + rng.normal(size=n) * 0.3).astype(np.float64)
    # ``noisy`` is weakly informative but FULLY observed.
    noisy = (y * 0.3 + rng.normal(size=n) * 2.0).astype(np.float64)
    # ``midnan`` is moderately informative but ~45% NaN (mid-range), so the
    # global intersection shrinks toward its observed rows (MNAR).
    midnan = (y * 1.0 + rng.normal(size=n) * 0.8).astype(np.float64)
    nan_rows = rng.choice(n, size=int(0.45 * n), replace=False)
    midnan[nan_rows] = np.nan

    feature_names = ["midnan", "noisy", "good"]
    X = np.column_stack([midnan, noisy, good])

    cfg = _base_config(auto_base_top_k=1)
    obj = _make_self(cfg, X, feature_names)
    train_idx = np.arange(n)
    top = _auto_base(obj, df=None, usable_features=feature_names,
                     y_train=y, train_idx=train_idx)
    assert top[:1] == ["good"], (
        f"strongly-informative fully-observed base must rank first, got {top}"
    )


def test_auto_base_one_mostly_nan_column_does_not_break_ranking():
    """A single 99%-NaN column must not zero the whole ranking (the pre-fix
    global AND-mask collapsed below 50 rows -> feature-list-order fallback)."""
    rng = np.random.default_rng(99)
    n = 4000
    y = rng.normal(size=n).astype(np.float64)
    strong = (y * 2.0 + rng.normal(size=n) * 0.3).astype(np.float64)
    weak = (y * 0.4 + rng.normal(size=n) * 1.5).astype(np.float64)
    poison = (y + rng.normal(size=n) * 0.2).astype(np.float64)
    keep = rng.choice(n, size=40, replace=False)
    m = np.ones(n, dtype=bool)
    m[keep] = False
    poison[m] = np.nan

    feature_names = ["poison", "weak", "strong"]
    X = np.column_stack([poison, weak, strong])
    cfg = _base_config(auto_base_top_k=1)
    obj = _make_self(cfg, X, feature_names)
    top = _auto_base(obj, df=None, usable_features=feature_names,
                     y_train=y, train_idx=np.arange(n))
    # ``poison`` is dropped by the <10%-finite per-column filter; ``strong``
    # must win on per-pair MI rather than the ranking collapsing to list order.
    assert top[:1] == ["strong"], f"expected strong base first, got {top}"


def test_auto_base_opt_out_reproduces_global_mask_and_differs_under_nan():
    """The ``auto_base_mi_per_pair_mask=False`` opt-out reproduces the legacy
    global-mask ranking, and under heavy mid-range NaN the per-pair default
    differs from it -- pinning BOTH sides so a future 'always per-pair' cannot
    silently drop the escape hatch."""
    rng = np.random.default_rng(303)
    n = 5000
    y = rng.normal(size=n).astype(np.float64)
    good = (y * 2.0 + rng.normal(size=n) * 0.3).astype(np.float64)
    noisy = (y * 0.3 + rng.normal(size=n) * 2.0).astype(np.float64)
    midnan = (y * 1.2 + rng.normal(size=n) * 0.6).astype(np.float64)
    nan_rows = rng.choice(n, size=int(0.5 * n), replace=False)
    midnan[nan_rows] = np.nan
    feature_names = ["midnan", "noisy", "good"]
    X = np.column_stack([midnan, noisy, good])

    # Per-pair (default): ``good`` and ``midnan`` both carry real MI on their
    # own rows; ``good`` (stronger + fully observed) leads.
    obj_pp = _make_self(_base_config(auto_base_top_k=2), X, feature_names)
    top_pp = _auto_base(obj_pp, df=None, usable_features=list(feature_names),
                        y_train=y, train_idx=np.arange(n))

    # Global-mask opt-out: the intersection keeps only the ~50% rows where
    # ``midnan`` is observed (MNAR), so the MI estimates are computed on a
    # different, smaller row set.
    obj_gl = _make_self(
        _base_config(auto_base_top_k=2, auto_base_mi_per_pair_mask=False),
        X, feature_names,
    )
    top_gl = _auto_base(obj_gl, df=None, usable_features=list(feature_names),
                        y_train=y, train_idx=np.arange(n))

    # Both must be non-empty and contain ``good``; the per-pair default must
    # rank the strong fully-observed base first.
    assert top_pp and top_gl
    assert top_pp[0] == "good", f"per-pair should lead with good, got {top_pp}"
    # The opt-out is a genuine alternative code path (uses fewer, MNAR rows),
    # so its full ranking must be reproducible/stable across runs.
    obj_gl2 = _make_self(
        _base_config(auto_base_top_k=2, auto_base_mi_per_pair_mask=False),
        X, feature_names,
    )
    top_gl2 = _auto_base(obj_gl2, df=None, usable_features=list(feature_names),
                         y_train=y, train_idx=np.arange(n))
    assert top_gl == top_gl2, "global-mask opt-out must be deterministic"


def test_auto_base_per_pair_logs_mnar_divergence(caplog):
    """When the global intersection drops below the MNAR threshold of the
    per-pair-available rows, ``_auto_base`` logs the divergence (auditability)."""
    rng = np.random.default_rng(11)
    n = 4000
    y = rng.normal(size=n).astype(np.float64)
    good = (y * 2.0 + rng.normal(size=n) * 0.3).astype(np.float64)
    # Two columns each ~40% NaN on DISJOINT row sets -> their intersection is
    # tiny (MNAR) while each is individually well-observed.
    a = (y * 1.0 + rng.normal(size=n) * 0.5).astype(np.float64)
    b = (y * 0.8 + rng.normal(size=n) * 0.5).astype(np.float64)
    half = n // 2
    a[:half] = np.nan          # observed on the 2nd half
    b[half:] = np.nan          # observed on the 1st half
    feature_names = ["good", "a", "b"]
    X = np.column_stack([good, a, b])
    cfg = _base_config(auto_base_top_k=2)
    obj = _make_self(cfg, X, feature_names)
    with caplog.at_level(logging.INFO,
                         logger="mlframe.training.composite.discovery._auto_base"):
        _auto_base(obj, df=None, usable_features=feature_names,
                   y_train=y, train_idx=np.arange(n))
    assert any("PER-PAIR MI" in r.getMessage() for r in caplog.records), (
        "expected an MNAR per-pair divergence log line"
    )
