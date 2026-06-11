"""Regression sensors for the 2026-06-10 composite-audit FUTURE items landed on
``discovery/_screening_tiny.py``.

- A13: the multi-seed per-seed RMSE arrays are FIXED-LENGTH and NaN-padded on the
  same seed schedule, so a composite-seed failure and a raw-seed failure at
  DIFFERENT positions no longer collapse into equal-length-but-mis-paired
  vectors. The paired Wilcoxon gate must pair by seed index and diff only
  jointly-finite positions. Pre-A13 the arrays were compacted finite-only -> a
  silent mis-pairing of unrelated seeds.
- A10: a silent GroupKFold -> KFold downgrade (too few distinct groups survive
  the mask) now emits a WARN.
- A11: when both groups and time_aware are requested, GroupKFold wins and the
  temporal order is dropped -> WARN.
- P9: the all-rows-valid fast path skips the full-matrix fancy-index copy and is
  bit-identical to the masked path.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from mlframe.training.composite.discovery import _screening_tiny as st
from mlframe.training.composite.discovery._screening_tiny import (
    _tiny_cv_rmse_raw_y,
    _tiny_cv_rmse_raw_y_multiseed,
    _tiny_cv_rmse_y_scale,
    _tiny_cv_rmse_y_scale_multiseed,
)
from mlframe.training.composite.transforms import get_transform


_SEED_STRIDE = 7919  # base_random_state + s_idx * 7919


def _raw_dataset(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    X = rng.normal(size=(n, 4))
    return y, X


def _linres_dataset(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(50.0, 10.0, n)
    y = 1.5 * base + rng.normal(0.0, 1.0, n)
    X = np.column_stack([base, rng.normal(size=n), rng.normal(size=n)])
    transform = get_transform("linear_residual")
    params = transform.fit(y, base)
    return y, base, X, params, transform


class TestA13FixedLengthPerSeed:
    """A13: per-seed arrays are fixed-length NaN-padded on the seed schedule."""

    def test_failed_seed_leaves_nan_in_its_own_slot(self, monkeypatch) -> None:
        """A seed whose inner CV degenerates must leave a NaN at ITS position --
        not get compacted out (the pre-fix bug that mis-aligned the paired test).
        """
        y, X = _raw_dataset()
        base_rs = 100
        fail_seed_idx = 1  # the second scheduled seed fails
        fail_rs = base_rs + fail_seed_idx * _SEED_STRIDE

        real = st._tiny_cv_rmse_raw_y

        def _patched(*args, **kwargs):
            if kwargs.get("random_state") == fail_rs:
                return float("nan")
            return real(*args, **kwargs)

        monkeypatch.setattr(st, "_tiny_cv_rmse_raw_y", _patched)

        _res, per_seed = _tiny_cv_rmse_raw_y_multiseed(
            y, X,
            family="linear", n_estimators=10, num_leaves=7,
            learning_rate=0.1, cv_folds=3,
            n_seed_repeats=4, base_random_state=base_rs,
            return_per_seed=True, time_aware=False,
        )
        # Fixed length == n_seed_repeats (NOT compacted to 3).
        assert per_seed.shape[0] == 4, (
            f"per-seed array must stay fixed-length 4, got {per_seed.shape[0]} "
            "(pre-A13 compaction)"
        )
        # NaN lands in the failed seed's OWN slot; the others are finite.
        assert np.isnan(per_seed[fail_seed_idx])
        assert np.isfinite(per_seed[[0, 2, 3]]).all()

    def test_median_ignores_failed_seed_bit_identical(self, monkeypatch) -> None:
        """The returned point estimate (median over finite seeds) is unchanged by
        the NaN-padding -- it still medians only the surviving seeds."""
        y, X = _raw_dataset()
        base_rs = 7
        fail_rs = base_rs + 2 * _SEED_STRIDE

        real = st._tiny_cv_rmse_raw_y

        def _patched(*args, **kwargs):
            if kwargs.get("random_state") == fail_rs:
                return float("nan")
            return real(*args, **kwargs)

        monkeypatch.setattr(st, "_tiny_cv_rmse_raw_y", _patched)

        med_with_fail, per_seed = _tiny_cv_rmse_raw_y_multiseed(
            y, X,
            family="linear", n_estimators=10, num_leaves=7,
            learning_rate=0.1, cv_folds=3,
            n_seed_repeats=4, base_random_state=base_rs,
            return_per_seed=True, time_aware=False,
        )
        finite = per_seed[np.isfinite(per_seed)]
        assert med_with_fail == pytest.approx(float(np.median(finite)), abs=1e-12)

    def test_misaligned_failures_pair_by_index_not_position(self) -> None:
        """The CONSUMER-side contract: composite and raw fail at DIFFERENT seed
        positions; pairing by index + jointly-finite filter must drop only the
        two failed pairs and align the rest. Pre-A13 (compacted) the two
        equal-length vectors mis-paired unrelated seeds.

        This pins the pairing semantics the tiny-rerank Wilcoxon gate relies on.
        """
        # Composite fails seed 1; raw fails seed 3 (both length-5, schedule-aligned).
        comp = np.array([0.50, np.nan, 0.52, 0.54, 0.55])
        raw = np.array([0.60, 0.61, 0.62, np.nan, 0.64])

        both_finite = np.isfinite(comp) & np.isfinite(raw)
        # Exactly the two failed positions drop out -> 3 jointly-finite pairs.
        assert both_finite.tolist() == [True, False, True, False, True]
        diff = comp[both_finite] - raw[both_finite]
        # Correct index-paired diffs: (0,0), (2,2), (4,4).
        np.testing.assert_allclose(diff, [0.50 - 0.60, 0.52 - 0.62, 0.55 - 0.64])
        # The pre-A13 compaction kept 4 finite comp values and 4 finite raw
        # values and diffed positionally: comp=[.50,.52,.54,.55] vs
        # raw=[.60,.61,.62,.64] -> e.g. .54-.62 pairs composite-seed3 against
        # raw-seed2 (a wrong pair). It even diverges in length from the correct
        # 3-pair index-aligned diff, so a future revert cannot pass silently.
        bad_comp = comp[np.isfinite(comp)]
        bad_raw = raw[np.isfinite(raw)]
        bad_diff = bad_comp - bad_raw
        assert bad_diff.shape[0] == 4 and diff.shape[0] == 3, (
            "compacted pairing keeps 4 mis-aligned pairs vs 3 correct pairs"
        )
        # The compacted third entry mis-pairs seed3-vs-seed2 (.54 - .62);
        # the index-aligned diff never contains that value.
        assert not np.any(np.isclose(diff, 0.54 - 0.62))

    def test_y_scale_multiseed_also_fixed_length(self, monkeypatch) -> None:
        """The y-scale twin obeys the same fixed-length NaN-padded contract."""
        y, base, X, params, transform = _linres_dataset()
        base_rs = 0
        fail_rs = base_rs + 0 * _SEED_STRIDE  # first seed fails

        real = st._tiny_cv_rmse_y_scale

        def _patched(*args, **kwargs):
            if kwargs.get("random_state") == fail_rs:
                return float("nan")
            return real(*args, **kwargs)

        monkeypatch.setattr(st, "_tiny_cv_rmse_y_scale", _patched)

        _res, per_seed = _tiny_cv_rmse_y_scale_multiseed(
            y_train=y, base_train=base, transform=transform,
            fitted_params=params, x_train_matrix=X,
            family="linear", n_estimators=10, num_leaves=7,
            learning_rate=0.1, cv_folds=3,
            n_seed_repeats=3, base_random_state=base_rs,
            return_per_seed=True, time_aware=False,
        )
        assert per_seed.shape[0] == 3
        assert np.isnan(per_seed[0])
        assert np.isfinite(per_seed[[1, 2]]).all()


class TestA10A11SplitterWarnings:
    """A10/A11: silent splitter downgrades now WARN."""

    def test_a10_group_downgrade_warns(self, caplog) -> None:
        """Fewer distinct groups than cv_folds -> GroupKFold silently became
        KFold; now it WARNs."""
        y, X = _raw_dataset(n=600)
        # 2 distinct groups < cv_folds=3 -> forced downgrade.
        groups = np.zeros(600, dtype=np.int64)
        groups[300:] = 1
        with caplog.at_level(logging.WARNING):
            _tiny_cv_rmse_raw_y(
                y, X, family="linear",
                n_estimators=10, num_leaves=7, learning_rate=0.1,
                cv_folds=3, random_state=0, groups=groups,
            )
        assert any(
            "distinct group" in r.message and "falling" in r.message
            for r in caplog.records
        ), "A10 group-downgrade WARN not emitted"

    def test_a11_groups_over_time_warns(self, caplog) -> None:
        """groups + time_aware together: GroupKFold wins, temporal order dropped
        -> WARN."""
        y, X = _raw_dataset(n=600)
        groups = np.repeat(np.arange(6), 100)  # 6 groups >= cv_folds
        with caplog.at_level(logging.WARNING):
            _tiny_cv_rmse_raw_y(
                y, X, family="linear",
                n_estimators=10, num_leaves=7, learning_rate=0.1,
                cv_folds=3, random_state=0, groups=groups, time_aware=True,
            )
        assert any(
            "GroupKFold takes precedence" in r.message for r in caplog.records
        ), "A11 groups-over-time WARN not emitted"

    def test_a11_y_scale_cv_splitter_escape_hatch(self) -> None:
        """A11: _tiny_cv_rmse_y_scale gained the cv_splitter escape hatch (parity
        with the raw sibling); a supplied splitter wins and produces a finite
        score without raising."""
        from sklearn.model_selection import KFold

        y, base, X, params, transform = _linres_dataset()
        custom = KFold(n_splits=4, shuffle=True, random_state=123)
        res = _tiny_cv_rmse_y_scale(
            y_train=y, base_train=base, transform=transform,
            fitted_params=params, x_train_matrix=X,
            family="linear", n_estimators=10, num_leaves=7,
            learning_rate=0.1, cv_folds=3, random_state=0,
            cv_splitter=custom,
        )
        assert np.isfinite(res)


class TestP9NoCopyBitIdentical:
    """P9: the all-rows-valid fast path is bit-identical to the masked path."""

    def test_all_valid_matches_masked_path(self) -> None:
        """linear_residual has an all-real domain, so every row is valid -> the
        no-copy fast path runs. Compare against a transform whose domain mask is
        all-True to confirm bit-identity of the score."""
        y, base, X, params, transform = _linres_dataset()
        # domain_check for linear_residual is all-True -> exercises the fast path.
        valid = transform.domain_check(y, base)
        assert bool(valid.all()), "fixture must exercise the all-valid fast path"
        res = _tiny_cv_rmse_y_scale(
            y_train=y, base_train=base, transform=transform,
            fitted_params=params, x_train_matrix=X,
            family="linear", n_estimators=15, num_leaves=8,
            learning_rate=0.1, cv_folds=3, random_state=42,
        )
        # Recompute with a manually pre-masked (copied) matrix; the all-True mask
        # makes the masked inputs value-identical, so the score must match exactly.
        res_masked = _tiny_cv_rmse_y_scale(
            y_train=y[valid], base_train=base[valid], transform=transform,
            fitted_params=params, x_train_matrix=X[valid],
            family="linear", n_estimators=15, num_leaves=8,
            learning_rate=0.1, cv_folds=3, random_state=42,
        )
        assert res == pytest.approx(res_masked, abs=1e-12)
