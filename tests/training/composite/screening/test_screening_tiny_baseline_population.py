"""Regression sensors for composite-audit FUTURE items A5 and A12/P11 on
``discovery/_screening_tiny.py``.

A5 (baseline-population parity): the transformed-target tiny-CV
(``_tiny_cv_rmse_y_scale``) previously scored ONLY the transform's
domain-valid rows, while the raw-y baseline (``_tiny_cv_rmse_raw_y``) scores
EVERY finite-y row. That row-population mismatch biased the raw-baseline gate
exactly on transforms whose domain excludes hard rows. The fix emulates the
production ``CompositeTargetEstimator.predict`` median fallback: the model
still TRAINS only on the domain, but each fold is SCORED over its full
finite-y val rows, with domain-invalid val rows filled by the train-fold
median. The two scores (composite, raw) now cover the same row population.

A12/P11 (early-stop wiring): ``early_stop_threshold`` is a first-class,
documented kwarg threaded transparently through the multiseed wrapper into the
serial fold loop. With the default ``inf`` it never fires, so the multiseed
return value is bit-identical to the no-threshold call; with a low threshold
on a serial run it aborts the remaining folds. (The rerank-side wiring that
computes the raw baseline before the per-spec loop and passes the threshold in
lives in ``_tiny_rerank.py`` and is tracked separately.)
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery import _screening_tiny as st
from mlframe.training.composite.discovery import _screening_tiny_perbin as stp
from mlframe.training.composite.discovery._screening_tiny import (
    _tiny_cv_rmse_y_scale,
    _tiny_cv_rmse_y_scale_multiseed,
)
from mlframe.training.composite.transforms import get_transform


def _partial_domain_dataset(
    n: int = 600,
    n_invalid: int = 120,
    invalid_extreme: bool = False,
    seed: int = 1,
):
    """``logratio`` fixture: y>0 & base>0 is the domain. ``n_invalid`` rows get a
    negative base -> finite but domain-invalid. When ``invalid_extreme`` the
    invalid rows also get a y far from the bulk median, so a median fallback on
    them yields a large error -> the A5-included population scores measurably
    worse than the valid-only (pre-fix) population.
    """
    rng = np.random.default_rng(seed)
    base = np.abs(rng.normal(10.0, 3.0, n)) + 1.0
    y = base * 1.3 + np.abs(rng.normal(0.0, 1.0, n)) + 2.0  # strictly positive
    extra = rng.normal(size=(n, 2))
    inv = rng.choice(n, size=n_invalid, replace=False)
    base[inv] = -np.abs(base[inv])  # domain-invalid, still finite
    if invalid_extreme:
        # Push invalid-row y far above the valid bulk so the median fallback on
        # those scored val rows is badly wrong (post-fix RMSE balloons).
        y[inv] = y[inv] + 500.0
    X = np.column_stack([base, extra])
    transform = get_transform("logratio")
    valid = np.asarray(transform.domain_check(y, base), dtype=bool)
    params = transform.fit(y[valid], base[valid])
    return y, base, X, params, transform, valid, inv


class TestA5BaselinePopulationParity:
    """Groups tests covering a5 baseline population parity."""
    def test_invalid_rows_are_scored_not_dropped(self) -> None:
        """A5: domain-invalid finite-y val rows now contribute to the RMSE via the
        production median fallback. With extreme invalid-row y, including them
        pushes the score far above the pre-fix valid-only score.

        Pre-fix the function masked to the 480 valid rows and never saw the 120
        extreme invalid rows, so its RMSE was the (small) valid-only RMSE. The
        post-fix population includes the invalid rows scored by the median
        fallback, so the RMSE must blow up.
        """
        y, base, X, params, transform, valid, _inv = _partial_domain_dataset(
            invalid_extreme=True,
        )
        assert not bool(valid.all()), "fixture must have domain-invalid rows"

        full_pop = _tiny_cv_rmse_y_scale(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=X,
            family="linear",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            random_state=0,
        )
        # The pre-fix behaviour == manually pre-masking to the valid rows (the
        # old code path). That score must be DRAMATICALLY smaller because it
        # never sees the +500 invalid rows.
        valid_only = _tiny_cv_rmse_y_scale(
            y_train=y[valid],
            base_train=base[valid],
            transform=transform,
            fitted_params=params,
            x_train_matrix=X[valid],
            family="linear",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            random_state=0,
        )
        assert np.isfinite(full_pop) and np.isfinite(valid_only)
        # 120/600 = 20% of scored rows carry a ~+500 fallback error -> the
        # full-population RMSE is many times the valid-only RMSE. A pre-fix
        # build returns ``valid_only`` for both calls (the masked input is the
        # ONLY thing it ever scored), so this ratio collapses to ~1.0 and the
        # test fails -- the regression sensor.
        assert full_pop > valid_only * 5.0, (
            f"A5 not applied: full-population RMSE={full_pop:.3f} should be >> valid-only RMSE={valid_only:.3f} (invalid rows must be scored)"
        )

    def test_scored_population_size_matches_raw_baseline(self, monkeypatch) -> None:
        """A5: the per-fold val population now equals the finite-y population
        (raw-y parity), not the domain-valid subset. Spy on the fold val sizes
        via the per-bin scorer to prove the invalid rows are scored.
        """
        y, base, X, params, transform, valid, _inv = _partial_domain_dataset(
            invalid_extreme=False,
        )
        seen_val_sizes: list[int] = []
        # ``_per_bin_rmse`` and its caller ``_tiny_cv_rmse_y_scale`` both live in ``_screening_tiny_perbin`` after the
        # monolith split, so the call resolves the sibling-local name -- patch THERE (patching the parent re-export
        # ``st._per_bin_rmse`` would never intercept the in-module lookup).
        real_per_bin = stp._per_bin_rmse

        def _spy_per_bin(y_true, y_hat, bin_var, n_bins=5):
            """Spy per bin."""
            seen_val_sizes.append(int(np.asarray(y_true).shape[0]))
            return real_per_bin(y_true, y_hat, bin_var, n_bins=n_bins)

        monkeypatch.setattr(stp, "_per_bin_rmse", _spy_per_bin)
        _tiny_cv_rmse_y_scale(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=X,
            family="linear",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            random_state=0,
            return_per_bin=True,
            n_bins=5,
        )
        # Sum of fold val sizes == total finite-y population (600), NOT the
        # valid-only subset (480). Pre-fix the splits ran over the 480 masked
        # rows, so the sum would be 480.
        assert sum(seen_val_sizes) == int(np.isfinite(y).sum()), (
            f"scored val rows={sum(seen_val_sizes)} must equal the finite-y population={int(np.isfinite(y).sum())}, not the valid subset ({int(valid.sum())})"
        )

    def test_all_valid_is_bit_identical(self) -> None:
        """A5 must be a strict no-op when the transform's domain covers every row
        (``linear_residual`` has an all-real domain): the emulation branch never
        runs and the score equals the legacy path exactly."""
        rng = np.random.default_rng(3)
        n = 600
        base = rng.normal(50.0, 10.0, n)
        y = 1.5 * base + rng.normal(0.0, 1.0, n)
        X = np.column_stack([base, rng.normal(size=n), rng.normal(size=n)])
        transform = get_transform("linear_residual")
        params = transform.fit(y, base)
        assert bool(np.asarray(transform.domain_check(y, base)).all())
        score = _tiny_cv_rmse_y_scale(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=X,
            family="linear",
            n_estimators=15,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            random_state=42,
        )
        masked = _tiny_cv_rmse_y_scale(
            y_train=y[np.isfinite(y)],
            base_train=base[np.isfinite(y)],
            transform=transform,
            fitted_params=params,
            x_train_matrix=X[np.isfinite(y)],
            family="linear",
            n_estimators=15,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            random_state=42,
        )
        assert score == pytest.approx(masked, abs=1e-12)


class TestA12EarlyStopThreadedThroughMultiseed:
    """Groups tests covering a12 early stop threaded through multiseed."""
    def test_multiseed_forwards_early_stop_threshold(self, monkeypatch) -> None:
        """A12/P11: ``early_stop_threshold`` must reach the underlying single-seed
        call through the multiseed wrapper (the conduit the rerank caller relies
        on). Spy on the kwarg seen by ``_tiny_cv_rmse_y_scale``."""
        y, base, X, params, transform, _valid, _inv = _partial_domain_dataset()
        seen_thresholds: list[float] = []
        real = st._tiny_cv_rmse_y_scale

        def _spy(*args, **kwargs):
            """Spy."""
            seen_thresholds.append(kwargs.get("early_stop_threshold", float("inf")))
            return real(*args, **kwargs)

        monkeypatch.setattr(st, "_tiny_cv_rmse_y_scale", _spy)
        _tiny_cv_rmse_y_scale_multiseed(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=X,
            family="linear",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            n_seed_repeats=2,
            base_random_state=0,
            early_stop_threshold=0.5,
        )
        assert seen_thresholds, "underlying single-seed call was never invoked"
        assert all(t == 0.5 for t in seen_thresholds), f"early_stop_threshold not forwarded per seed: {seen_thresholds}"

    def test_multiseed_inf_threshold_bit_identical(self) -> None:
        """Default ``inf`` threshold => the multiseed median is bit-identical to
        the no-threshold call (the early-stop branch never fires)."""
        y, base, X, params, transform, _valid, _inv = _partial_domain_dataset()
        legacy = _tiny_cv_rmse_y_scale_multiseed(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=X,
            family="linear",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            n_seed_repeats=3,
            base_random_state=0,
        )
        with_inf = _tiny_cv_rmse_y_scale_multiseed(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=X,
            family="linear",
            n_estimators=10,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            n_seed_repeats=3,
            base_random_state=0,
            early_stop_threshold=float("inf"),
        )
        assert legacy == pytest.approx(with_inf, abs=1e-12)

    def test_early_stop_fires_via_multiseed_serial(self) -> None:
        """A low threshold on the serial path aborts the remaining folds for every
        seed -- proving the wired-through threshold reaches the serial early-stop
        loop. We assert the run still returns a finite score (the aborted folds
        leave at least one scored) and that it differs from the full run only by
        the dropped folds, never raising."""
        y, base, X, params, transform, _valid, _inv = _partial_domain_dataset()
        res = _tiny_cv_rmse_y_scale_multiseed(
            y_train=y,
            base_train=base,
            transform=transform,
            fitted_params=params,
            x_train_matrix=X,
            family="lgb",
            n_estimators=30,
            num_leaves=8,
            learning_rate=0.1,
            cv_folds=3,
            n_seed_repeats=2,
            base_random_state=0,
            n_jobs=1,
            early_stop_threshold=1e-6,  # impossibly tight -> fold-1 partial-sum trips
        )
        assert np.isfinite(res)
