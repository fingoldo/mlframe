"""Sequential multiseed early-stop (``enable_multiseed_early_stop`` / ``seed_early_stop_threshold``)
correctness: an early-stopped candidate's returned score may differ numerically from the all-seeds
score (fewer seeds contributed to the median), but the ACCEPT/REJECT decision -- and therefore the
kept-spec set and the ranking among survivors -- must never flip relative to running all seeds.

Safety argument under test (see ``_seed_median_lower_bound`` in ``_screening_tiny.py`` and the
``enable_multiseed_early_stop`` docstring in ``_composite_target_discovery_config_base.py``):
RMSE-like seed scores are non-negative, so padding the not-yet-run seeds with hypothetical zeros can
only ever pull the running median DOWN. ``median(observed + [0]*remaining)`` is therefore a rigorous
LOWER BOUND on the eventual full-sample median. Early-stop only fires when that lower bound already
clears (>=) the comparison threshold -- i.e. the candidate is *already a guaranteed reject* no matter
what the unrun seeds turn out to be. Consequences directly tested here:

  1. A clear WINNER (score well under threshold) can never trigger the bound -- ON and OFF must run
     the identical seed schedule and return a bit-identical per-seed array / median.
  2. A clear LOSER can trigger the bound -- ON may run fewer seeds than OFF, but whenever it *does*
     stop early, its returned median must still be >= threshold, i.e. still a reject, matching OFF's
     reject verdict (both reject, exact rejected-candidate scores need not match).
  3. A BORDERLINE/noisy case may or may not trigger the bound seed-by-seed; whichever happens, the
     accept/reject verdict (score >= threshold ?) is asserted identical between ON and OFF.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.discovery.screening import _tiny_cv_rmse_y_scale_multiseed
from mlframe.training.composite.transforms import get_transform


def _make_dataset(rng: np.random.Generator, n: int, noise_scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(y, base, x)`` for a near-linear ``y = 2*base + 3 + noise`` DGP, noise scaled by ``noise_scale``."""
    base = rng.normal(20.0, 5.0, n)
    y = 2.0 * base + 3.0 + rng.standard_normal(n) * noise_scale
    x = np.column_stack([base, rng.standard_normal(n), rng.standard_normal(n)])
    return y, base, x


def _run(y, base, x, *, threshold: float, n_seed_repeats: int = 6, random_state: int = 0):
    """``(rmse, per_seed_scores)`` from a multiseed tiny-CV rerank with the given early-stop threshold."""
    transform = get_transform("linear_residual")
    params = transform.fit(y, base)
    return _tiny_cv_rmse_y_scale_multiseed(
        y_train=y, base_train=base, transform=transform, fitted_params=params,
        x_train_matrix=x,
        family="linear",
        n_estimators=10, num_leaves=8, learning_rate=0.1,
        cv_folds=3, n_jobs=1,
        n_seed_repeats=n_seed_repeats,
        base_random_state=random_state,
        return_per_seed=True,
        cv_selector_mode="mean",
        seed_early_stop_threshold=threshold,
    )


class TestClearWinnerUnaffected:
    """Early-stop must be a pure no-op when the score never approaches the threshold."""

    def test_winner_bit_identical_on_vs_off(self) -> None:
        """Score far below threshold: the lower bound can never clear it, so early-stop never fires --
        ON and OFF must run the identical seed schedule and return identical results."""
        rng = np.random.default_rng(1)
        y, base, x = _make_dataset(rng, n=400, noise_scale=1.0)
        rmse_off, seeds_off = _run(y, base, x, threshold=float("inf"))
        # Threshold set far above any plausible RMSE for this near-linear low-noise fit.
        rmse_on, seeds_on = _run(y, base, x, threshold=1e6)
        assert rmse_off == pytest.approx(rmse_on, abs=1e-9)
        np.testing.assert_allclose(seeds_off, seeds_on, equal_nan=True)


class TestClearLoserDecisionPreserved:
    """Early-stop may cut seeds short, but must never flip the accept/reject verdict."""

    def test_loser_early_stops_but_reject_verdict_matches(self) -> None:
        """Score far above a tiny threshold: the lower bound clears immediately -- ON should stop
        after fewer seeds than OFF, but BOTH must land on the reject side of the threshold."""
        rng = np.random.default_rng(2)
        # Heavy noise relative to signal -> high, unstable RMSE.
        y, base, x = _make_dataset(rng, n=200, noise_scale=200.0)
        threshold = 0.01  # Unreachable by construction; forces the reject verdict either way.
        rmse_off, seeds_off = _run(y, base, x, threshold=float("inf"))
        rmse_on, seeds_on = _run(y, base, x, threshold=threshold)
        n_seeds_run_off = int(np.isfinite(seeds_off).sum())
        n_seeds_run_on = int(np.isfinite(seeds_on).sum())
        assert n_seeds_run_on <= n_seeds_run_off
        assert n_seeds_run_on < len(seeds_on), "expected the early-stop to actually fire in this scenario"
        # Same verdict: both reject (score clears the threshold).
        assert rmse_off >= threshold
        assert rmse_on >= threshold


class TestEndToEndKeptSpecParity:
    """``CompositeTargetDiscovery.fit()`` with ``enable_multiseed_early_stop`` OFF vs ON must produce
    IDENTICAL kept-spec name sets in IDENTICAL rank order, across a clear-winner, a clear-loser, and a
    borderline/noisy DGP -- the borderline case is the one most likely to expose a stopping-rule bug
    since its per-seed scores straddle the raw-baseline gate threshold."""

    @staticmethod
    def _config(enable_early_stop: bool):
        """Fast tiny-model discovery config with ``enable_multiseed_early_stop`` toggled per-call."""
        from mlframe.training.configs import CompositeTargetDiscoveryConfig

        return CompositeTargetDiscoveryConfig(
            enabled=True,
            screening="tiny_model",
            tiny_screening_models="single_lgbm",
            tiny_model_n_estimators=15,
            tiny_model_cv_folds=3,
            tiny_model_sample_n=800,
            tiny_model_n_seed_repeats=5,
            top_m_after_tiny=5,
            require_beats_raw_baseline=True,
            raw_baseline_tolerance=1.02,
            honest_oof_selection=False,  # keep the pre-computed early-stop threshold path deterministic
            use_wilcoxon_gate=False,
            enable_multiseed_early_stop=enable_early_stop,
            random_state=0,
        )

    @classmethod
    def _fit_names(cls, df, enable_early_stop: bool) -> list[str]:
        """Kept-spec names, in rank order, from a full ``fit()`` run with early-stop ON/OFF."""
        from mlframe.training.composite import CompositeTargetDiscovery

        disc = CompositeTargetDiscovery(cls._config(enable_early_stop))
        feature_cols = [c for c in df.columns if c != "y"]
        disc.fit(df, target_col="y", feature_cols=feature_cols, train_idx=np.arange(len(df)))
        return [s.name for s in disc.specs_]

    @staticmethod
    def _clear_winner_df(n: int = 1500, seed: int = 0):
        """Strong near-linear base signal -- the composite spec should clearly win."""
        import pandas as pd

        rng = np.random.default_rng(seed)
        base = rng.normal(loc=10.0, scale=3.0, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 0.97 * base + 0.4 * x1 - 0.2 * x2 + rng.normal(scale=0.15, size=n)
        return pd.DataFrame({"base_col": base, "x1": x1, "x2": x2, "y": y})

    @staticmethod
    def _clear_loser_df(n: int = 1500, seed: int = 1):
        """No structural relation between ``y`` and any candidate column -- no spec should survive."""
        import pandas as pd

        rng = np.random.default_rng(seed)
        base = rng.normal(loc=5.0, scale=2.0, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = rng.normal(scale=1.0, size=n)  # no structural relation to base/x1/x2
        return pd.DataFrame({"base_col": base, "x1": x1, "x2": x2, "y": y})

    @staticmethod
    def _borderline_df(n: int = 1500, seed: int = 2):
        """Weak base signal drowned in noise -- per-seed scores straddle the raw-baseline gate threshold."""
        import pandas as pd

        rng = np.random.default_rng(seed)
        base = rng.normal(loc=5.0, scale=2.0, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        y = 0.15 * base + 0.05 * x1 + rng.normal(scale=1.0, size=n)
        return pd.DataFrame({"base_col": base, "x1": x1, "x2": x2, "y": y})

    @pytest.mark.parametrize(
        "make_df,label",
        [
            (_clear_winner_df.__func__, "clear_winner"),
            (_clear_loser_df.__func__, "clear_loser"),
            (_borderline_df.__func__, "borderline"),
        ],
    )
    def test_kept_specs_and_ranking_identical(self, make_df, label) -> None:
        """The kept-spec name set and rank order must be identical with early-stop ON vs OFF."""
        df = make_df()
        names_off = self._fit_names(df, enable_early_stop=False)
        names_on = self._fit_names(df, enable_early_stop=True)
        assert names_on == names_off, f"[{label}] enable_multiseed_early_stop changed the kept-spec set/ranking: " f"OFF={names_off} ON={names_on}"


class TestBorderlineVerdictParity:
    """Across many noisy-DGP/threshold combinations, early-stop must never flip the accept/reject verdict."""

    def test_borderline_verdict_matches_across_many_thresholds_and_seeds(self) -> None:
        """Sweep several noisy DGP realizations and thresholds straddling the typical score. Whatever
        each run's ON/OFF seed count turns out to be, the accept/reject verdict (score >= threshold)
        must agree -- this is the invariant the lower-bound proof guarantees, independent of how noisy
        or ambiguous the per-seed scores are."""
        mismatches = []
        for trial_seed in range(8):
            rng = np.random.default_rng(100 + trial_seed)
            y, base, x = _make_dataset(rng, n=250, noise_scale=15.0)
            rmse_off, _ = _run(y, base, x, threshold=float("inf"), random_state=trial_seed)
            # Threshold set at the OFF-run's own score: the case most likely to sit right on the
            # boundary where a naive/unsound early-stop rule could flip the verdict.
            threshold = rmse_off
            rmse_on, seeds_on = _run(y, base, x, threshold=threshold, random_state=trial_seed)
            verdict_off = rmse_off >= threshold  # True by construction (threshold == rmse_off).
            verdict_on = rmse_on >= threshold
            if verdict_off != verdict_on:
                mismatches.append((trial_seed, rmse_off, rmse_on, threshold, seeds_on.tolist()))
        assert not mismatches, f"early-stop flipped the accept/reject verdict: {mismatches}"
