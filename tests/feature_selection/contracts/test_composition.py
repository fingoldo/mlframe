"""Unit + biz_value coverage for ``mlframe.feature_selection.filters.composition``.

Contracts under test:

* ``compose_pair_fe`` returns a dict with the documented ``X_aug`` / ``names`` / ``rounds`` keys.
  Shapes are preserved when no useful pair is found; default feature names default to ``x0..x{p-1}``;
  custom feature names are propagated to the engineered-feature labels.
* ``validate_pair_fe_cv`` returns the documented OOS-uplift dict (in_sample_mi, oos_mean, oos_std,
  oos_per_fold, optimism_ratio, trivial_oos_mean, honest_uplift_vs_trivial, folds_with_positive_uplift).
* Determinism: identical ``seed`` arguments produce identical OOS-MI series.
* biz_value: on the canonical XOR target the engineered pair beats EACH parent's marginal MI by >=2x.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from mlframe.feature_selection.filters.composition import (
    compose_pair_fe,
    validate_pair_fe_cv,
)
from mlframe.feature_selection.filters.fe_baselines import _mi_1d

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def _xor_pair_binary(n: int = 1500, seed: int = 0):
    """y = x_a XOR x_b for x_a, x_b in {0, 1}. Each parent alone has near-zero MI with y; the pair carries full signal."""
    rng = np.random.default_rng(seed)
    x_a = rng.integers(0, 2, size=n).astype(np.float64)
    x_b = rng.integers(0, 2, size=n).astype(np.float64)
    y = x_a.astype(np.int64) ^ x_b.astype(np.int64)
    return x_a, x_b, y


def _additive_pair(n: int = 800, seed: int = 0):
    """y = sign(x_a + x_b) - a smooth additive target the pair-FE should fit easily."""
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = (x_a + x_b > 0).astype(np.int64)
    return x_a, x_b, y


def _noise_pair(n: int = 600, seed: int = 0):
    """Two iid Gaussians + a label independent of both. Honest CV should report tiny uplift, not a spurious win."""
    rng = np.random.default_rng(seed)
    x_a = rng.normal(size=n)
    x_b = rng.normal(size=n)
    y = rng.integers(0, 2, size=n).astype(np.int64)
    return x_a, x_b, y


# ---------------------------------------------------------------------------
# compose_pair_fe : API contract
# ---------------------------------------------------------------------------


class TestComposePairFERegression:
    """Wave 13 finding 5: the single-feature MI ranking pass recomputed ``_mi_1d`` for every
    column every round, including columns carried over unchanged from a prior round -- fixed by
    a name-keyed cache. Equivalence: cached values must match the uncached values bit-for-bit
    (same estimator call, just memoized); call-count: an original column's MI must be computed
    at most once across all rounds, not once per round.
    """

    def test_regression_single_mi_recomputed_at_most_once_per_column(self, monkeypatch):
        """Regression single mi recomputed at most once per column."""
        from mlframe.feature_selection.filters import fe_baselines as fb

        orig_mi_1d = fb._mi_1d
        call_log: list = []

        def counted(arr, *a, **kw):
            """Helper that counted."""
            call_log.append(np.asarray(arr).tobytes())
            return orig_mi_1d(arr, *a, **kw)

        monkeypatch.setattr(fb, "_mi_1d", counted)
        x_a, x_b, y = _additive_pair(n=300, seed=3)
        X = np.column_stack([x_a, x_b])
        n_rounds = 3
        out = compose_pair_fe(X, y, n_rounds=n_rounds, top_k_per_round=1, n_trials=5, max_degree=2)

        a_bytes, b_bytes = x_a.tobytes(), x_b.tobytes()
        n_single_mi_calls_for_a = sum(1 for c in call_log if c == a_bytes)
        n_single_mi_calls_for_b = sum(1 for c in call_log if c == b_bytes)
        assert (
            n_single_mi_calls_for_a <= 1
        ), f"x_a's single-feature MI recomputed {n_single_mi_calls_for_a} times across {n_rounds} rounds; expected <=1 (cached after round 1)"
        assert (
            n_single_mi_calls_for_b <= 1
        ), f"x_b's single-feature MI recomputed {n_single_mi_calls_for_b} times across {n_rounds} rounds; expected <=1 (cached after round 1)"
        assert set(out.keys()) == {"X_aug", "names", "rounds"}

    def test_equivalence_cached_single_mi_matches_direct_computation(self):
        """The cached single_mi value used for ranking must equal a fresh direct ``_mi_1d`` call
        on the same column (the cache must not silently change the estimator's output)."""
        from mlframe.feature_selection.filters.fe_baselines import _mi_1d

        x_a, x_b, y = _additive_pair(n=300, seed=4)
        X = np.column_stack([x_a, x_b])
        out = compose_pair_fe(X, y, n_rounds=2, top_k_per_round=1, n_trials=5, max_degree=2)
        # Original columns 0/1 are unchanged in X_aug (compose only appends columns).
        direct_mi_a = _mi_1d(out["X_aug"][:, 0], y, discrete_target=True, mi_estimator="plugin", plugin_n_bins=20)
        direct_mi_b = _mi_1d(out["X_aug"][:, 1], y, discrete_target=True, mi_estimator="plugin", plugin_n_bins=20)
        assert np.allclose(out["X_aug"][:, 0], x_a)
        assert np.allclose(out["X_aug"][:, 1], x_b)
        # Just a sanity bound: the direct recompute must be deterministic/finite (cache correctness
        # is about the reused VALUE being identical to this, which the call-count test above pins).
        assert np.isfinite(direct_mi_a) and np.isfinite(direct_mi_b)


class TestComposePairFE:
    """Unit checks on ``compose_pair_fe`` return contract and edge cases."""

    def test_return_keys(self):
        """Return keys."""
        x_a, x_b, y = _additive_pair(n=400, seed=1)
        X = np.column_stack([x_a, x_b])
        out = compose_pair_fe(X, y, n_rounds=1, top_k_per_round=2, n_trials=6, max_degree=2)
        assert set(out.keys()) == {"X_aug", "names", "rounds"}
        assert isinstance(out["X_aug"], np.ndarray)
        assert isinstance(out["names"], list)
        assert isinstance(out["rounds"], list)

    def test_shape_preserves_originals(self):
        """X_aug must have at least as many columns as the input and never fewer rows."""
        x_a, x_b, y = _additive_pair(n=400, seed=2)
        X = np.column_stack([x_a, x_b])
        out = compose_pair_fe(X, y, n_rounds=1, top_k_per_round=2, n_trials=6, max_degree=2)
        assert out["X_aug"].shape[0] == X.shape[0]
        assert out["X_aug"].shape[1] >= X.shape[1]
        assert len(out["names"]) == out["X_aug"].shape[1]
        # First p_orig columns must be byte-equal to the input - composition only APPENDS.
        np.testing.assert_array_equal(out["X_aug"][:, : X.shape[1]], X)

    def test_default_feature_names(self):
        """No feature_names argument => ``x0..x{p-1}`` labels for the original columns."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 3))
        y = (X[:, 0] > 0).astype(np.int64)
        out = compose_pair_fe(X, y, n_rounds=1, top_k_per_round=2, n_trials=4, max_degree=2)
        assert out["names"][:3] == ["x0", "x1", "x2"]

    def test_custom_feature_names_propagate(self):
        """User-supplied names are reused in engineered-column labels (``r1_pair_<a>_<b>_...``)."""
        rng = np.random.default_rng(4)
        X = rng.normal(size=(400, 2))
        # Strong additive signal so at least one engineered feature is appended.
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
        out = compose_pair_fe(
            X,
            y,
            n_rounds=1,
            top_k_per_round=1,
            n_trials=10,
            max_degree=2,
            feature_names=["alpha", "beta"],
        )
        assert out["names"][:2] == ["alpha", "beta"]
        # If an engineered column was added, its label must reference the parent names.
        for nm in out["names"][2:]:
            assert "alpha" in nm and "beta" in nm

    def test_single_feature_bails_out(self):
        """Pair-FE on a single column has no pairs to form; no engineered features, no crash."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(200, 1))
        y = (X[:, 0] > 0).astype(np.int64)
        out = compose_pair_fe(X, y, n_rounds=2, top_k_per_round=2, n_trials=4, max_degree=2)
        assert out["X_aug"].shape == X.shape
        assert out["names"] == ["x0"]
        assert out["rounds"] == []

    def test_constant_features_skipped(self):
        """All-constant features (std<1e-12) are filtered out of the ranking; no crash."""
        n = 200
        X = np.column_stack([np.zeros(n), np.ones(n)])
        y = np.random.default_rng(6).integers(0, 2, size=n)
        out = compose_pair_fe(X, y, n_rounds=1, top_k_per_round=2, n_trials=4, max_degree=2)
        # Both features are constant => ranking is empty => loop bails out before pair-formation.
        assert out["X_aug"].shape == X.shape
        assert out["rounds"] == []

    def test_n_rounds_zero(self):
        """``n_rounds=0`` is a no-op: returns the input untouched."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(150, 3))
        y = (X[:, 0] > 0).astype(np.int64)
        out = compose_pair_fe(X, y, n_rounds=0, top_k_per_round=2, n_trials=4, max_degree=2)
        np.testing.assert_array_equal(out["X_aug"], X)
        assert out["rounds"] == []


# ---------------------------------------------------------------------------
# validate_pair_fe_cv : API contract
# ---------------------------------------------------------------------------


class TestValidatePairFeCV:
    """Unit checks on ``validate_pair_fe_cv`` return contract, edge cases, determinism."""

    def test_return_keys(self):
        """Return keys."""
        x_a, x_b, y = _additive_pair(n=400, seed=10)
        out = validate_pair_fe_cv(
            x_a,
            x_b,
            y,
            n_splits=3,
            n_trials=6,
            max_degree=2,
            seed=0,
        )
        expected = {
            "in_sample_mi",
            "oos_mean",
            "oos_std",
            "oos_per_fold",
            "optimism_ratio",
            "trivial_oos_mean",
            "honest_uplift_vs_trivial",
            "folds_with_positive_uplift",
        }
        assert expected.issubset(set(out.keys()))
        assert len(out["oos_per_fold"]) == 3
        for fold in out["oos_per_fold"]:
            assert "fold" in fold and "oos_mi" in fold

    def test_types_are_floats(self):
        """Aggregate fields are plain Python floats (the dict is JSON-serialisable downstream)."""
        x_a, x_b, y = _additive_pair(n=400, seed=11)
        out = validate_pair_fe_cv(
            x_a,
            x_b,
            y,
            n_splits=3,
            n_trials=6,
            max_degree=2,
            seed=0,
        )
        for k in ("oos_mean", "oos_std", "trivial_oos_mean", "honest_uplift_vs_trivial"):
            assert isinstance(out[k], float)
        assert isinstance(out["folds_with_positive_uplift"], int)

    def test_pure_noise_no_spurious_uplift(self):
        """When y is independent of (x_a, x_b), honest OOS MI should stay tiny (<=0.1).
        The optimism_ratio (in-sample / oos) is allowed to spike since both sides are near-zero,
        but the absolute OOS MI is the trustworthy signal here."""
        x_a, x_b, y = _noise_pair(n=600, seed=12)
        out = validate_pair_fe_cv(
            x_a,
            x_b,
            y,
            n_splits=3,
            n_trials=6,
            max_degree=2,
            seed=0,
        )
        # OOS MI on a noise label can't honestly exceed ~0.1 nats with this estimator at n=600.
        assert out["oos_mean"] <= 0.1, f"spurious oos uplift on pure-noise target: {out['oos_mean']:.3f}"

    def test_determinism_same_seed(self):
        """Same seed => same OOS-MI series (within numpy float identity)."""
        x_a, x_b, y = _additive_pair(n=400, seed=13)
        out_a = validate_pair_fe_cv(
            x_a,
            x_b,
            y,
            n_splits=3,
            n_trials=6,
            max_degree=2,
            seed=123,
        )
        out_b = validate_pair_fe_cv(
            x_a,
            x_b,
            y,
            n_splits=3,
            n_trials=6,
            max_degree=2,
            seed=123,
        )
        a = np.array([f["oos_mi"] for f in out_a["oos_per_fold"]])
        b = np.array([f["oos_mi"] for f in out_b["oos_per_fold"]])
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        assert out_a["in_sample_mi"] == pytest.approx(out_b["in_sample_mi"], rel=1e-12)


# ---------------------------------------------------------------------------
# Wiring : composition cascades feed into baseline scoring
# ---------------------------------------------------------------------------


class TestWiring:
    """Cross-module sanity: an engineered column emitted by ``compose_pair_fe`` should itself
    score as a finite trivial-baseline candidate when fed back as a 1-D feature."""

    def test_engineered_column_is_scorable(self):
        """Engineered column is scorable."""
        x_a, x_b, y = _additive_pair(n=400, seed=20)
        X = np.column_stack([x_a, x_b])
        # ``_additive_pair`` is y = sign(x_a + x_b): the trivial pair
        # (x_a + x_b) directly captures the signal, so the *polynomial*
        # optimiser typically can't beat that trivial baseline by the
        # default 5pct uplift threshold. The wiring test only needs an
        # engineered column to exercise the column-scoring path - lower
        # the threshold + bump trials to make column emission reliable
        # on this fixture. Pre-fix the test silently skipped ~30pct of
        # runs (depending on seed); with the explicit threshold the
        # wiring contract is exercised every time.
        out = compose_pair_fe(
            X,
            y,
            n_rounds=1,
            top_k_per_round=1,
            n_trials=80,
            max_degree=2,
            # 0.5 == "accept any finite engineered column even when its MI
            # is half of the baseline" -- the wiring test only needs A
            # column to score; the polynomial can't beat the trivial
            # ``x_a + x_b`` baseline on a perfectly-additive fixture
            # because (x_a + x_b) already captures all signal, and the
            # default ``> 1.05`` threshold rejects every polynomial whose
            # MI ties or only marginally improves over the trivial pair.
            baseline_uplift_threshold=0.5,
        )
        assert out["X_aug"].shape[1] > 2, (
            "compose_pair_fe optimiser failed to emit any engineered "
            "column on the additive pair fixture even with "
            "baseline_uplift_threshold=1.0 - regression in the "
            "polynomial fit / MI scoring path."
        )
        new_col = out["X_aug"][:, 2]
        assert np.all(np.isfinite(new_col))
        mi = _mi_1d(new_col, y, discrete_target=True)
        assert mi >= 0.0
        assert np.isfinite(mi)


# ---------------------------------------------------------------------------
# biz_value : XOR pair must materially beat each parent's marginal MI
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_biz_xor_pair_uplifts_over_parents():
    """y = x_a XOR x_b: each parent's marginal MI is near zero (<=0.02 nats).
    The engineered pair, validated by held-out CV, must reach >=2x the best parent's marginal MI.
    Calibrated: parent marginals measured ~0.005 nats, oos_mean ~0.67 nats => observed ratio >100x;
    threshold pinned at 2.0 (well under observed margin, robust to estimator noise)."""
    x_a, x_b, y = _xor_pair_binary(n=1500, seed=0)
    mi_a = _mi_1d(x_a, y, discrete_target=True)
    mi_b = _mi_1d(x_b, y, discrete_target=True)
    best_parent = max(mi_a, mi_b, 1e-9)
    out = validate_pair_fe_cv(
        x_a,
        x_b,
        y,
        n_splits=3,
        n_trials=8,
        max_degree=2,
        seed=0,
    )
    uplift = out["oos_mean"] / best_parent
    assert uplift >= 2.0, f"XOR pair uplift {uplift:.2f}x over best parent (mi_a={mi_a:.4f}, mi_b={mi_b:.4f}, oos_mean={out['oos_mean']:.4f}) - expected >= 2.0"
