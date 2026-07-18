"""Synthetic temporal-bias PU-learning test.

Mirrors the production drift pattern from the Upwork-jobs-hired dataset:
- True P(y=1|x) ≈ 0.40 across the whole timeline
- Most months: only positives observed (positive-only / "biased") because
  the data source only surfaces hired jobs and never lists the rest
- A handful of "unbiased" months: both classes observed (because all
  active job uids were captured)

The graph the user shared shows ~98% positive-rate from 2018 to 2021Q3,
~40% positive-rate during 2021Q4-2022Q1 (unbiased window), then ~98%
again until very recently. The same shape is reproduced synthetically:

    rate
    1.0  ████████████████████████  ──────────────────────────  ████████████
    0.4                            ████████ <- unbiased window  ────  unbiased now

Test invariants:
1. Naive classifier on biased-dominated data:
   - LEARNS to over-predict y=1 (mean predicted prob >> true prior 0.40).
   - Has poor calibration on the unbiased TEST set (Brier loss high).
2. Each PU strategy beats naive on calibration / Brier:
   - ``unbiased_only``: simplest, ignores biased data.
   - ``importance_weighted``: uses biased data with downweighting.
   - ``elkan_noto``: classical PU classifier.
   - ``auto``: picks one of the above by data size.
3. Discrimination (ROC-AUC) doesn't catastrophically drop relative to naive.

Why this test matters: in the user's production regime (~4% unbiased
data) the proxy classifier needed by Elkan-Noto is severely class-skewed
on s, and pure Elkan-Noto without proxy balancing fails. The wrapper's
default ``balance_proxy=True`` plus the option to fall back to simpler
strategies (`unbiased_only`, `importance_weighted`) make the wrapper
useful in the realistic regime — this test pins each.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from mlframe.training.pu_learning import (
    PULearningWrapper,
    estimate_c_from_unbiased_positives,
)
from tests.conftest import is_fast_mode


# Pin global RNG state per-test. HistGradientBoostingClassifier passes
# ``random_state=0`` through to its own RNG, but sklearn internals (subsampling,
# tie-breaking in some estimators) can still touch the numpy global stream;
# pytest-randomly reseeds that stream per-test, which made the statistical
# tolerance bands on ``elkan_noto`` / ``prior_shift_correction`` flake.
@pytest.fixture(autouse=True)
def _stable_global_rng():
    """Stable global rng."""
    import random

    np.random.seed(0)
    random.seed(0)


# ----- Synthetic data generator -------------------------------------------------


def _gen_temporal_pu_dataset(
    n_total: int = 60_000,
    n_features: int = 8,
    true_prior: float = 0.40,
    n_months: int = 96,
    unbiased_month_indices: tuple = (45, 46, 47, 95),
    biased_visible_pos_rate: float = 0.98,
    seed: int = 0,
):
    """Build a synthetic timeline that matches the user's graph shape.

    Returns dict with X, y_true, month, is_unbiased, y_observed,
    visible_mask, true_prior.
    """
    rng = np.random.default_rng(seed)

    # 1. Features + true labels via simple linear-logistic generator.
    X = rng.standard_normal((n_total, n_features)).astype(np.float32)
    weights = np.array([1.5, -1.2, 0.8, -0.6, 0.0, 0.0, 0.0, 0.0])
    intercept = np.log(true_prior / (1 - true_prior))
    logits = X @ weights[:n_features] + intercept
    proba = 1.0 / (1.0 + np.exp(-logits))
    y_true = (rng.uniform(size=n_total) < proba).astype(np.int8)

    # 2. Random month assignment.
    month = rng.integers(low=0, high=n_months, size=n_total).astype(np.int32)

    # 3. Unbiased-month flag.
    unbiased_set = set(unbiased_month_indices)
    is_unbiased = np.array([m in unbiased_set for m in month], dtype=bool)

    # 4. Visibility / observed-label logic.
    visible_mask = np.zeros(n_total, dtype=bool)
    y_observed = np.zeros(n_total, dtype=np.int8)

    # Unbiased rows: all visible, label = true.
    visible_mask[is_unbiased] = True
    y_observed[is_unbiased] = y_true[is_unbiased]

    # Biased rows: only positives visible (with stochastic drop).
    biased_pos = (~is_unbiased) & (y_true == 1)
    keep_pos = rng.uniform(size=biased_pos.sum()) < biased_visible_pos_rate
    biased_pos_idx = np.where(biased_pos)[0]
    visible_mask[biased_pos_idx[keep_pos]] = True
    y_observed[biased_pos_idx[keep_pos]] = 1

    return {
        "X": X,
        "y_true": y_true,
        "month": month,
        "is_unbiased": is_unbiased,
        "y_observed": y_observed,
        "visible_mask": visible_mask,
        "true_prior": true_prior,
    }


@pytest.fixture(scope="module")
def synthetic_pu():
    """Module-scoped fixture: train (everything except last unbiased month)
    + TEST (last unbiased month, fully labeled)."""
    # Fast mode: 12k rows is the smallest size that still produces a stable PU
    # estimator signal (drops fixture wallclock from ~20s to ~3s).
    n_total = 12_000 if is_fast_mode() else 80_000
    data = _gen_temporal_pu_dataset(
        n_total=n_total,
        n_features=8,
        true_prior=0.40,
        n_months=96,
        unbiased_month_indices=(45, 46, 47, 95),
        biased_visible_pos_rate=0.98,
        seed=0,
    )

    last_unbiased_month = 95
    test_mask = (data["month"] == last_unbiased_month) & data["visible_mask"]
    train_mask = data["visible_mask"] & ~test_mask

    return {
        "X_train": data["X"][train_mask],
        "y_train_observed": data["y_observed"][train_mask],
        "y_train_true": data["y_true"][train_mask],
        "is_unbiased_train": data["is_unbiased"][train_mask],
        "X_test": data["X"][test_mask],
        "y_test_true": data["y_true"][test_mask],
        "true_prior": data["true_prior"],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }


# ----- Sanity checks on the synthetic data -------------------------------------


def test_synthetic_data_shape_resembles_user_graph(synthetic_pu):
    """Train should look ~98% positive (biased dominates); TEST should
    look ~40% positive (unbiased month). This is the very pattern the
    user described."""
    train_pos_rate = synthetic_pu["y_train_observed"].mean()
    test_pos_rate = synthetic_pu["y_test_true"].mean()

    assert train_pos_rate > 0.90, f"train should be heavily positive-skewed; got {train_pos_rate:.3f}"
    assert 0.30 < test_pos_rate < 0.50, f"test should approximate true prior 0.40; got {test_pos_rate:.3f}"


def test_unbiased_subset_has_enough_positives(synthetic_pu):
    """For c-estimation to be stable we need plenty of unbiased
    positives in train. Default fixture sizing should give ~1k+."""
    ub = synthetic_pu["is_unbiased_train"]
    n_ub_pos = int((ub & (synthetic_pu["y_train_true"] == 1)).sum())
    assert n_ub_pos >= 100, f"need >=100 unbiased positives for c-estimation; got {n_ub_pos}"


# ----- Naive baseline (the failure mode we're trying to fix) -------------------


def _naive_test_metrics(synthetic_pu, base_factory):
    """Train naive classifier on observed labels, return TEST metrics."""
    clf = base_factory()
    clf.fit(synthetic_pu["X_train"], synthetic_pu["y_train_observed"])
    probs = clf.predict_proba(synthetic_pu["X_test"])[:, 1]
    return {
        "mean_pred": float(probs.mean()),
        "auc": float(roc_auc_score(synthetic_pu["y_test_true"], probs)),
        "brier": float(brier_score_loss(synthetic_pu["y_test_true"], probs)),
    }


def test_naive_classifier_is_miscalibrated(synthetic_pu):
    """The naive classifier overestimates P(y=1) on TEST because train
    is biased toward positives. This is the failure mode."""
    m = _naive_test_metrics(synthetic_pu, lambda: HistGradientBoostingClassifier(max_iter=200, random_state=0))
    true_prior = synthetic_pu["true_prior"]
    # Naive mean prediction must be markedly above the true 0.40 prior.
    assert m["mean_pred"] - true_prior > 0.20, f"naive should be miscalibrated; mean_pred={m['mean_pred']:.3f} vs true_prior={true_prior:.3f}"


# ----- Per-strategy tests ------------------------------------------------------


@pytest.mark.parametrize(
    "strategy",
    [
        "unbiased_only",
        "prior_shift_correction",
        "elkan_noto",
    ],
)
def test_strategy_recovers_calibration_on_test(strategy, synthetic_pu):
    """Each strategy should produce mean predicted prob much closer to
    the true 0.40 prior than the naive baseline (which sits ~0.88).

    The looseness of the bound is per-strategy: unbiased_only is best,
    elkan_noto is worst (still beats naive by a large margin).

    Fast mode shrinks the fixture to 12k rows -> ~160 unbiased positives,
    which is below the proxy/Saerens calibration-recovery floor (the
    estimate's std is ~0.5/sqrt(160) ~ 0.04 BEFORE the strategy amplifies
    it). The tolerances below are calibrated for the prod regime (80k
    rows, ~1.4k unbiased pos); they don't make sense in fast mode.
    """
    if is_fast_mode():
        pytest.skip(
            "calibration-recovery contract needs the full 80k-row fixture "
            "(fast-mode 12k rows shrinks the unbiased subset below the "
            "Saerens/Elkan-Noto noise floor)."
        )
    base = HistGradientBoostingClassifier(max_iter=200, random_state=0)
    pu = PULearningWrapper(
        base_estimator=base,
        strategy=strategy,
        true_prior=0.40 if strategy == "prior_shift_correction" else None,
        min_unbiased_positives=100,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )

    pu_probs = pu.predict_proba(synthetic_pu["X_test"])[:, 1]
    pu_mean = float(pu_probs.mean())
    true_prior = synthetic_pu["true_prior"]
    distance = abs(pu_mean - true_prior)

    # Per-strategy tolerance — unbiased_only and prior_shift_correction
    # should both be tight (within ~10pp of true prior); elkan_noto is
    # noisy in the small-unbiased-subset regime so gets more slack.
    tolerance = {
        "unbiased_only": 0.10,
        "prior_shift_correction": 0.10,
        "elkan_noto": 0.30,
    }[strategy]

    assert (
        distance < tolerance
    ), f"strategy={strategy}: PU mean_pred={pu_mean:.3f} too far from true prior {true_prior:.3f} (Δ={distance:.3f}, tolerance={tolerance})"


@pytest.mark.parametrize(
    "strategy",
    [
        "unbiased_only",
        "prior_shift_correction",
        "elkan_noto",
    ],
)
def test_strategy_lowers_brier_vs_naive(strategy, synthetic_pu):
    """Headline win: every strategy must reduce TEST Brier loss vs naive
    by a meaningful margin. This is what calibration recovery buys.

    Skipped in fast mode -- see test_strategy_recovers_calibration_on_test
    for the rationale (the Brier delta minimums below are calibrated for
    the prod-scale fixture, not the 12k-row fast-mode shrink).
    """
    if strategy == "elkan_noto" and is_fast_mode():
        pytest.skip("elkan_noto's proxy classifier needs the full unbiased subset (>=1k pos/neg) to estimate c reliably; fast mode skipped.")
    # Fewer boosting iterations under --fast keep the per-fit cost low so a full-suite ``-n`` contended worker doesn't hit
    # the timeout; the Brier drop is comfortably above the floor at 100 iters (measured drop ~0.21 vs the 0.05 floor).
    max_iter = 100 if is_fast_mode() else 200
    naive = _naive_test_metrics(synthetic_pu, lambda: HistGradientBoostingClassifier(max_iter=max_iter, random_state=0))

    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=max_iter, random_state=0),
        strategy=strategy,
        true_prior=0.40 if strategy == "prior_shift_correction" else None,
        min_unbiased_positives=100,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )
    pu_probs = pu.predict_proba(synthetic_pu["X_test"])[:, 1]
    pu_brier = float(brier_score_loss(synthetic_pu["y_test_true"], pu_probs))

    # All three should beat naive. Margin is per-strategy:
    # unbiased_only typically halves Brier; the others give ~30% reduction.
    min_brier_drop = {
        "unbiased_only": 0.10,
        "prior_shift_correction": 0.05,
        "elkan_noto": 0.05,
    }[strategy]

    drop = naive["brier"] - pu_brier
    assert drop > min_brier_drop, f"strategy={strategy}: Brier drop {drop:.4f} < min {min_brier_drop} (naive={naive['brier']:.4f}, pu={pu_brier:.4f})"


@pytest.mark.parametrize(
    "strategy",
    [
        "unbiased_only",
        "prior_shift_correction",
        "elkan_noto",
    ],
)
def test_strategy_preserves_or_improves_auc(strategy, synthetic_pu):
    """Discrimination shouldn't catastrophically drop. We allow a 5pp
    AUC slack relative to naive — strategies trade some AUC for
    calibration but shouldn't tank.

    Skipped for elkan_noto in fast mode: small unbiased subset makes
    proxy g(x)/c noisy and the clipped recovered probabilities can
    drop AUC > 5pp even when the strategy is implemented correctly.
    """
    if strategy == "elkan_noto" and is_fast_mode():
        pytest.skip("elkan_noto needs the full unbiased subset to keep AUC; fast-mode shrink violates the 5pp slack legitimately.")
    # Fewer boosting iterations under --fast so two HGB fits don't push a full-suite ``-n`` contended worker past the timeout; the AUC slack
    # is comparison-relative (PU vs naive at the SAME max_iter) so reducing both arms keeps the 5pp contract honest.
    max_iter = 100 if is_fast_mode() else 200
    naive = _naive_test_metrics(synthetic_pu, lambda: HistGradientBoostingClassifier(max_iter=max_iter, random_state=0))

    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=max_iter, random_state=0),
        strategy=strategy,
        true_prior=0.40 if strategy == "prior_shift_correction" else None,
        min_unbiased_positives=100,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )
    pu_probs = pu.predict_proba(synthetic_pu["X_test"])[:, 1]
    pu_auc = float(roc_auc_score(synthetic_pu["y_test_true"], pu_probs))

    assert pu_auc >= naive["auc"] - 0.05, f"strategy={strategy}: AUC dropped too much (pu={pu_auc:.4f}, naive={naive['auc']:.4f}). Allowed slack 5pp."


# ----- auto strategy resolution ------------------------------------------------


def test_auto_picks_unbiased_only_when_subset_large_enough(synthetic_pu):
    """With "large enough" unbiased pos+neg counts auto picks unbiased_only.

    Fast mode (12k total rows) only produces ~160 unbiased positives so we
    pass the threshold explicitly rather than relying on the prod default of
    1000 -- the test exercises the AUTO-DISPATCH contract, not the threshold
    value itself.
    """
    ub = synthetic_pu["is_unbiased_train"]
    y_train_obs = synthetic_pu["y_train_observed"]
    n_ub_pos = int((ub & (y_train_obs == 1)).sum())
    n_ub_neg = int((ub & (y_train_obs == 0)).sum())
    target_threshold = max(50, min(n_ub_pos, n_ub_neg) - 20)
    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=50, random_state=0),
        strategy="auto",
        min_unbiased_positives=50,
        auto_strategy_unbiased_count_threshold=target_threshold,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )
    assert pu.strategy_ == "unbiased_only"


def test_auto_falls_back_to_importance_weighted_when_subset_small():
    """Smaller fixture: only a handful of unbiased rows → auto picks
    importance_weighted."""
    rng = np.random.default_rng(0)
    n = 5000
    X = rng.standard_normal((n, 6)).astype(np.float32)
    is_ub = np.zeros(n, dtype=bool)
    is_ub[:200] = True  # 200 unbiased rows total — below threshold
    y = np.ones(n, dtype=np.int8)
    y[is_ub] = (rng.uniform(size=is_ub.sum()) < 0.4).astype(np.int8)

    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=50, random_state=0),
        strategy="auto",
        min_unbiased_positives=20,
        auto_strategy_unbiased_count_threshold=1000,
    )
    pu.fit(X=X, y=y, is_unbiased=is_ub)
    assert pu.strategy_ == "prior_shift_correction"


# ----- API correctness checks --------------------------------------------------


def test_predict_classes(synthetic_pu):
    """Predict classes."""
    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=50, random_state=0),
        strategy="unbiased_only",
        min_unbiased_positives=100,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )
    preds = pu.predict(synthetic_pu["X_test"])
    assert set(preds.tolist()).issubset({0, 1})
    assert pu.classes_.tolist() == [0, 1]


def test_predict_proba_shape(synthetic_pu):
    """Predict proba shape."""
    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=50, random_state=0),
        strategy="elkan_noto",
        min_unbiased_positives=100,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )
    p = pu.predict_proba(synthetic_pu["X_test"])
    assert p.shape == (synthetic_pu["n_test"], 2)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_decision_function_returns_pos_prob(synthetic_pu):
    """Decision function returns pos prob."""
    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=50, random_state=0),
        strategy="unbiased_only",
        min_unbiased_positives=100,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )
    df = pu.decision_function(synthetic_pu["X_test"])
    p = pu.predict_proba(synthetic_pu["X_test"])[:, 1]
    np.testing.assert_array_equal(df, p)


def test_rejects_too_few_unbiased_positives():
    """Rejects too few unbiased positives."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 4))
    y = (rng.uniform(size=200) < 0.5).astype(np.int8)
    is_unbiased = np.zeros(200, dtype=bool)
    is_unbiased[:10] = True

    pu = PULearningWrapper(
        base_estimator=LogisticRegression(max_iter=1000, random_state=0),
        strategy="unbiased_only",
    )
    with pytest.raises(ValueError, match="unbiased positive samples"):
        pu.fit(X=X, y=y, is_unbiased=is_unbiased)


def test_rejects_non_binary_y():
    """Rejects non binary y."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    y = rng.integers(0, 3, size=300).astype(np.int8)
    is_unbiased = np.ones(300, dtype=bool)
    pu = PULearningWrapper(
        base_estimator=LogisticRegression(max_iter=1000, random_state=0),
    )
    with pytest.raises(ValueError, match="binary-only"):
        pu.fit(X=X, y=y, is_unbiased=is_unbiased)


def test_estimate_c_methods():
    """Estimate c methods."""
    proxy_probs = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    assert estimate_c_from_unbiased_positives(proxy_probs, "mean_unbiased_pos") == pytest.approx(0.7)
    assert estimate_c_from_unbiased_positives(proxy_probs, "max_unbiased_pos") == 0.9
    assert estimate_c_from_unbiased_positives(proxy_probs, "median_unbiased_pos") == 0.7

    with pytest.raises(ValueError, match="empty"):
        estimate_c_from_unbiased_positives(np.array([]))
    with pytest.raises(ValueError, match="Unknown c-estimation method"):
        estimate_c_from_unbiased_positives(proxy_probs, "bogus")  # type: ignore


def test_unfitted_raises():
    """Unfitted raises."""
    from sklearn.exceptions import NotFittedError

    pu = PULearningWrapper(base_estimator=LogisticRegression())
    # NotFittedError is the canonical sklearn signal; RuntimeError is the
    # legacy shape kept for narrative compatibility.
    with pytest.raises((NotFittedError, RuntimeError), match="not fitted"):
        pu.predict_proba(np.zeros((5, 4)))


def test_prior_shift_uses_provided_true_prior(synthetic_pu):
    """When true_prior is explicitly provided, the wrapper uses it
    rather than estimating from the unbiased subset."""
    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=50, random_state=0),
        strategy="prior_shift_correction",
        true_prior=0.4,
        min_unbiased_positives=100,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )
    assert pu.estimated_prior_ == 0.4
    assert hasattr(pu, "train_prior_")
    assert pu.train_prior_ > 0.9  # train is heavily biased toward positives


def test_prior_shift_validates_prior_range(synthetic_pu):
    """Prior shift validates prior range."""
    pu = PULearningWrapper(
        base_estimator=LogisticRegression(max_iter=200, random_state=0),
        strategy="prior_shift_correction",
        true_prior=1.5,  # invalid
        min_unbiased_positives=100,
    )
    with pytest.raises(ValueError, match="true_prior must be in"):
        pu.fit(
            X=synthetic_pu["X_train"],
            y=synthetic_pu["y_train_observed"],
            is_unbiased=synthetic_pu["is_unbiased_train"],
        )


def test_elkan_noto_records_c_attribute(synthetic_pu):
    """Elkan noto records c attribute."""
    pu = PULearningWrapper(
        base_estimator=HistGradientBoostingClassifier(max_iter=100, random_state=0),
        strategy="elkan_noto",
        min_unbiased_positives=100,
    )
    pu.fit(
        X=synthetic_pu["X_train"],
        y=synthetic_pu["y_train_observed"],
        is_unbiased=synthetic_pu["is_unbiased_train"],
    )
    assert hasattr(pu, "c_")
    assert 0 < pu.c_ <= 1
