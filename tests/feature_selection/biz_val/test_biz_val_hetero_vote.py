"""biz_value + behavioral coverage for ``heterogeneous_relevance_vote`` (hetero_vote).

Closes the hetero_vote gaps flagged by the 2026-06-10 FS-tests audit:
param_axes-07 (regression panel + R2 skill dark), coverage_asymmetry_wrappers-08/09/10/11
(regression path, models=/percentile=/per_model_hit_frac=/vote_threshold boundary/ndarray input,
the discriminating skill-weighting flip, and the permutation-importance fallback) and
gaps_selection_masking-17 (regression-target path entirely untested).

All quantitative floors are calibrated from a measured dev run and pinned 5-15% below the
measured value per CLAUDE.md. Seeds are fixed everywhere; the default 3-member panel fit is a
few seconds, so heavier multi-call legs carry @pytest.mark.slow with a fast representative kept
via MLFRAME_FAST=1 (the conftest fast-mode collection hook skips slow-marked tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlframe.feature_selection.hetero_vote import heterogeneous_relevance_vote, _importance
from tests.conftest import fast_n_estimators


# ---------------------------------------------------------------------------
# Synthetic data: continuous (regression) and binary (classification) targets.
# Both share the canonical hetero_vote architecture: 4 marginally-strong signals
# + p_noise pure-noise columns, signal weights [1.5, -1.2, 1.0, 0.9].
# ---------------------------------------------------------------------------

_SIGNAL_WEIGHTS = np.array([1.5, -1.2, 1.0, 0.9])



pytestmark = pytest.mark.timeout(120)  # untimed biz_val real-fit tier: surface a hang fast (the multi-shadow-trial RF/Ridge/kNN panel legitimately needs >60s; global --timeout=600 is a coarse backstop)

def _reg_data(seed: int = 0, n: int = 1500, p_sig: int = 4, p_noise: int = 20):
    """Continuous target ``y = z @ [1.5,-1.2,1.0,0.9] + 0.3*noise``."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p_sig))
    y = z @ _SIGNAL_WEIGHTS + 0.3 * rng.standard_normal(n)
    cols = {f"sig_{i}": z[:, i] for i in range(p_sig)}
    for j in range(p_noise):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    signal = [f"sig_{i}" for i in range(p_sig)]
    noise = [f"noise_{j}" for j in range(p_noise)]
    return pd.DataFrame(cols), pd.Series(y, name="y"), signal, noise


def _clf_data(seed: int = 0, n: int = 1200, p_sig: int = 4, p_noise: int = 20):
    """Binary target via the logistic of the same linear score."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p_sig))
    logit = z @ _SIGNAL_WEIGHTS
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {f"sig_{i}": z[:, i] for i in range(p_sig)}
    for j in range(p_noise):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    signal = [f"sig_{i}" for i in range(p_sig)]
    noise = [f"noise_{j}" for j in range(p_noise)]
    return pd.DataFrame(cols), pd.Series(y, name="y"), signal, noise


def _clf_data_ndarray(seed: int = 0, n: int = 1200, p_sig: int = 4, p_noise: int = 20):
    """Same binary problem but X delivered as a bare ndarray (no column names)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, p_sig))
    logit = z @ _SIGNAL_WEIGHTS
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    X = np.column_stack([z] + [rng.standard_normal(n) for _ in range(p_noise)])
    return X, y, list(range(p_sig))


# ---------------------------------------------------------------------------
# Deterministic stub estimators (injected via models=) for exact vote control.
# ---------------------------------------------------------------------------


class _StubFI(BaseEstimator):
    """Estimator whose post-fit ``feature_importances_`` is 1.0 on ``hit_cols``, 0.0 elsewhere.

    Cloneable (sklearn ``clone`` reads ``get_params``). With ``percentile=100`` the shadow
    bar is ``max(shadow_importances) == 0``, so a real column with importance 1.0 strictly
    exceeds it (hits) and a 0.0 column does not. This lets a test construct an EXACT panel
    vote fraction independent of any learned model, pinning the vote_threshold / per_model_hit_frac
    boundary numerics directly.
    """

    def __init__(self, hit_cols=()):
        self.hit_cols = hit_cols

    def fit(self, X, y):
        n_cols = np.asarray(X).shape[1]
        imp = np.zeros(n_cols, dtype=float)
        for c in self.hit_cols:
            if c < n_cols:
                imp[c] = 1.0
        self.feature_importances_ = imp
        return self


class _BlindFI(DummyClassifier):
    """A near-chance ("blind") panel member exposing UNIFORM feature_importances_.

    It predicts the prior (CV ROC-AUC ~ 0.5 -> cv_skill clamps to the floor) and, because its
    importances are uniform, with ``percentile=100`` the shadow max equals every feature's
    importance, so ``imp > thr`` is never strictly true and the member hits nothing. Both make it
    the exact "structurally-blind voter" the skill-weighting option exists to downweight.
    """

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        n_cols = np.asarray(X).shape[1]
        self.feature_importances_ = np.ones(n_cols, dtype=float) / n_cols
        return self


# ===========================================================================
# (a) REGRESSION path: classification=False -> regressor panel + R2 skill.
#     Findings param_axes-07, coverage_asymmetry_wrappers-08, gaps_selection_masking-17.
# ===========================================================================


def test_hetero_vote_regression_keeps_signal_drops_noise():
    """classification=False fits the RF/Ridge/kNN REGRESSOR panel + KFold-R2 skill.

    Measured (seed=0, n=1500): all 4 signals accepted, 0 noise columns admitted.
    Floor pins all-4-signal recovery and noise admission <= 1 (5-15% headroom over the
    measured 0 noise)."""
    X, y, signal, noise = _reg_data(seed=0)
    accepted, info = heterogeneous_relevance_vote(
        X, y, classification=False, n_shadow_trials=3, vote_threshold=0.5, random_state=0
    )
    acc = set(accepted)
    for s in signal:
        assert s in acc, f"regression panel dropped signal {s} (vote_fraction={info['vote_fraction'][s]})"
    n_noise = len(acc & set(noise))
    assert n_noise <= 1, f"regression vote admitted {n_noise} noise columns: {sorted(acc & set(noise))}"
    assert info["n_models"] == 3
    assert set(info["vote_fraction"]) == set(X.columns)
    assert all(0.0 <= v <= 1.0 for v in info["vote_fraction"].values())


def test_hetero_vote_regression_skill_weights_are_r2_derived():
    """weight_by_cv_skill=True on the regression panel weights members by (R2 - 0), clamped to
    the floor. Measured (seed=0): weights {tree~0.896, linear~0.983, distance~0.576} -- every one
    lands strictly inside [cv_skill_floor=0.05, 1.0] (R2 <= 1 by construction), and the strong
    signals still survive."""
    X, y, signal, _ = _reg_data(seed=0)
    accepted, info = heterogeneous_relevance_vote(
        X, y, classification=False, n_shadow_trials=3, weight_by_cv_skill=True,
        cv_skill_folds=3, cv_skill_floor=0.05, random_state=0,
    )
    for s in signal:
        assert s in set(accepted), f"R2-skill-weighted regression vote dropped signal {s}"
    w = info["model_weights"]
    assert set(w) == {"tree", "linear", "distance"}
    for name, v in w.items():
        assert 0.05 <= v <= 1.0, f"R2-derived weight for {name} out of [0.05, 1.0]: {v}"


# ===========================================================================
# (b) ndarray input -> xN column names. Finding coverage_asymmetry_wrappers-09.
# ===========================================================================


def test_hetero_vote_ndarray_input_uses_xN_names():
    """A bare ndarray X yields synthetic ``x{i}`` names in both accepted and info['vote_fraction']."""
    X, y, sig_idx = _clf_data_ndarray(seed=0)
    p = X.shape[1]
    accepted, info = heterogeneous_relevance_vote(
        X, y, classification=True, n_shadow_trials=3, vote_threshold=0.5, random_state=0
    )
    expected_names = {f"x{i}" for i in range(p)}
    assert set(info["vote_fraction"]) == expected_names
    assert set(accepted) <= expected_names
    # The 4 signal columns x0..x3 must survive.
    for i in sig_idx:
        assert f"x{i}" in set(accepted), f"ndarray-named signal x{i} dropped"


# ===========================================================================
# (c) custom 2-model panel via models=. Finding coverage_asymmetry_wrappers-09.
# ===========================================================================


def test_hetero_vote_custom_two_model_panel():
    """A user-supplied 2-member panel: info['n_models']==2 and model_weights keys == panel keys."""
    X, y, signal, _ = _clf_data(seed=0)
    panel = {
        "rf": RandomForestClassifier(n_estimators=fast_n_estimators(120), random_state=0),
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
    }
    accepted, info = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=3, vote_threshold=0.5, random_state=0
    )
    assert info["n_models"] == 2
    assert set(info["model_weights"]) == {"rf", "lr"}
    # The 2-member panel still recovers the strong marginal signals.
    for s in signal:
        assert s in set(accepted), f"2-model panel dropped signal {s}"


# ===========================================================================
# (d) percentile monotonicity: pct=50 accepts a superset of pct=100.
#     Finding coverage_asymmetry_wrappers-09.
# ===========================================================================


def test_hetero_vote_percentile_monotonicity():
    """Lowering ``percentile`` lowers the shadow bar -> more hits -> the accepted set GROWS.

    Same data + seed: percentile=50 accepts a superset of percentile=100. Measured (seed=0):
    pct100 accepts the 4 signals; pct50 accepts those 4 plus ~11 extra columns -- a strict superset."""
    X, y, _, _ = _clf_data(seed=0)
    acc100, _ = heterogeneous_relevance_vote(
        X, y, classification=True, n_shadow_trials=3, percentile=100, vote_threshold=0.5, random_state=0
    )
    acc50, _ = heterogeneous_relevance_vote(
        X, y, classification=True, n_shadow_trials=3, percentile=50, vote_threshold=0.5, random_state=0
    )
    assert set(acc100) <= set(acc50), (
        f"percentile=50 is not a superset of percentile=100: "
        f"only-in-100={sorted(set(acc100) - set(acc50))}"
    )
    # The lower bar genuinely admits MORE (measured ~11 extra noise columns); pin a non-trivial gap.
    assert len(acc50) > len(acc100)


# ===========================================================================
# (e) per_model_hit_frac and vote_threshold >= boundary with a deterministic stub panel.
#     Finding coverage_asymmetry_wrappers-09.
# ===========================================================================


def _stub_panel_three_cols(seed: int = 0, n: int = 200):
    """3 columns x0,x1,x2 + a 3-member stub panel where exactly 2 of 3 members hit x0.

    With percentile=100 and shadows at importance 0, vote_fraction[x0] == 2/3 EXACTLY,
    independent of the (random) labels -- the discriminating point for the >= boundary."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["x0", "x1", "x2"])
    y = pd.Series((rng.random(n) < 0.5).astype(int))
    panel = {"a": _StubFI(hit_cols=(0,)), "b": _StubFI(hit_cols=(0,)), "c": _StubFI(hit_cols=(1,))}
    return X, y, panel


def test_hetero_vote_vote_threshold_ge_boundary_is_inclusive():
    """The accept rule is ``vote_fraction >= vote_threshold`` (inclusive). Construct vote_frac[x0]
    exactly 2/3 via the stub panel: accepted at vote_threshold=2/3, rejected at 2/3 + 1e-9."""
    X, y, panel = _stub_panel_three_cols()
    acc_at, info_at = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=1, percentile=100,
        vote_threshold=2.0 / 3.0, random_state=0,
    )
    assert info_at["vote_fraction"]["x0"] == pytest.approx(2.0 / 3.0)
    assert "x0" in acc_at, "x0 (vote_frac=2/3) must be accepted at vote_threshold=2/3 (>= is inclusive)"
    acc_above, _ = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=1, percentile=100,
        vote_threshold=2.0 / 3.0 + 1e-9, random_state=0,
    )
    assert "x0" not in acc_above, "x0 (vote_frac=2/3) must be rejected just above the 2/3 threshold"


class _AltTrialFI(BaseEstimator):
    """Hits x0 on a deterministic per-trial parity, 0.0 on everything else (incl. shadows).

    hetero_vote rebuilds shadows per trial with ``default_rng(random_state + tr)``, so the shadow
    block [P:] is a deterministic function of the trial. The stub hits x0 iff ``argmax`` of the first
    shadow column is even -- a stable per-(data, trial) fingerprint. At random_state=2 / n_shadow_trials=2
    this realizes a hit-rate of exactly 1/2 for x0 (verified below), the discriminating point for the
    ``>= per_model_hit_frac`` boundary. ``feature_importances_`` is ALWAYS set (0.0 on a miss) so the
    estimator never falls through to permutation_importance."""

    def fit(self, X, y):
        Xa = np.asarray(X)
        n_cols = Xa.shape[1]
        P = n_cols // 2
        imp = np.zeros(n_cols, dtype=float)
        if int(np.argmax(Xa[:, P])) % 2 == 0:
            imp[0] = 1.0
        self.feature_importances_ = imp
        return self


def test_hetero_vote_per_model_hit_frac_boundary_is_inclusive():
    """A member PASSES a feature when its hit-rate ``>= per_model_hit_frac`` (inclusive). The stub
    hits x0 in exactly 1 of 2 trials (rate 0.5): the member passes x0 at per_model_hit_frac=0.5
    (vote_frac 1.0) but fails it at 0.6 (vote_frac 0.0)."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((200, 3)), columns=["x0", "x1", "x2"])
    y = pd.Series((rng.random(200) < 0.5).astype(int))
    panel = {"alt": _AltTrialFI()}
    # random_state=2 is the verified config where the parity fingerprint realizes a 1-of-2 hit-rate.
    _, info_half = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=2, percentile=100,
        per_model_hit_frac=0.5, vote_threshold=0.5, random_state=2,
    )
    # Single-member panel -> vote_frac == member pass (0 or 1). At hit-rate 0.5 with the inclusive
    # >= boundary the member passes, so vote_frac is 1.0.
    assert info_half["vote_fraction"]["x0"] == 1.0, (
        f"a 1-of-2 hit-rate must PASS per_model_hit_frac=0.5 (>= is inclusive); "
        f"got {info_half['vote_fraction']['x0']}"
    )
    _, info_strict = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=2, percentile=100,
        per_model_hit_frac=0.6, vote_threshold=0.5, random_state=2,
    )
    assert info_strict["vote_fraction"]["x0"] == 0.0, (
        "a 1-of-2 hit-rate must FAIL per_model_hit_frac=0.6 (0.5 < 0.6)"
    )


# ===========================================================================
# (f) skill-weighting discriminator: a blind member is downweighted to the floor, flipping
#     a vote_threshold=0.7 decision from REJECT (equal) to ACCEPT (skill).
#     Findings coverage_asymmetry_wrappers-10 (the biz_value win) + param_axes-07.
# ===========================================================================


def _two_signal_with_blind_panel(seed: int = 0, n: int = 1500):
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = rng.standard_normal(n)
    logit = 1.6 * x0 - 1.4 * x1
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    cols = {"sig_0": x0, "sig_1": x1}
    for j in range(6):
        cols[f"noise_{j}"] = rng.standard_normal(n)
    panel = {
        "tree": RandomForestClassifier(n_estimators=fast_n_estimators(120), random_state=0),
        "dist": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "blind": _BlindFI(strategy="prior"),
    }
    return pd.DataFrame(cols), pd.Series(y), ["sig_0", "sig_1"], panel


@pytest.mark.slow
def test_biz_val_hetero_vote_skill_weighting_rescues_blind_vetoed_signal():
    """biz_value: skill weighting flips a real signal from REJECTED to ACCEPTED.

    Panel = {tree, dist, blind}. Both real members detect the 2-feature signal; the blind member
    (uniform FI, prior predictor) detects nothing and earns cv_skill == floor 0.05. At
    vote_threshold=0.7:
      - EQUAL weighting: vote_frac = 2/3 = 0.667 < 0.7  -> signal REJECTED (measured 0.667).
      - SKILL weighting: (w_tree + w_dist)/(w_tree + w_dist + 0.05) ~ 0.935 >= 0.7 -> ACCEPTED.
    Assert the flip on BOTH signal columns and pin the blind member at the floor. Floors carry
    >=15% headroom over the 0.7 threshold on the measured 0.935 / 0.667 gap."""
    X, y, signal, panel = _two_signal_with_blind_panel(seed=0)

    acc_eq, info_eq = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=3, vote_threshold=0.7,
        weight_by_cv_skill=False, random_state=0,
    )
    for s in signal:
        assert info_eq["vote_fraction"][s] == pytest.approx(2.0 / 3.0), (
            f"equal-weight vote_frac for {s} should be 2/3, got {info_eq['vote_fraction'][s]}"
        )
        assert s not in set(acc_eq), f"under equal weighting {s} must be REJECTED (2/3 < 0.7)"

    acc_sk, info_sk = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=3, vote_threshold=0.7,
        weight_by_cv_skill=True, cv_skill_folds=3, cv_skill_floor=0.05, random_state=0,
    )
    for s in signal:
        assert info_sk["vote_fraction"][s] >= 0.85, (
            f"skill-weighted vote_frac for {s} should clear 0.85 (measured ~0.935), "
            f"got {info_sk['vote_fraction'][s]}"
        )
        assert s in set(acc_sk), f"under skill weighting {s} must be ACCEPTED (~0.935 >= 0.7)"
    assert info_sk["model_weights"]["blind"] == pytest.approx(0.05), (
        "the near-chance blind member must be pinned at cv_skill_floor=0.05"
    )


def test_biz_val_hetero_vote_skill_weighting_rescues_blind_vetoed_signal_fast():
    """Fast representative of the skill-weighting flip (smaller n, fewer trials) so MLFRAME_FAST=1
    still exercises the equal-vs-skill decision flip on at least one signal column."""
    X, y, signal, panel = _two_signal_with_blind_panel(seed=0, n=700)
    acc_eq, info_eq = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=2, vote_threshold=0.7,
        weight_by_cv_skill=False, random_state=0,
    )
    acc_sk, info_sk = heterogeneous_relevance_vote(
        X, y, classification=True, models=panel, n_shadow_trials=2, vote_threshold=0.7,
        weight_by_cv_skill=True, cv_skill_folds=3, cv_skill_floor=0.05, random_state=0,
    )
    # At least one signal flips REJECT->ACCEPT; the blind member sits at the floor.
    flipped = [s for s in signal if s not in set(acc_eq) and s in set(acc_sk)]
    assert flipped, (
        f"skill weighting must flip >=1 signal from rejected to accepted; "
        f"eq={sorted(set(acc_eq) & set(signal))} sk={sorted(set(acc_sk) & set(signal))}"
    )
    assert info_sk["model_weights"]["blind"] == pytest.approx(0.05)
    assert set(info_eq["model_weights"].values()) == {1.0}


# ===========================================================================
# (g) _importance permutation-fallback (Pipeline exposes neither FI nor coef_).
#     Finding coverage_asymmetry_wrappers-11.
# ===========================================================================


def test_importance_permutation_fallback_ranks_signal_first():
    """A Pipeline(StandardScaler, Ridge) exposes neither feature_importances_ nor coef_, so
    ``_importance`` routes through permutation_importance. On ``y = 3*x0 + noise`` the permutation
    importance of x0 strictly exceeds every other column (measured x0~1.97 vs <=3e-4 elsewhere)."""
    est = make_pipeline(StandardScaler(), Ridge())
    assert not hasattr(est, "feature_importances_")
    assert not hasattr(est, "coef_")

    rng = np.random.default_rng(0)
    n = 300
    x0 = rng.standard_normal(n)
    X = np.column_stack([x0] + [rng.standard_normal(n) for _ in range(5)])
    y = 3.0 * x0 + 0.3 * rng.standard_normal(n)

    imp = _importance(est, X, y, random_state=0)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))
    assert imp[0] > imp[1:].max(), (
        f"permutation-fallback importance of the true signal x0 ({imp[0]:.4f}) must strictly "
        f"exceed all others (max {imp[1:].max():.4f})"
    )


def test_importance_permutation_fallback_subsample_is_seeded_deterministic():
    """For n > 1000 the fallback subsamples 1000 rows via default_rng(random_state); two fits with
    the same seed must produce bit-identical importance vectors (the subsample is reproducible)."""
    rng = np.random.default_rng(0)
    n = 2500
    x0 = rng.standard_normal(n)
    X = np.column_stack([x0] + [rng.standard_normal(n) for _ in range(5)])
    y = 3.0 * x0 + 0.3 * rng.standard_normal(n)

    imp_a = _importance(make_pipeline(StandardScaler(), Ridge()), X, y, random_state=0)
    imp_b = _importance(make_pipeline(StandardScaler(), Ridge()), X, y, random_state=0)
    assert np.array_equal(imp_a, imp_b), "seeded n>1000 subsample must yield identical importances"


def test_importance_coef_branch_collapses_multiclass():
    """A bare LogisticRegression exposes coef_; for 3-class y the |coef| is (3, p) and ``_importance``
    collapses it per-feature via max over classes -> length-p finite vector with the signal on top."""
    rng = np.random.default_rng(0)
    n = 600
    x0 = rng.standard_normal(n)
    X = np.column_stack([x0] + [rng.standard_normal(n) for _ in range(4)])
    # 3-class target driven by x0 terciles.
    y = np.digitize(x0, np.quantile(x0, [1.0 / 3, 2.0 / 3]))
    est = LogisticRegression(max_iter=2000)
    imp = _importance(est, X, y, random_state=0)
    assert imp.shape == (X.shape[1],)
    assert np.all(np.isfinite(imp))
    assert imp[0] == imp.max(), "the |coef| signal column should dominate after the per-class max-collapse"


# ===========================================================================
# (h) determinism: same seed -> identical accepted + bit-equal vote_fraction;
#     different seed -> vote_fraction differs on >=1 column.
#     Finding coverage_asymmetry_wrappers-13 (seed-sensitivity guard).
# ===========================================================================


def test_hetero_vote_same_seed_bit_identical():
    """Two fits with the same random_state produce identical accepted lists and bit-equal
    vote_fraction dicts (the shadow rng is fully seeded)."""
    X, y, _, _ = _clf_data(seed=0)
    a1, i1 = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=3, random_state=0)
    a2, i2 = heterogeneous_relevance_vote(X, y, classification=True, n_shadow_trials=3, random_state=0)
    assert a1 == a2
    assert i1["vote_fraction"] == i2["vote_fraction"]


def test_hetero_vote_different_seed_changes_vote_fraction():
    """A different random_state redraws the shadows, so the borderline-column vote fractions must
    differ on at least one column (guards against an ignored random_state). Using percentile=50
    (a low shadow bar -> many borderline noise columns) the seed change reliably moves several
    fractions; measured: seed 0 vs seed 42 differ on multiple columns at percentile=100 too."""
    X, y, _, _ = _clf_data(seed=0)
    _, i0 = heterogeneous_relevance_vote(
        X, y, classification=True, n_shadow_trials=3, percentile=50, vote_threshold=0.5, random_state=0
    )
    _, i42 = heterogeneous_relevance_vote(
        X, y, classification=True, n_shadow_trials=3, percentile=50, vote_threshold=0.5, random_state=42
    )
    diffs = [k for k in i0["vote_fraction"] if i0["vote_fraction"][k] != i42["vote_fraction"][k]]
    assert diffs, "different random_state must change vote_fraction on >=1 column (random_state ignored?)"
