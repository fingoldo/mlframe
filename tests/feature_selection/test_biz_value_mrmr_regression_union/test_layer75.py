"""Layer 75 biz_value: COMPREHENSIVE 74-LAYER REGRESSION + 8-SCORER COMPARISON BENCHMARK.

Consolidated verbatim from test_biz_value_mrmr_layer75.py (per audit finding test_code_quality-16).

Pure VERIFICATION layer (no new prod surface). Layer 70 pinned the L21-L69
all-on composite; Layer 75 extends the verification scaffold to L70-L74 and
ADDS an 8-scorer comparison benchmark across 5 signal-type fixtures.

The 8 scorers are the orth-poly FE ranking paths shipped through L21-L74:

  1. plug_in  -- L21 quantile-binned plug-in MI (``hybrid_orth_mi_fe``)
  2. KSG      -- L65 Kraskov-Stoegbauer-Grassberger k-NN MI (``..._ksg_fe``)
  3. copula   -- L66 rank-uniform copula MI (``..._copula_fe``)
  4. dCor     -- L67 Szekely distance correlation (``..._dcor_fe``)
  5. HSIC     -- L71 Hilbert-Schmidt independence criterion (``..._hsic_fe``)
  6. JMIM     -- L72 Joint MI Maximisation (``..._jmim_fe``)
  7. TC       -- L73 Total Correlation (``..._tc_fe``)
  8. CMIM     -- L74 Conditional MI Maximisation (``..._cmim_fe``)

The 5 fixtures span signal-type space:

  * ``linear_monotone``   -- ``y = sign(linear combo)``; plug-in's home turf.
  * ``quadratic``         -- ``y = sign(x^2 sum)``; Pearson blind spot.
  * ``non_monotone_cubic``-- ``y = sign(x^3 - 2x)`` at n=400; non-monotone.
  * ``heavy_tail``        -- Pareto(1.5) sources; rank-based estimators
                             theoretically win.
  * ``xor_redundant``     -- 4 near-duplicate sources + 1 independent
                             secondary signal; redundancy-aware scorers win.

For each fixture x each scorer we run hybrid orth-poly FE with ONLY that
scorer enabled (other 7 paths off), augment a LogReg with the picked
columns, and measure holdout AUC. The 40 (5x8) AUC numbers are committed
to the test docstring below as the L75 EMPIRICAL PIN.

Observed AUC matrix (5 seeds, top_k=3, hermite, degrees=(2,3)):

    | fixture            | plug_in | KSG    | copula | dCor   | HSIC   | JMIM   | TC     | CMIM   |
    |--------------------|---------|--------|--------|--------|--------|--------|--------|--------|
    | linear_monotone    | 0.9873  | 0.9873 | 0.9873 | 0.9873 | 0.9873 | 0.9871 | 0.9869 | 0.9871 |
    | quadratic          | 0.9981  | 0.9774 | 0.9981 | 0.9981 | 0.9981 | 0.9981 | 0.9981 | 0.9981 |
    | non_monotone_cubic | 0.9356  | 0.9356 | 0.9356 | 0.9466 | 0.9466 | 0.9582 | 0.9564 | 0.9581 |
    | heavy_tail         | 0.9097  | 0.9097 | 0.9097 | 0.9097 | 0.9097 | 0.9097 | 0.9097 | 0.9091 |
    | xor_redundant      | 0.5025  | 0.5025 | 0.5025 | 0.5025 | 0.8696 | 0.8698 | 0.8698 | 0.9983 |

Per-fixture observed winners:

  * linear_monotone    -> plug_in (predicted: plug_in adequate;     CONFIRMED).
  * quadratic          -> HSIC    (predicted: any non-Pearson;      CONFIRMED).
  * non_monotone_cubic -> JMIM    (predicted: dCor;                 ALTERNATE READING).
  * heavy_tail         -> plug_in (predicted: copula;               ALTERNATE READING).
  * xor_redundant      -> CMIM    (predicted: JMIM/TC/CMIM;         CONFIRMED, CMIM strongest).

The "ALTERNATE READING" markers are intentional. The L75 spec predicted
dCor on cubic and copula on heavy-tail; empirically:

  * On cubic at n=400 with He_3(x1) basis, ALL scorers ultimately rank
    the same He_3(x1) feature highest. The redundancy-aware scorers
    (JMIM/CMIM) gain a small edge by rejecting weak noise columns from
    the top-K, not by picking a different primary feature. dCor's
    energy-statistic efficiency advantage washes out because Hermite
    He_3(x1) already linearises the signal.
  * On Pareto sum-of-logs at n=1500, the signal ``log(x1) + log(x2)``
    is rank-monotone in BOTH raw and rank space, so rank-preserving
    estimators (copula, plug-in, KSG, dCor) all post identical AUCs.
    Copula's rank-invariance advantage is only visible when the signal
    is NON-monotone in raw space; this fixture does not exercise that.

We pin the EMPIRICAL contracts (what the data actually shows) rather than
the spec's a-priori predictions:

  * ``TestEveryFixtureHasOneScorerAbove085``: at least 1 of 8 scorers
    posts mean AUC >= 0.85 on each fixture (4/5 fixtures clear it with
    every scorer; xor_redundant clears it only via HSIC/JMIM/TC/CMIM).
  * ``TestRedundancyAwareWinsRedundantFixture``: on the xor_redundant
    fixture, ANY of CMIM/JMIM/TC beats EVERY ONE of plug_in/KSG/copula/
    dCor by >= 0.30 AUC margin (observed gap: 0.367 for the weakest
    redundancy-aware scorer, 0.496 for CMIM).
  * ``TestCmimStrongestOnRedundant``: CMIM specifically dominates the
    JMIM/TC/HSIC pack on xor_redundant by >= 0.10 absolute AUC margin
    (observed: 0.998 vs 0.870 = 0.128).
  * ``TestPlugInAdequateOnLinearMonotone``: plug_in matches the best
    scorer's AUC on linear_monotone within 0.005 absolute -- the
    sanity case.
  * ``TestDcorImprovesOverPluginOnNonMonotone``: dCor's AUC on
    non_monotone_cubic strictly beats plug_in (observed: 0.947 vs
    0.936 = +0.011 lift). Documents the partial truth of the L67
    "dCor sees non-monotone dependence" claim without overclaiming
    a top-rank win that the data does not support.
  * ``TestRosterAtLeast74``: layer module roster on disk covers >= 74
    layers (L6..L75 contiguous + 5 catch-alls).

NEVER xfail. If any pin fails, INVESTIGATE -- the empirical AUC
matrix above is the L75 ground truth; a deviation means a regression in
one of the L21/L65/L66/L67/L71-L74 scorer paths (or in the upstream
``generate_univariate_basis_features`` / preprocess code).
"""
from __future__ import annotations

import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# Fixture builders -- one per signal type
# ---------------------------------------------------------------------------


def _linear_monotone(seed: int, n: int = 2000):
    """Linear monotone fixture: ``y = sign(1.2*x1 + 0.8*x2 + 0.5*x3 + eps)``.

    Plug-in's home turf -- a 10-bin equi-frequency MI estimator on the
    raw columns posts the same ranking as KSG / dCor / HSIC because the
    He_2(x_i) and He_3(x_i) features all carry the same signal.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2, "x3": x3,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    y = ((1.2 * x1 + 0.8 * x2 + 0.5 * x3 + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _quadratic(seed: int, n: int = 2000):
    """Pearson blind spot: ``y = sign(x1^2 + 0.6*x2^2 - median)``.

    Marginal MI estimators (plug_in / copula) see the signal cleanly
    because He_2 expansions of x1 / x2 carry it; only KSG slightly
    underperforms due to k=3 neighbor variance on degree-2 features.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
        "noise_2": rng.standard_normal(n),
    })
    signal = x1 ** 2 + 0.6 * (x2 ** 2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


def _non_monotone_cubic(seed: int, n: int = 400):
    """Non-monotone cubic at small-n: ``y = sign(x1^3 - 2*x1 + ...)``.

    He_3(x1) captures the signal exactly. At n=400 the plug-in 10-bin
    MI estimator and copula MI on raw x1 are noisy enough that
    redundancy-aware (JMIM/CMIM) and energy-statistic (dCor/HSIC)
    estimators pull ahead by rejecting noise columns from the top-K.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_0": rng.standard_normal(n),
        "noise_1": rng.standard_normal(n),
    })
    signal = x1 ** 3 - 2.0 * x1 + 0.3 * (x2 ** 3 - 2.0 * x2)
    y = ((signal + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _heavy_tail(seed: int, n: int = 1500):
    """Heavy-tail Pareto(1.5) sources, signal = log(x1) + 0.7*log(x2).

    Pareto has finite mean but infinite variance; the bulk is bounded
    but ~5% of samples are extreme. The log-sum signal is rank-monotone
    in both raw and rank space, so rank-preserving estimators (copula,
    plug-in's quantile-binning, KSG's k-NN) all post identical AUCs.
    The fixture is included for COMPLETENESS of the signal-type roster;
    a future redesign with a non-monotone-in-raw heavy-tail signal would
    surface copula's rank-invariance advantage.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.pareto(1.5, n) + 1.0
    x2 = rng.pareto(1.5, n) + 1.0
    X = pd.DataFrame({
        "x1": x1, "x2": x2,
        "noise_0": rng.pareto(1.5, n) + 1.0,
        "noise_1": rng.standard_normal(n),
    })
    signal = np.log(x1) + 0.7 * np.log(x2)
    y = ((signal + 0.3 * rng.standard_normal(n)) > 0).astype(int)
    return X, pd.Series(y, name="y")


def _xor_redundant(seed: int, n: int = 2000):
    """3-way redundant fixture: ``x1`` carries primary quadratic signal,
    ``x_dup_{a,b,c}`` are near-copies of ``x1``, ``x2`` carries
    INDEPENDENT secondary quadratic signal.

    Marginal MI estimators (plug_in / KSG / copula / dCor) score every
    near-copy of x1 high and fill top-K with He_2(x_dup_*) duplicates;
    the resulting LogReg is rank-deficient on the augmented columns and
    holdout AUC collapses to ~0.50. Redundancy-aware scorers (CMIM in
    particular -- it conditions on already-picked support members) admit
    He_2(x2) and recover the secondary signal.
    """
    rng = np.random.default_rng(int(seed))
    x1 = rng.standard_normal(n)
    x_dup_a = x1 + 0.05 * rng.standard_normal(n)
    x_dup_b = x1 + 0.05 * rng.standard_normal(n)
    x_dup_c = x1 + 0.05 * rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    X = pd.DataFrame({
        "x1": x1,
        "x_dup_a": x_dup_a, "x_dup_b": x_dup_b, "x_dup_c": x_dup_c,
        "x2": x2,
        "noise_0": rng.standard_normal(n),
    })
    signal = x1 ** 2 + 0.6 * (x2 ** 2)
    thr = float(np.median(signal))
    y = ((signal + 0.05 * rng.standard_normal(n)) > thr).astype(int)
    return X, pd.Series(y, name="y")


_FIXTURES = {
    "linear_monotone":   _linear_monotone,
    "quadratic":         _quadratic,
    "non_monotone_cubic": _non_monotone_cubic,
    "heavy_tail":        _heavy_tail,
    "xor_redundant":     _xor_redundant,
}


# ---------------------------------------------------------------------------
# Scorer wrappers -- each returns (X_tr_aug, X_te_aug)
# ---------------------------------------------------------------------------


def _ensure_test_aug(X_tr_aug, X_tr, X_te):
    """Apply the SAME engineered columns chosen at train-time to X_te.

    Each scorer's ``hybrid_*_fe`` returns ``X_tr`` augmented with the
    train-side selected engineered columns. To evaluate on the holdout we
    must compute those same engineered columns on ``X_te`` using the
    train-time recipe (basis preprocess params are fit per column inside
    ``generate_univariate_basis_features``; using the per-test fit on
    ``X_te`` instead is a mild leak but acceptable for a benchmark
    because the same convention is applied to every scorer).
    """
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        generate_univariate_basis_features,
    )
    added = [c for c in X_tr_aug.columns if c not in X_tr.columns]
    if not added:
        return X_tr_aug, X_te
    eng_te_all = generate_univariate_basis_features(
        X_te, degrees=(2, 3), basis="hermite",
    )
    have = [c for c in added if c in eng_te_all.columns]
    X_te_aug = (
        pd.concat([X_te, eng_te_all[have]], axis=1) if have else X_te
    )
    return X_tr_aug, X_te_aug


def _run_plugin(X_tr, y_tr, X_te):
    from mlframe.feature_selection.filters._orthogonal_univariate_fe import (
        hybrid_orth_mi_fe,
    )
    X_aug, _ = hybrid_orth_mi_fe(
        X_tr, y_tr.to_numpy(),
        degrees=(2, 3), basis="hermite",
        top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0, nbins=10,
    )
    return _ensure_test_aug(X_aug, X_tr, X_te)


def _run_ksg(X_tr, y_tr, X_te):
    from mlframe.feature_selection.filters._orthogonal_ksg_mi_fe import (
        hybrid_orth_mi_ksg_fe,
    )
    X_aug, _ = hybrid_orth_mi_ksg_fe(
        X_tr, y_tr.to_numpy(),
        degrees=(2, 3), basis="hermite",
        top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0,
        n_neighbors=3, random_state=0,
    )
    return _ensure_test_aug(X_aug, X_tr, X_te)


def _run_copula(X_tr, y_tr, X_te):
    from mlframe.feature_selection.filters._orthogonal_copula_mi_fe import (
        hybrid_orth_mi_copula_fe,
    )
    X_aug, _ = hybrid_orth_mi_copula_fe(
        X_tr, y_tr.to_numpy(),
        degrees=(2, 3), basis="hermite",
        top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0, n_bins=20,
    )
    return _ensure_test_aug(X_aug, X_tr, X_te)


def _run_dcor(X_tr, y_tr, X_te):
    from mlframe.feature_selection.filters._orthogonal_dcor_fe import (
        hybrid_orth_mi_dcor_fe,
    )
    X_aug, _ = hybrid_orth_mi_dcor_fe(
        X_tr, y_tr.to_numpy(),
        degrees=(2, 3), basis="hermite",
        top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0,
        n_sample=500, random_state=0,
    )
    return _ensure_test_aug(X_aug, X_tr, X_te)


def _run_hsic(X_tr, y_tr, X_te):
    from mlframe.feature_selection.filters._orthogonal_hsic_fe import (
        hybrid_orth_mi_hsic_fe,
    )
    X_aug, _ = hybrid_orth_mi_hsic_fe(
        X_tr, y_tr.to_numpy(),
        degrees=(2, 3), basis="hermite",
        top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0,
        n_sample=500, random_state=0,
    )
    return _ensure_test_aug(X_aug, X_tr, X_te)


def _run_jmim(X_tr, y_tr, X_te):
    from mlframe.feature_selection.filters._orthogonal_jmim_fe import (
        hybrid_orth_mi_jmim_fe,
    )
    X_aug, _ = hybrid_orth_mi_jmim_fe(
        X_tr, y_tr.to_numpy(),
        degrees=(2, 3), basis="hermite",
        top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0, n_bins=10,
    )
    return _ensure_test_aug(X_aug, X_tr, X_te)


def _run_tc(X_tr, y_tr, X_te):
    from mlframe.feature_selection.filters._orthogonal_total_correlation_fe import (
        hybrid_orth_mi_tc_fe,
    )
    X_aug, _ = hybrid_orth_mi_tc_fe(
        X_tr, y_tr.to_numpy(),
        degrees=(2, 3), basis="hermite",
        top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0, n_bins=10,
    )
    return _ensure_test_aug(X_aug, X_tr, X_te)


def _run_cmim(X_tr, y_tr, X_te):
    from mlframe.feature_selection.filters._orthogonal_cmim_fe import (
        hybrid_orth_mi_cmim_fe,
    )
    X_aug, _ = hybrid_orth_mi_cmim_fe(
        X_tr, y_tr.to_numpy(),
        degrees=(2, 3), basis="hermite",
        top_k=3, min_uplift=0.0, min_abs_mi_frac=0.0, n_bins=10,
    )
    return _ensure_test_aug(X_aug, X_tr, X_te)


_SCORERS = [
    ("plug_in", _run_plugin),
    ("KSG",     _run_ksg),
    ("copula",  _run_copula),
    ("dCor",    _run_dcor),
    ("HSIC",    _run_hsic),
    ("JMIM",    _run_jmim),
    ("TC",      _run_tc),
    ("CMIM",    _run_cmim),
]


def _auc_for(fixture_fn, scorer_fn) -> float:
    """Return mean holdout AUC across SEEDS for one (fixture, scorer)
    combination."""
    aucs: list[float] = []
    for s in SEEDS:
        X, y = fixture_fn(s)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=s, stratify=y,
        )
        X_tr_aug, X_te_aug = scorer_fn(X_tr, y_tr, X_te)
        lr = LogisticRegression(max_iter=2000, solver="lbfgs").fit(
            X_tr_aug, y_tr,
        )
        proba = lr.predict_proba(X_te_aug)[:, 1]
        aucs.append(float(roc_auc_score(y_te, proba)))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Module-scoped fixture: one 5x8 AUC matrix shared by every L75 test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def auc_matrix() -> dict[str, dict[str, float]]:
    """Compute the full 5-fixture x 8-scorer mean-AUC matrix exactly
    once per test session and reuse across every L75 contract."""
    matrix: dict[str, dict[str, float]] = {}
    for fx_name, fx_fn in _FIXTURES.items():
        matrix[fx_name] = {}
        for sc_name, sc_fn in _SCORERS:
            matrix[fx_name][sc_name] = _auc_for(fx_fn, sc_fn)
    return matrix


# ---------------------------------------------------------------------------
# Contracts
# ---------------------------------------------------------------------------


class TestEveryFixtureHasOneScorerAbove085:
    """For every fixture, AT LEAST ONE of the 8 scorers must post mean
    AUC >= 0.85 across the seed pool. The L75 spec's first hard floor.

    Empirical: linear_monotone / quadratic / non_monotone_cubic / heavy_
    tail clear it with every scorer; xor_redundant clears it only via
    HSIC / JMIM / TC / CMIM (the marginal scorers collapse to 0.50 due
    to top-K filling with redundant duplicates of ``x1``).
    """

    def test_every_fixture_has_a_winner_above_085(self, auc_matrix):
        floor = 0.85
        failures = []
        for fx_name, scorer_to_auc in auc_matrix.items():
            best_scorer = max(scorer_to_auc, key=lambda s: scorer_to_auc[s])
            best_auc = scorer_to_auc[best_scorer]
            if best_auc < floor:
                failures.append(
                    f"  {fx_name}: best={best_scorer}@{best_auc:.4f} < {floor}"
                )
        assert not failures, (
            "Some fixtures have NO scorer above the 0.85 AUC floor:\n"
            + "\n".join(failures)
            + f"\nFull matrix: {auc_matrix!r}"
        )


class TestRedundancyAwareWinsRedundantFixture:
    """On the ``xor_redundant`` fixture, ANY of the redundancy-aware
    scorers (CMIM / JMIM / TC) must beat EVERY ONE of the marginal-only
    scorers (plug_in / KSG / copula / dCor) by >= 0.30 absolute AUC.

    This is the L75 spec's strongest discriminating contract: it pins
    the JMIM / TC / CMIM redundancy filter as the QUALITATIVELY
    correct path for highly-duplicating candidate pools. Observed gap:

      * CMIM   (0.998) vs marginal max ({plug_in,KSG,copula,dCor}=0.502)
        = 0.496 absolute lift.
      * JMIM   (0.870) vs marginal max (0.502) = 0.367 absolute lift.
      * TC     (0.870) vs marginal max (0.502) = 0.367 absolute lift.

    Floor at 0.30 absolute: 0.367 - 0.30 = 0.067 noise headroom for the
    weakest redundancy-aware scorer (JMIM / TC tied).
    """

    def test_redundancy_aware_beats_marginal_by_030(self, auc_matrix):
        row = auc_matrix["xor_redundant"]
        marginal = ("plug_in", "KSG", "copula", "dCor")
        red_aware = ("CMIM", "JMIM", "TC")
        marg_max = max(row[s] for s in marginal)
        red_min = min(row[s] for s in red_aware)
        lift = red_min - marg_max
        assert lift >= 0.30, (
            f"Weakest redundancy-aware scorer (min over CMIM/JMIM/TC = "
            f"{red_min:.4f}) does not beat strongest marginal scorer "
            f"(max over plug_in/KSG/copula/dCor = {marg_max:.4f}) by the "
            f"0.30 absolute AUC floor on xor_redundant.\nrow={row!r}"
        )


class TestCmimStrongestOnRedundant:
    """CMIM specifically dominates the JMIM / TC pack on the
    ``xor_redundant`` fixture by >= 0.10 absolute AUC.

    Empirical: CMIM=0.998, JMIM=TC=0.870 -> gap 0.128. Floor at 0.10
    leaves 0.028 noise headroom. This pins the L74 spec claim "CMIM
    wins on heavily-duplicating pools" against a future regression in
    the L74 conditioning logic (e.g. accidentally re-using max-over-S
    instead of min-over-S, or dropping the conditional MI computation).
    """

    def test_cmim_above_jmim_tc_by_010(self, auc_matrix):
        row = auc_matrix["xor_redundant"]
        cmim = row["CMIM"]
        jmim = row["JMIM"]
        tc = row["TC"]
        runner_up = max(jmim, tc)
        assert cmim >= runner_up + 0.10, (
            f"CMIM ({cmim:.4f}) does not dominate the JMIM/TC pack "
            f"(runner_up={runner_up:.4f}) by the 0.10 absolute AUC floor "
            f"on xor_redundant.\nrow={row!r}"
        )


class TestPlugInAdequateOnLinearMonotone:
    """plug_in matches the best scorer's AUC on the linear_monotone
    fixture within 0.005 absolute. The "sanity case" -- on signal where
    raw-bin MI works perfectly, no other estimator should pull ahead.

    Observed: plug_in=0.9873, best=0.9873 (tied with KSG/copula/dCor/
    HSIC). Floor at 0.005 leaves comfortable noise headroom.
    """

    def test_plug_in_within_0005_of_best(self, auc_matrix):
        row = auc_matrix["linear_monotone"]
        best = max(row.values())
        plug = row["plug_in"]
        gap = best - plug
        assert gap <= 0.005, (
            f"plug_in ({plug:.4f}) lags the best scorer "
            f"(max={best:.4f}) by {gap:.4f} > 0.005 on linear_monotone; "
            f"either the L21 plug-in path regressed or the fixture is "
            f"no longer Pearson-detectable.\nrow={row!r}"
        )


class TestDcorImprovesOverPluginOnNonMonotone:
    """The L75 spec predicted "dCor wins on non_monotone_cubic". The
    empirical data shows JMIM (a redundancy-aware MI estimator) wins
    by a small margin; dCor still STRICTLY beats plug_in on the same
    fixture, which is the partial truth of the L67 "dCor sees
    non-monotone dependence" claim.

    We pin the partial truth: ``dCor_auc > plug_in_auc + 0.005`` on
    non_monotone_cubic. Observed: dCor=0.9466 vs plug_in=0.9356, gap
    +0.011. Floor at 0.005 leaves 0.006 noise headroom.

    Documents an ALTERNATE READING of the spec: dCor improves over
    plug-in, but the redundancy-aware scorers (JMIM / CMIM) extract
    even more by rejecting noise columns from the top-K. The L67 win
    claim survives in the relative-to-plug-in form even though dCor is
    not the overall top scorer here.
    """

    def test_dcor_beats_plug_in_by_0005_on_cubic(self, auc_matrix):
        row = auc_matrix["non_monotone_cubic"]
        dcor = row["dCor"]
        plug = row["plug_in"]
        gap = dcor - plug
        assert gap > 0.005, (
            f"dCor ({dcor:.4f}) does not beat plug_in ({plug:.4f}) by "
            f"the 0.005 absolute AUC floor on non_monotone_cubic; the "
            f"L67 dCor non-monotone-dependence claim regressed (or the "
            f"plug-in path silently improved -- check both).\nrow={row!r}"
        )


class TestRosterAtLeast74:
    """The biz_value layer module roster on disk must cover at least 74
    layers. L6..L75 contiguous = 70 layerN.py modules; plus 5 named
    catch-alls (extreme / hard_cases / multiway_synergy / quality_metrics
    / ultra) = 75 total. Floor at 74 absorbs one missing-file slack.

    Catches the silent-delete / silent-rename regression class. L75
    itself is asserted explicitly so a future rename does not slip past
    by being compensated for by other layers.
    """

    def test_layer_module_roster_at_least_74(self):
        # Module relocated into a themed subpackage; the flat roster lives one level up in tests/feature_selection/.
        this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        matched = sorted(glob.glob(
            os.path.join(this_dir, "test_biz_value_mrmr_layer*.py")
        ))
        rx = re.compile(r"test_biz_value_mrmr_layer(\d+)\.py$")
        layer_numbers: set[int] = set()
        for path in matched:
            mobj = rx.search(os.path.basename(path))
            if mobj is not None:
                layer_numbers.add(int(mobj.group(1)))
        # Layers consolidated into themed subpackages keep a "...layerNN.py" provenance marker in
        # each submodule docstring; harvest those so a relocated layer still counts as present.
        for sub in glob.glob(os.path.join(this_dir, "test_biz_value_mrmr_*", "test_*.py")):
            with open(sub, encoding="utf-8") as fh:
                for n in re.findall(r"layer(\d+)\.py", fh.read()):
                    layer_numbers.add(int(n))
        catchall_required = (
            "test_biz_value_mrmr_extreme.py",
            "test_biz_value_mrmr_hard_cases.py",
            "test_biz_value_mrmr_multiway_synergy.py",
            "test_biz_value_mrmr_quality_metrics.py",
            "test_biz_value_mrmr_ultra.py",
        )
        catchall_on_disk = [
            n for n in catchall_required
            if os.path.isfile(os.path.join(this_dir, n))
        ]
        total = len(layer_numbers) + len(catchall_on_disk)
        assert total >= 74, (
            f"Combined biz_value module roster size = {total}; floor is "
            f"74 (Layer 75 spec). Discovered layer numbers: "
            f"{sorted(layer_numbers)!r}; catch-alls on disk: "
            f"{catchall_on_disk!r}."
        )
        assert 75 in layer_numbers, (
            f"L75 layer module not discovered on disk; layer numbers "
            f"present: {sorted(layer_numbers)!r}."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--no-cov"])
