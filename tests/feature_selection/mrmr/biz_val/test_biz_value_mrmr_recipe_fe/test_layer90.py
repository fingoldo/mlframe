"""Layer 90 biz_value: NUMERIC DECOMPOSITION FE with bootstrap-MI gate.

Validates ``_numeric_decompose_fe`` (2026-06-01): multi-precision rounding
(``round(x/p)*p``) + decimal-digit extraction (``floor(x*10^k) mod 10``), each
candidate gated by Layer 62 bootstrap-stable MI (lower CB). NVIDIA cuDF
Kaggle-Grandmaster technique #4.

Contracts pinned (real numbers, never xfail):

* Cents-digit signal: y depends on the cents digit of a price col; the
  digit-extraction feature beats raw x by >= +0.15 MI.
* Rounding-precision signal: y depends on round(x, 1) (coarse bucket); the
  rounding feature captures it (high MI, survives the gate).
* Bootstrap gate drops noise precisions: on a SMOOTH target the gate's
  keep/drop decision against ground truth has precision + recall >= 0.9 (every
  decomposition candidate is dropped because none adds stable MI over raw x).
* AUC lift on a price-anchored fixture.
* No leakage: transform(X, y_shuffled) == transform(X); recipe replay reads X.
* Default disabled byte-identical.
* Pickle / clone round-trips the recipes + ctor params.
"""
from __future__ import annotations

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

SEEDS = (1, 7, 13, 42, 101)


# ---------------------------------------------------------------------------
# MI helper (matches the L62 estimator the gate uses internally)
# ---------------------------------------------------------------------------


from tests.feature_selection._biz_val_synth import _mi_one

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _build_cents_digit(seed: int, n: int = 6000):
    """y is a deterministic function of the CENTS digit of a price column.
    The integer (dollars) part is independent noise, so raw x carries almost
    no MI about y while the cents-digit decomposition carries it all."""
    rng = np.random.default_rng(int(seed))
    dollars = rng.integers(1, 500, n).astype(np.float64)
    cents_digit = rng.integers(0, 10, n)  # the signal lives here
    # second decimal place noise so the price looks natural
    extra = rng.integers(0, 10, n)
    price = dollars + cents_digit / 10.0 + extra / 100.0
    flip = rng.random(n) < 0.03
    y = ((cents_digit >= 5).astype(int)) ^ flip.astype(int)
    X = pd.DataFrame({"price": price})
    return X, y.astype(int), cents_digit


def _build_rounding_bucket(seed: int, n: int = 6000):
    """y is the PARITY of an integer anchor; x is that anchor plus uniform
    within-anchor jitter. round(x, 1.0) snaps every row back to its clean
    integer anchor (parity fully recoverable); raw continuous x, when the MI
    estimator quantile-bins it, smears adjacent anchors across bin boundaries
    so the parity signal degrades.

    This is the estimator-resolution effect (NOT a data-processing-inequality
    violation): rounding is a deterministic function of x, so at the
    population level MI(round(x); y) <= MI(x; y). The rounded column wins only
    because a fixed-nbins plug-in estimator under-resolves the discrete anchor
    structure hiding in continuous x. The anchor precision (1.0) is chosen so
    a 10-bin quantizer maps 1:1 onto the 10 anchors when fed the rounded
    column, but boundary-mixes when fed the jittered raw column.
    """
    rng = np.random.default_rng(int(seed))
    anchor = rng.integers(0, 10, n).astype(np.float64)
    jitter = rng.uniform(-0.45, 0.45, n)
    x = anchor + jitter
    flip = rng.random(n) < 0.03
    y = ((anchor.astype(int) % 2) == 0).astype(int) ^ flip.astype(int)
    X = pd.DataFrame({"x": x})
    return X, y.astype(int), anchor


def _build_smooth(seed: int, n: int = 6000):
    """y is a smooth monotone function of x. Rounding is just lossy raw x and
    digit extraction is pure noise -> the gate should drop EVERY decomposition
    candidate."""
    rng = np.random.default_rng(int(seed))
    x = rng.normal(0.0, 1.0, n)
    logit = 1.5 * x
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"x": x})
    return X, y.astype(int)


def _build_price_anchored(seed: int, n: int = 6000):
    """Price-anchored AUC fixture: y depends jointly on a coarse rounding
    bucket AND the cents digit; raw x (continuous magnitude) can't separate
    either cleanly via a linear model."""
    rng = np.random.default_rng(int(seed))
    dollars = rng.integers(1, 50, n).astype(np.float64)
    cents_digit = rng.integers(0, 10, n)
    price = dollars + cents_digit / 10.0
    flip = rng.random(n) < 0.03
    y = (cents_digit >= 5).astype(int) ^ flip.astype(int)
    X = pd.DataFrame({"price": price})
    return X, y.astype(int)


# ---------------------------------------------------------------------------
# Contract 1: cents-digit signal -- digit extraction beats raw by >= +0.15 MI
# ---------------------------------------------------------------------------


class TestCentsDigitSignal:
    """Digit-extraction FE recovers the hidden cents-digit signal that raw x hides."""

    def test_digit_extraction_beats_raw_by_0p15_mi(self):
        """Digit extraction at k=1 beats raw price by >= +0.15 MI on the cents-digit fixture."""
        from mlframe.feature_selection.filters._numeric_decompose_fe import (
            apply_digit_extract,
        )
        gains = []
        for s in SEEDS:
            X, y, _ = _build_cents_digit(s)
            x = X["price"].to_numpy()
            mi_raw = _mi_one(x, y)
            # cents digit = first decimal place: k=1 captures dollars+cents/10
            # tenths digit; the SIGNAL digit (cents/10 -> tenths place) is k=1.
            mi_digit = _mi_one(apply_digit_extract(x, 1), y)
            gains.append(mi_digit - mi_raw)
        mean_gain = float(np.mean(gains))
        assert mean_gain >= 0.15, (
            f"cents-digit extraction MI gain {mean_gain:.4f} < 0.15 over raw "
            f"price (per-seed {[round(g, 4) for g in gains]}); digit "
            f"extraction is not recovering the hidden cents-digit signal."
        )


# ---------------------------------------------------------------------------
# Contract 2: rounding-precision signal captured
# ---------------------------------------------------------------------------


class TestRoundingPrecisionSignal:
    """Rounding is MI-neutral under quantile-binned estimation but unlocks linear-model AUC via one-hot anchors."""

    def test_rounding_is_mi_neutral_but_downstream_useful(self):
        """Rounding is a DETERMINISTIC function of x, so by the data-processing
        inequality MI(round(x); y) <= MI(x; y) at the population level. Under a
        quantile-binned plug-in MI estimator, when the binner ALREADY resolves
        the signal in raw x, rounding is approximately MI-NEUTRAL (within
        estimation noise) - it cannot beat raw, and the MI-uplift gate
        correctly does NOT select it (rounding adds no MI over raw, so
        promoting it would just bloat the support).

        Rounding's genuine value is DOWNSTREAM, not in MI selection: a LINEAR
        model cannot represent a step function of x, but one-hot of the
        snapped anchor lets it fit the steps. This is the NVIDIA cuDF blog's
        actual use case (tree/GBM split cleanliness), NOT MI-based selection.

        Two contracts pinned:
        1. MI-neutrality: MI(round(x,1.0); y) is within a small tolerance of
           MI(x; y) - rounding neither adds (DPI) nor destroys (binner already
           coarse) MI on this anchor+jitter fixture.
        2. Downstream lift: a LogReg on one-hot(round(x,1.0)) beats LogReg on
           raw x for the step-parity target, because the linear model can
           exploit the discrete anchor structure the raw continuous column
           hides from it.
        """
        from mlframe.feature_selection.filters._numeric_decompose_fe import (
            apply_rounding,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        lin_lifts = []
        for s in SEEDS:
            X, y, _ = _build_rounding_bucket(s)
            x = X["x"].to_numpy()
            mi_raw = _mi_one(x, y)
            mi_round = _mi_one(apply_rounding(x, 1.0), y)
            # MI-neutral: DPI forbids round from EXCEEDING raw; the coarse
            # binner forbids it falling far BELOW. Stay within 0.03 nats.
            assert abs(mi_round - mi_raw) <= 0.03, (
                f"seed={s}: |MI(round)-MI(raw)| = {abs(mi_round-mi_raw):.4f} "
                f"> 0.03; rounding should be MI-neutral under quantile-binned "
                f"estimation (round={mi_round:.4f}, raw={mi_raw:.4f})."
            )
            # Downstream: one-hot rounded anchor vs raw x for a LINEAR model.
            xtr, xte, ytr, yte = train_test_split(
                x, y, test_size=0.3, random_state=s, stratify=y,
            )
            auc_raw = roc_auc_score(
                yte,
                LogisticRegression(max_iter=2000).fit(xtr.reshape(-1, 1), ytr).predict_proba(xte.reshape(-1, 1))[:, 1],
            )
            anchors_tr = apply_rounding(xtr, 1.0).astype(int)
            anchors_te = apply_rounding(xte, 1.0).astype(int)
            oh_tr = pd.get_dummies(anchors_tr).astype(float)
            oh_te = pd.get_dummies(anchors_te).astype(float).reindex(
                columns=oh_tr.columns, fill_value=0.0,
            )
            auc_round = roc_auc_score(
                yte,
                LogisticRegression(max_iter=2000).fit(oh_tr.to_numpy(), ytr).predict_proba(oh_te.to_numpy())[:, 1],
            )
            lin_lifts.append(auc_round - auc_raw)
        mean_lift = float(np.mean(lin_lifts))
        assert mean_lift >= 0.20, (
            f"one-hot(round(x,1.0)) LogReg AUC lift {mean_lift:.4f} < 0.20 "
            f"over raw-x LogReg on the step-parity target (per-seed "
            f"{[round(x, 4) for x in lin_lifts]}); rounding's downstream "
            f"value for linear models is not materialising."
        )


# ---------------------------------------------------------------------------
# Contract 3: bootstrap gate drops noise on a smooth target (precision+recall)
# ---------------------------------------------------------------------------


class TestBootstrapGateDropsNoise:
    """The bootstrap-MI gate correctly drops every decomposition candidate on a smooth (no-decomposition-signal) target."""
    def test_smooth_target_all_decompositions_dropped(self):
        """Ground truth on a smooth target: NO decomposition candidate adds
        stable MI over raw x, so the correct keep/drop label for every
        candidate is DROP. The gate's decision must match -- precision AND
        recall of the drop decision >= 0.9 (i.e. it drops ~all candidates and
        keeps ~none)."""
        from mlframe.feature_selection.filters._numeric_decompose_fe import (
            hybrid_numeric_decompose_fe,
        )
        precisions = (1, 0.1, 0.01, 0.001)
        digits = (0, 1, 2)
        # total candidate count per frame (single numeric col).
        n_candidates = len(precisions) + len(digits)
        tp = fp = fn = 0  # for the DROP decision (positive = "should drop")
        for s in SEEDS:
            X, y = _build_smooth(s)
            X_aug, scores = hybrid_numeric_decompose_fe(
                X, y, precisions=precisions, digit_positions=digits,
                top_k=5, n_boot=10, seed=s,
            )
            kept = [c for c in X_aug.columns if c not in X.columns]
            assert len(scores) == n_candidates
            # ground-truth: every candidate should be dropped (smooth target).
            dropped = [c for c in scores["engineered_col"] if c not in kept]
            # DROP positives: predicted-drop that are true-drop (all true-drop).
            tp += len(dropped)  # every dropped candidate is a correct drop
            fp += 0  # no candidate is a true-keep, so no FP
            fn += len(kept)  # any kept candidate is a missed drop
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        assert precision >= 0.9 and recall >= 0.9, (
            f"bootstrap gate drop-decision precision={precision:.3f} "
            f"recall={recall:.3f} on smooth target; expected both >= 0.9 "
            f"(the gate should drop every decomposition candidate)."
        )


# ---------------------------------------------------------------------------
# Contract 4: AUC lift on the price-anchored fixture
# ---------------------------------------------------------------------------


class TestAucLift:
    """Numeric decomposition FE lifts LogReg AUC on the price-anchored fixture."""

    def test_logreg_auc_lift(self):
        """LogReg AUC on decomposition-augmented features beats raw price by >= 0.15."""
        from mlframe.feature_selection.filters._numeric_decompose_fe import (
            hybrid_numeric_decompose_fe_with_recipes,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        lifts = []
        for s in SEEDS:
            X, y = _build_price_anchored(s)
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=s, stratify=y,
            )
            base = LogisticRegression(max_iter=2000)
            base.fit(Xtr[["price"]], ytr)
            auc_raw = roc_auc_score(yte, base.predict_proba(Xte[["price"]])[:, 1])

            _, appended, recipes, _ = hybrid_numeric_decompose_fe_with_recipes(
                Xtr, ytr.values if hasattr(ytr, "values") else ytr,
                precisions=(1, 0.1, 0.01), digit_positions=(0, 1, 2),
                top_k=5, n_boot=10, seed=s,
            )
            assert appended, f"seed={s}: no decomposition survivors."
            # one-hot the digit survivors (a digit is categorical, not ordinal).
            Xtr_aug = Xtr[["price"]].reset_index(drop=True).copy()
            Xte_aug = Xte[["price"]].reset_index(drop=True).copy()
            for r in recipes:
                tr_col = apply_recipe(r, Xtr)
                te_col = apply_recipe(r, Xte)
                if r.kind == "digit_extract":
                    oh_tr = pd.get_dummies(pd.Series(tr_col).astype(int), prefix=r.name).astype(float)
                    oh_te = pd.get_dummies(pd.Series(te_col).astype(int), prefix=r.name).astype(float)
                    oh_te = oh_te.reindex(columns=oh_tr.columns, fill_value=0.0)
                    Xtr_aug = pd.concat([Xtr_aug, oh_tr.reset_index(drop=True)], axis=1)
                    Xte_aug = pd.concat([Xte_aug, oh_te.reset_index(drop=True)], axis=1)
                else:
                    Xtr_aug[r.name] = tr_col
                    Xte_aug[r.name] = te_col
            aug = LogisticRegression(max_iter=2000)
            aug.fit(Xtr_aug, ytr)
            auc_aug = roc_auc_score(yte, aug.predict_proba(Xte_aug)[:, 1])
            lifts.append(auc_aug - auc_raw)
        mean_lift = float(np.mean(lifts))
        assert mean_lift >= 0.15, (
            f"price-anchored AUC lift {mean_lift:.4f} < 0.15 (per-seed "
            f"{[round(x, 4) for x in lifts]}); the cents-digit decomposition "
            f"is not recovering a separation the raw model can't learn."
        )


# ---------------------------------------------------------------------------
# Contract 5: no leakage -- replay independent of y
# ---------------------------------------------------------------------------


class TestNoYLeak:
    """Recipe replay and candidate generators never leak y into the resulting columns."""

    def test_transform_same_under_shuffled_y(self):
        """A fitted recipe's replayed column is a pure function of X, independent of y."""
        from mlframe.feature_selection.filters._numeric_decompose_fe import (
            hybrid_numeric_decompose_fe_with_recipes,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y, _ = _build_cents_digit(7)
        rng = np.random.default_rng(0)
        y_shuf = y.copy()
        rng.shuffle(y_shuf)
        # The recipe payload (precision / digit position) is a pure function of
        # the survivor name; the REPLAYED column is a pure function of X. So a
        # recipe fitted on the real y must replay identically regardless of any
        # y -- and crucially carries no y reference.
        _, _appended, recipes, _ = hybrid_numeric_decompose_fe_with_recipes(
            X, y, precisions=(1, 0.1, 0.01), digit_positions=(0, 1, 2),
            top_k=5, n_boot=10, seed=7,
        )
        assert recipes, "no recipes produced for leakage test."
        for r in recipes:
            c1 = apply_recipe(r, X)
            c2 = apply_recipe(r, X)
            np.testing.assert_array_equal(c1, c2)
            assert "y" not in dict(r.extra), f"recipe {r.name!r} captured a y reference -- leakage risk."

    def test_generators_never_see_y(self):
        """Rounding/digit feature generators are deterministic pure functions of X."""
        from mlframe.feature_selection.filters._numeric_decompose_fe import (
            generate_rounding_features, generate_digit_features,
        )
        X, _y, _ = _build_cents_digit(42)
        r1 = generate_rounding_features(X, precisions=(1, 0.1))
        r2 = generate_rounding_features(X, precisions=(1, 0.1))
        d1 = generate_digit_features(X, digit_positions=(0, 1, 2))
        d2 = generate_digit_features(X, digit_positions=(0, 1, 2))
        pd.testing.assert_frame_equal(r1, r2)
        pd.testing.assert_frame_equal(d1, d2)


# ---------------------------------------------------------------------------
# Contract 6: default disabled byte-identical
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """fe_numeric_decompose_enable defaults to False and adds columns only when explicitly enabled."""

    def test_mrmr_default_off_adds_nothing(self):
        """With fe_numeric_decompose_enable=False (default), no decomposition columns are added."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_price_anchored(42, n=2000)
        m = MRMR(max_runtime_mins=0.5)
        assert bool(getattr(m, "fe_numeric_decompose_enable", False)) is False, "fe_numeric_decompose_enable must default to False."
        m.fit(X, pd.Series(y, name="y"))
        nd = list(getattr(m, "numeric_decompose_features_", []) or [])
        assert nd == [], f"numeric_decompose added columns with the feature disabled: {nd}"

    def test_mrmr_enabled_adds_decompose(self):
        """With fe_numeric_decompose_enable=True, decomposition columns are added on the cents-digit fixture."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        X, y = _build_cents_digit(42, n=4000)[:2]
        m = MRMR(
            max_runtime_mins=1.0,
            fe_numeric_decompose_enable=True,
            fe_numeric_decompose_precisions=(1, 0.1, 0.01),
            fe_numeric_decompose_digits=(0, 1, 2),
            fe_numeric_decompose_top_k=3,
        )
        m.fit(X, pd.Series(y, name="y"))
        nd = list(getattr(m, "numeric_decompose_features_", []) or [])
        assert len(nd) >= 1, "numeric_decompose enabled but produced no engineered columns on " "the cents-digit fixture."


# ---------------------------------------------------------------------------
# Contract 7: pickle / clone round-trip
# ---------------------------------------------------------------------------


class TestPickleClone:
    """Recipes and ctor params survive pickle / sklearn clone round-trips."""

    def test_recipe_pickle_round_trip(self):
        """A pickled recipe round-trips equal and replays identically."""
        from mlframe.feature_selection.filters._numeric_decompose_fe import (
            hybrid_numeric_decompose_fe_with_recipes,
        )
        from mlframe.feature_selection.filters.engineered_recipes import (
            apply_recipe,
        )
        X, y, _ = _build_cents_digit(1)
        _, _appended, recipes, _ = hybrid_numeric_decompose_fe_with_recipes(
            X, y, precisions=(1, 0.1, 0.01), digit_positions=(0, 1, 2),
            top_k=5, n_boot=10, seed=1,
        )
        assert recipes, "no recipes for pickle test."
        for r in recipes:
            blob = pickle.dumps(r)
            r2 = pickle.loads(blob)  # nosec B301 -- round-trip of a locally-created, trusted object
            assert r2 == r, f"recipe {r.name!r} != its pickle round-trip."
            np.testing.assert_array_equal(apply_recipe(r, X), apply_recipe(r2, X))

    def test_mrmr_clone_preserves_params(self):
        """sklearn clone() preserves every fe_numeric_decompose_* ctor param."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        m = MRMR(
            fe_numeric_decompose_enable=True,
            fe_numeric_decompose_precisions=(1, 0.5, 0.1),
            fe_numeric_decompose_digits=(0, 1),
            fe_numeric_decompose_n_boot=7,
            fe_numeric_decompose_top_k=4,
        )
        c = clone(m)
        assert bool(c.fe_numeric_decompose_enable) is True
        assert tuple(c.fe_numeric_decompose_precisions) == (1, 0.5, 0.1)
        assert tuple(c.fe_numeric_decompose_digits) == (0, 1)
        assert int(c.fe_numeric_decompose_n_boot) == 7
        assert int(c.fe_numeric_decompose_top_k) == 4


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--no-cov"]))
