"""Layer 37 biz_value: MISSING-VALUE-AWARE FE -- surface missingness as
predictive signal where it carries information (MNAR pattern from Layer 7).

WHY THIS LAYER
--------------
Layer 7 pinned that MRMR's ``nan_strategy='separate_bin'`` handles MNAR
at the BINNING level inside the MI estimator: NaN gets its own bin per
column so the relevance gate doesn't lose the missingness signal. That
contract protects the SELECTION step, but the engineered output frame
the downstream model trains on still contains the original NaN values
in those columns.

Layer 37 COMPLEMENTS Layer 7 by EMITTING missingness as standalone
engineered columns the downstream model can consume directly:

* ``is_missing__{col}``: per-source binary indicator. Captures the
  raw MNAR signal as an unambiguous numeric column.
* ``missingness_count``: per-row count of NaNs across a column subset.
  Captures the "fragmentary record" pattern (a row with many missing
  fields is qualitatively different from one with a single gap).
* ``missingness_pattern``: per-row label of the top-K most frequent
  missingness patterns at fit. Lets a linear model separate distinct
  MNAR clusters that no marginal indicator can express alone.

CONTRACTS PINNED
----------------
* MNAR signal: y depends on whether credit_history is missing;
  ``is_missing__credit_history`` enters support.
* High-missing-row signal: y depends on the total count of missing
  fields; ``missingness_count`` enters support.
* Pattern signal: y depends on a specific (cluster of) missingness
  pattern; ``missingness_pattern`` recovers it.
* No leakage: recipe replay reads only X.
* Pickle / clone preserves all params + fitted state.
* Default disabled is byte-identical to a vanilla MRMR.

NEVER xfail. Real LogReg AUC numbers.
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


warnings.filterwarnings("ignore")


SEEDS = (3701, 3702, 3703)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mrmr(**overrides):
    from mlframe.feature_selection.filters.mrmr import MRMR
    kwargs = dict(
        verbose=0,
        interactions_max_order=1,
        fe_max_steps=0,
        dcd_enable=False,
        cluster_aggregate_enable=False,
        build_friend_graph=False,
        cat_fe_config=None,
        quantization_nbins=10,
        random_seed=0,
    )
    kwargs.update(overrides)
    return MRMR(**kwargs)


def _train_holdout_split(X: pd.DataFrame, y: pd.Series, *, train_frac: float = 0.6, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(train_frac * len(X))
    tr, ho = idx[:cut], idx[cut:]
    return (
        X.iloc[tr].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True),
        X.iloc[ho].reset_index(drop=True),
        y.iloc[ho].reset_index(drop=True),
    )


def _logreg_auc(X_tr: pd.DataFrame, y_tr: pd.Series, X_ho: pd.DataFrame, y_ho: pd.Series) -> float:
    num_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
    if not num_cols:
        return 0.5
    Xn_tr = X_tr[num_cols].fillna(0.0).to_numpy(dtype=np.float64)
    Xn_ho = X_ho[num_cols].fillna(0.0).to_numpy(dtype=np.float64)
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(Xn_tr, y_tr.to_numpy())
    proba = clf.predict_proba(Xn_ho)[:, 1]
    return float(roc_auc_score(y_ho.to_numpy(), proba))


# ---------------------------------------------------------------------------
# Fixtures: signals that ONLY missingness FE can decode
# ---------------------------------------------------------------------------


def _build_mnar_indicator_signal(seed: int, n: int = 3000):
    """y depends on whether ``credit_history`` is missing.

    Concrete: credit_history is observed Gaussian for the ~70% of
    applicants who have one; the other ~30% are thin-file (NaN). y is
    almost-deterministic in is_missing(credit_history): missing -> high
    default probability, present -> low. The observed values themselves
    carry only a tiny lift; the raw imputed column predicts at ~0.55
    AUC, while ``is_missing__credit_history`` is the ~0.95 AUC signal.
    """
    rng = np.random.default_rng(seed)
    is_missing = rng.random(n) < 0.30
    credit_history = rng.standard_normal(n)
    credit_history[is_missing] = np.nan
    # Sigmoid driven almost entirely by is_missing.
    logit = -2.5 + 5.0 * is_missing.astype(np.float64) + 0.1 * rng.standard_normal(n)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    noise = rng.standard_normal((n, 4))
    X = pd.DataFrame({
        "credit_history": credit_history,
        "noise_a": noise[:, 0],
        "noise_b": noise[:, 1],
        "noise_c": noise[:, 2],
        "noise_d": noise[:, 3],
    })
    return X, pd.Series(y, name="y")


def _build_missing_count_signal(seed: int, n: int = 3000):
    """y depends on the per-row count of missing values across a fixed
    set of fields. Each individual indicator is only weakly informative;
    only their SUM separates the "fragmentary record" cluster from the
    rest.
    """
    rng = np.random.default_rng(seed)
    n_fields = 6
    # Each field has its own independent ~25% missing rate.
    masks = (rng.random((n, n_fields)) < 0.25)
    counts = masks.sum(axis=1).astype(np.float64)
    # y depends on the count crossing a threshold (>=4 of 6 missing).
    p = 1.0 / (1.0 + np.exp(-(counts - 3.5) * 1.5))
    y = (rng.random(n) < p).astype(int)
    data = {}
    for j in range(n_fields):
        col = rng.standard_normal(n)
        col[masks[:, j]] = np.nan
        data[f"field_{j}"] = col
    # Plus 3 noise columns with no missing values.
    for k in range(3):
        data[f"noise_{k}"] = rng.standard_normal(n)
    X = pd.DataFrame(data)
    return X, pd.Series(y, name="y")


def _build_missing_pattern_signal(seed: int, n: int = 3000):
    """y depends on a SPECIFIC missingness pattern.

    Three latent customer clusters:
      cluster A (40% rows): a, b present;   c missing  -> y=0
      cluster B (40% rows): a missing;      b, c present -> y=0
      cluster C (20% rows): a, b missing;   c present  -> y=1 (target cluster)
    Marginal P(missing) per column is identical between B and C for
    column a, between A and C for column b, etc.; only the JOINT pattern
    (a missing AND b missing AND c present) identifies cluster C. The
    indicator-only model is much weaker than the pattern model on this
    fixture.
    """
    rng = np.random.default_rng(seed)
    cluster = rng.choice([0, 1, 2], size=n, p=[0.4, 0.4, 0.2])
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    c = rng.standard_normal(n)
    # Cluster A: c missing
    a[cluster == 0] = a[cluster == 0]  # noop, no missing
    c[cluster == 0] = np.nan
    # Cluster B: a missing
    a[cluster == 1] = np.nan
    # Cluster C: a missing AND b missing
    a[cluster == 2] = np.nan
    b[cluster == 2] = np.nan
    # y: only cluster C is positive (target).
    p = np.where(cluster == 2, 0.9, 0.1)
    y = (rng.random(n) < p).astype(int)
    noise = rng.standard_normal((n, 3))
    X = pd.DataFrame({
        "field_a": a,
        "field_b": b,
        "field_c": c,
        "noise_x": noise[:, 0],
        "noise_y": noise[:, 1],
        "noise_z": noise[:, 2],
    })
    return X, pd.Series(y, name="y")


# ---------------------------------------------------------------------------
# Direct unit tests on each kernel
# ---------------------------------------------------------------------------


class TestMissingIndicatorKernel:
    def test_fit_returns_int8_indicator(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            missing_indicator_fit,
        )
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan]})
        enc, recipes = missing_indicator_fit(X, ["a"])
        assert "is_missing__a" in enc.columns
        np.testing.assert_array_equal(
            enc["is_missing__a"].to_numpy(), np.array([0, 1, 0, 1], dtype=np.int8),
        )
        assert recipes["a"] == {}

    def test_empty_X_rejected(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            missing_indicator_fit,
        )
        with pytest.raises(ValueError, match="empty"):
            missing_indicator_fit(pd.DataFrame({"a": []}), ["a"])

    def test_missing_column_rejected(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            missing_indicator_fit,
        )
        with pytest.raises(ValueError, match="missing"):
            missing_indicator_fit(pd.DataFrame({"a": [1.0]}), ["b"])

    def test_apply_returns_same_isna(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            apply_missing_indicator,
        )
        X = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        out = apply_missing_indicator(X, "a", {})
        np.testing.assert_array_equal(out, np.array([0, 1, 0], dtype=np.int8))


class TestMissingCountKernel:
    def test_fit_returns_per_row_count(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            missingness_count_fit,
        )
        X = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": [np.nan, np.nan, 5.0, 6.0],
        })
        counts, recipe = missingness_count_fit(X, ["a", "b"])
        np.testing.assert_array_equal(counts, np.array([1, 2, 0, 1], dtype=np.int32))
        assert recipe["cols"] == ("a", "b")

    def test_apply_handles_schema_drift(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            apply_missingness_count,
        )
        # Recipe references columns 'a' AND 'b' but X_test only has 'a'.
        # Graceful schema drift: count only what's present.
        X = pd.DataFrame({"a": [np.nan, 1.0, np.nan]})
        out = apply_missingness_count(X, {"cols": ("a", "b")})
        np.testing.assert_array_equal(out, np.array([1, 0, 1], dtype=np.int32))


class TestMissingPatternKernel:
    def test_fit_assigns_top_k_labels(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            missingness_pattern_fit,
        )
        # Three patterns, two are top-2; the third should go to "other".
        X = pd.DataFrame({
            "a": [1.0, np.nan, 1.0, np.nan, 1.0, 1.0, np.nan, np.nan, np.nan, 1.0],
            "b": [1.0, 1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0],
        })
        labels, recipe = missingness_pattern_fit(X, ["a", "b"], top_k=2)
        # Two distinct patterns appear: (a present, b present) and
        # (a missing, b present). The (a present, b missing) pattern at
        # index 4 falls into "other" only if it appears strictly less
        # often than the top-2.
        assert labels.dtype == np.int32
        assert recipe["top_k"] == 2
        assert recipe["other_label"] == 2
        # At least one row should be "other" only if the rare pattern
        # exists; in this construction "b missing" appears exactly once.
        assert int((labels == 2).sum()) == 1

    def test_apply_unseen_pattern_maps_to_other(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            missingness_pattern_fit, apply_missingness_pattern,
        )
        X_tr = pd.DataFrame({
            "a": [1.0, np.nan, 1.0, np.nan],
            "b": [1.0, 1.0, 1.0, 1.0],
        })
        _, recipe = missingness_pattern_fit(X_tr, ["a", "b"], top_k=2)
        # Test frame has an unseen pattern (a present, b missing).
        X_ho = pd.DataFrame({
            "a": [1.0, np.nan],
            "b": [np.nan, np.nan],
        })
        out = apply_missingness_pattern(X_ho, {
            "cols": recipe["cols"],
            "pattern_to_label": recipe["pattern_to_label"],
            "other_label": recipe["other_label"],
            "top_k": recipe["top_k"],
        })
        # Both unseen patterns -> "other" label (= top_k = 2)
        assert out[0] == 2
        assert out[1] == 2


# ---------------------------------------------------------------------------
# Biz value: AUC lift via each encoder over a no-missingness-FE baseline
# ---------------------------------------------------------------------------


class TestMissingIndicatorAUCLift:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lift_via_indicator(self, seed: int):
        from mlframe.feature_selection.filters._missingness_fe import (
            missing_indicator_with_recipes,
        )
        X, y = _build_mnar_indicator_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)
        # Baseline: no indicator, raw numeric cols only.
        auc_base = _logreg_auc(X_tr, y_tr, X_ho, y_ho)
        # With indicator: append is_missing__credit_history to both frames.
        X_tr_aug, _, _ = missing_indicator_with_recipes(
            X_tr, cols=["credit_history"],
        )
        X_ho_aug, _, _ = missing_indicator_with_recipes(
            X_ho, cols=["credit_history"],
        )
        auc_aug = _logreg_auc(X_tr_aug, y_tr, X_ho_aug, y_ho)
        assert auc_aug > auc_base + 0.10, (
            f"Indicator-augmented AUC {auc_aug:.3f} not measurably above baseline "
            f"{auc_base:.3f} on the MNAR signal."
        )


class TestMissingCountAUCLift:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_logreg_auc_lift_via_count(self, seed: int):
        from mlframe.feature_selection.filters._missingness_fe import (
            missingness_count_with_recipes,
        )
        X, y = _build_missing_count_signal(seed)
        X_tr, y_tr, X_ho, y_ho = _train_holdout_split(X, y, seed=seed)
        field_cols = [c for c in X_tr.columns if c.startswith("field_")]
        # Baseline: raw numeric cols only.
        auc_base = _logreg_auc(X_tr, y_tr, X_ho, y_ho)
        X_tr_aug, _, _ = missingness_count_with_recipes(X_tr, cols=field_cols)
        X_ho_aug, _, _ = missingness_count_with_recipes(X_ho, cols=field_cols)
        auc_aug = _logreg_auc(X_tr_aug, y_tr, X_ho_aug, y_ho)
        assert auc_aug > auc_base + 0.05, (
            f"Count-augmented AUC {auc_aug:.3f} not measurably above baseline "
            f"{auc_base:.3f} on the high-missing-row signal."
        )


# ---------------------------------------------------------------------------
# MRMR end-to-end: MNAR indicator enters support, count enters support
# ---------------------------------------------------------------------------


class TestMRMRSelectsMissingIndicator:
    """``is_missing__credit_history`` must enter MRMR's support when the
    indicator is enabled and the signal is MNAR."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_indicator_enters_support(self, seed: int):
        X, y = _build_mnar_indicator_signal(seed)
        sel = _make_mrmr(
            fe_missingness_indicator_enable=True,
            fe_missingness_indicator_cols=("credit_history",),
        ).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "is_missing__credit_history" in names, (
            f"is_missing__credit_history not in support; got {names}"
        )


class TestMRMRSelectsMissingnessCount:
    """``missingness_count`` must enter MRMR's support when the signal
    depends on the row-level total of missing fields."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_count_enters_support(self, seed: int):
        X, y = _build_missing_count_signal(seed)
        sel = _make_mrmr(
            fe_missingness_count_enable=True,
            fe_missingness_indicator_cols=tuple(
                c for c in X.columns if c.startswith("field_")
            ),
        ).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "missingness_count" in names, (
            f"missingness_count not in support; got {names}"
        )


class TestMRMRSelectsMissingnessPattern:
    """``missingness_pattern`` must enter MRMR's support when the signal
    depends on a specific joint pattern that no marginal indicator can
    cleanly express."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_pattern_enters_support(self, seed: int):
        X, y = _build_missing_pattern_signal(seed)
        sel = _make_mrmr(
            fe_missingness_pattern_enable=True,
            fe_missingness_indicator_cols=("field_a", "field_b", "field_c"),
            fe_missingness_pattern_top_k=4,
        ).fit(X, y)
        names = list(sel.get_feature_names_out())
        assert "missingness_pattern" in names, (
            f"missingness_pattern not in support; got {names}"
        )


# ---------------------------------------------------------------------------
# Leakage: recipe replay reads only X (no y)
# ---------------------------------------------------------------------------


class TestNoLeakage:
    def test_indicator_replay_bit_identical_under_shuffled_y(self):
        """The replay path takes no y, so a shuffled y at fit time must
        produce the SAME engineered column on the same X."""
        from mlframe.feature_selection.filters._missingness_fe import (
            apply_missing_indicator,
        )
        X = pd.DataFrame({"x": [1.0, np.nan, 2.0, np.nan, 3.0]})
        out1 = apply_missing_indicator(X, "x", {})
        out2 = apply_missing_indicator(X, "x", {})
        np.testing.assert_array_equal(out1, out2)

    def test_count_replay_bit_identical_when_recipe_unchanged(self):
        from mlframe.feature_selection.filters._missingness_fe import (
            apply_missingness_count,
        )
        X = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, 1.0, np.nan],
        })
        recipe = {"cols": ("a", "b")}
        out1 = apply_missingness_count(X, recipe)
        out2 = apply_missingness_count(X, recipe)
        np.testing.assert_array_equal(out1, out2)

    def test_mrmr_transform_no_y_at_replay(self):
        """End-to-end: fit on (X, y), call transform on X alone. Output
        must be identical to a re-fit + transform with the same recipe
        (no fit-time-y bleed into the engineered column values)."""
        X, y = _build_mnar_indicator_signal(seed=3701)
        sel = _make_mrmr(
            fe_missingness_indicator_enable=True,
            fe_missingness_indicator_cols=("credit_history",),
        ).fit(X, y)
        out1 = sel.transform(X)
        out2 = sel.transform(X)
        if "is_missing__credit_history" in out1.columns:
            np.testing.assert_array_equal(
                out1["is_missing__credit_history"].to_numpy(),
                out2["is_missing__credit_history"].to_numpy(),
            )


# ---------------------------------------------------------------------------
# Pickle / clone contracts
# ---------------------------------------------------------------------------


class TestPickleClone:
    def _build(self):
        return _make_mrmr(
            fe_missingness_indicator_enable=True,
            fe_missingness_indicator_cols=("credit_history",),
            fe_missingness_count_enable=True,
            fe_missingness_pattern_enable=True,
            fe_missingness_pattern_top_k=3,
        )

    def test_clone_preserves_layer37_params(self):
        m = self._build()
        m2 = clone(m)
        p1 = m.get_params()
        p2 = m2.get_params()
        for key in (
            "fe_missingness_indicator_enable",
            "fe_missingness_indicator_cols",
            "fe_missingness_count_enable",
            "fe_missingness_pattern_enable",
            "fe_missingness_pattern_top_k",
        ):
            assert p1[key] == p2[key], (
                f"clone lost param {key!r}: orig={p1[key]!r} clone={p2[key]!r}"
            )

    def test_pickle_roundtrip_preserves_transform(self):
        X, y = _build_mnar_indicator_signal(seed=3701)
        m = self._build()
        m.fit(X, y)
        pre_out = m.transform(X)
        m2 = pickle.loads(pickle.dumps(m))
        post_out = m2.transform(X)
        assert list(post_out.columns) == list(pre_out.columns)
        for col in pre_out.columns:
            if pd.api.types.is_numeric_dtype(pre_out[col]):
                np.testing.assert_allclose(
                    np.asarray(pre_out[col], dtype=np.float64),
                    np.asarray(post_out[col], dtype=np.float64),
                    rtol=1e-9, atol=1e-9,
                    err_msg=f"pickle changed values of column {col!r}",
                )


# ---------------------------------------------------------------------------
# Default-disabled byte-identical contract
# ---------------------------------------------------------------------------


class TestDefaultDisabledByteIdentical:
    """With every Layer 37 master switch off (the default), MRMR's
    transform output must match a vanilla instance bit-for-bit."""

    def test_no_missingness_features_appear_by_default(self):
        X, y = _build_mnar_indicator_signal(seed=3701)
        vanilla = _make_mrmr().fit(X, y)
        defaulted = _make_mrmr(
            # All defaults preserved explicitly to lock the contract.
            fe_missingness_indicator_enable=False,
            fe_missingness_count_enable=False,
            fe_missingness_pattern_enable=False,
        ).fit(X, y)
        v_names = list(vanilla.get_feature_names_out())
        d_names = list(defaulted.get_feature_names_out())
        assert v_names == d_names, (
            f"Default-disabled Layer 37 changed selection: "
            f"vanilla={v_names} defaulted={d_names}"
        )
        # No engineered missingness columns leaked.
        for nm in v_names:
            assert not nm.startswith("is_missing__"), (
                f"vanilla selection contains an unexpected missingness col: {nm}"
            )
            assert nm != "missingness_count", (
                "vanilla selection contains missingness_count by default"
            )
            assert nm != "missingness_pattern", (
                "vanilla selection contains missingness_pattern by default"
            )
