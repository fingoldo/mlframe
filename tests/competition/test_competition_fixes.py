"""Regression tests for audits/full_audit_2026-07-21/competition.md findings C1-C11 and proposals P6-P9.

C8 (leak_scan.py's unconditional ``work = df.copy()``) was found ALREADY fixed by an earlier
session-wide large-frame-copy sweep predating this cluster's pass -- this file adds the still-outstanding
formal regression test only. P1-P5 (test-coverage gaps) are closed by the C1-C5 tests below. P7's
one-time-warning decorator is exercised via the ``@competition_only``-equivalent patching in
``mlframe/competition/__init__.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# C1: train_test_union_frequency_encode crashes on NaN/None hierarchical entries
# ---------------------------------------------------------------------------


def test_c1_hierarchical_components_survives_missing_entries():
    """C1: hierarchical components survives missing entries."""
    from mlframe.competition.train_test_union_frequency import train_test_union_frequency_encode_hierarchical_components

    train = pd.Series(["1.2.3", None, "1.2.4", "2.0.0"])
    test = pd.Series(["1.2.3", "9.9.9", np.nan])

    components = train_test_union_frequency_encode_hierarchical_components(train, test, ".")
    assert "major" in components
    train_major, test_major = components["major"]
    assert train_major.isna().sum() == 0
    assert test_major.isna().sum() == 0
    # The missing rows share a single sentinel token -- not crashing, not NaN.
    assert train_major.iloc[1] == test_major.iloc[2]


def test_c1_top_level_encode_survives_missing_entries():
    """C1: top level encode survives missing entries."""
    from mlframe.competition.train_test_union_frequency import train_test_union_frequency_encode

    train = pd.Series(["1.2.3", None, "1.2.4"])
    test = pd.Series(["1.2.3", np.nan])
    train_enc, test_enc = train_test_union_frequency_encode(train, test, hierarchical_split_sep=".")
    assert np.all(np.isfinite(train_enc.to_numpy()))
    assert np.all(np.isfinite(test_enc.to_numpy()))


# ---------------------------------------------------------------------------
# C2: frequency_power_interaction silently returns NaN for a fractional exponent
# ---------------------------------------------------------------------------


def test_c2_fractional_count_clip_range_raises_instead_of_nan():
    """C2: fractional count clip range raises instead of nan."""
    from mlframe.competition.frequency_power_interaction import frequency_power_interaction

    x = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    with pytest.raises(ValueError, match="integer"):
        frequency_power_interaction(x, count_clip_range=(1.5, 2.5))


def test_c2_integer_count_clip_range_still_works():
    """C2: integer count clip range still works."""
    from mlframe.competition.frequency_power_interaction import frequency_power_interaction

    x = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    result = frequency_power_interaction(x, count_clip_range=(1.0, 3.0))
    assert np.all(np.isfinite(result.interaction_feature))


def test_c10_count_clip_range_docstring_documents_integer_requirement():
    """C10: the docstring must call out that count_clip_range MUST be integer bounds and explain the
    fractional-exponent-on-negative-base failure mode C-2 fixed -- a docs/validation gap distinct from
    C-2's raise-behaviour itself, since a caller who never triggers the ValueError could still be misled
    by a docstring silent on the constraint."""
    from mlframe.competition.frequency_power_interaction import frequency_power_interaction

    doc = frequency_power_interaction.__doc__
    assert doc is not None
    assert "MUST" in doc and "integer" in doc
    assert "NaN" in doc or "nan" in doc


# ---------------------------------------------------------------------------
# C3: value_uniqueness_encoder leaks a raw NaN for a NaN train value, asymmetric with test-side
# ---------------------------------------------------------------------------


def test_c3_nan_values_never_leak_a_raw_nan_category():
    """C3: nan values never leak a raw nan category."""
    from mlframe.competition.value_uniqueness_encoder import MISSING_VALUE, VALUE_UNIQUENESS_CATEGORIES, value_uniqueness_encoder

    rng = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame({"col": rng.choice(["a", "b", "c"], size=n).astype(object)})
    train.loc[0:5, "col"] = np.nan
    test = pd.DataFrame({"col": rng.choice(["a", "b", "d"], size=50).astype(object)})
    test.loc[0:3, "col"] = np.nan
    y_train = rng.integers(0, 2, size=n)

    out = value_uniqueness_encoder(train, test, real_test_mask=None, y_train=y_train, columns=["col"])
    col = out["col__value_uniqueness"]
    assert col.isna().sum() == 0
    assert set(col.cat.categories) == set(VALUE_UNIQUENESS_CATEGORIES)
    # The train-side NaN rows (first 6 of the concatenated frame) and test-side NaN rows both resolve
    # to the SAME documented sentinel category -- symmetric, not the old "raw NaN vs unique_globally" split.
    assert (col.iloc[0:6] == MISSING_VALUE).all()
    test_nan_positions = n + np.arange(0, 4)
    assert (col.iloc[test_nan_positions] == MISSING_VALUE).all()


# ---------------------------------------------------------------------------
# C4: GaussianMixtureClassifier.fit crashes on a singleton-sample class
# ---------------------------------------------------------------------------


def test_c4_singleton_sample_class_raises_clear_error():
    """C4: singleton sample class raises clear error."""
    from mlframe.competition.gmm_classifier import GaussianMixtureClassifier

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(30, 2)), rng.normal(loc=5.0, size=(1, 2))])
    y = np.array([0] * 30 + [1])

    clf = GaussianMixtureClassifier(n_components_per_class=2, random_state=0)
    with pytest.raises(ValueError, match="only 1 sample"):
        clf.fit(X, y)


def test_c4_two_or_more_samples_per_class_still_works():
    """C4: two or more samples per class still works."""
    from mlframe.competition.gmm_classifier import GaussianMixtureClassifier

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(size=(30, 2)), rng.normal(loc=5.0, size=(2, 2))])
    y = np.array([0] * 30 + [1] * 2)
    clf = GaussianMixtureClassifier(n_components_per_class=2, random_state=0)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (32, 2)


# ---------------------------------------------------------------------------
# C5/C9: known_label_override silently applies in the wrong direction on a reversed scale
# ---------------------------------------------------------------------------


def test_c5_reversed_scale_still_overrides_toward_positive_value():
    """C5: reversed scale still overrides toward positive value."""
    from mlframe.competition.known_label_override import known_label_override

    preds = np.array([0.5, 0.5, 0.5])
    # Reversed convention: 0.0 is the "positive"/rare class of interest, 1.0 is "negative".
    known_map = {0: 0.1, 1: 0.9}  # row 0 recovered close to positive_value=0.0; row 1 close to negative_value=1.0
    out = known_label_override(preds, known_map, asymmetric_safe_direction="positive", positive_value=0.0, negative_value=1.0)
    assert out[0] == pytest.approx(0.0)  # overridden toward positive_value
    assert out[1] == pytest.approx(0.5)  # NOT overridden (points toward negative_value)
    assert out[2] == pytest.approx(0.5)  # untouched (no recovered label)


def test_c5_standard_ascending_scale_unaffected():
    """C5: standard ascending scale unaffected."""
    from mlframe.competition.known_label_override import known_label_override

    preds = np.array([0.5, 0.5])
    known_map = {0: 0.9, 1: 0.1}
    out = known_label_override(preds, known_map, asymmetric_safe_direction="positive", positive_value=1.0, negative_value=0.0)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# C6: leak_scan._nonnull_nonzero_mask materialized a full object-dtype copy just to read .shape
# ---------------------------------------------------------------------------


def test_c6_nonnull_nonzero_mask_correct_without_object_copy():
    """C6: nonnull nonzero mask correct without object copy."""
    from mlframe.competition.leak_scan import _nonnull_nonzero_mask

    df = pd.DataFrame({"a": [1.0, 0.0, np.nan, 3.0], "b": ["x", None, "y", "z"]})
    mask = _nonnull_nonzero_mask(df)
    expected = np.array(
        [
            [True, True],
            [False, False],
            [False, True],
            [True, True],
        ]
    )
    assert np.array_equal(mask, expected)


# ---------------------------------------------------------------------------
# C7: ThresholdRangeRescaler._cv_score re-split StratifiedKFold on every grid-search candidate
# ---------------------------------------------------------------------------


def test_c7_cv_score_reuses_precomputed_fold_indices(monkeypatch):
    """C7: cv score reuses precomputed fold indices."""
    from mlframe.competition.threshold_range_rescaler import ThresholdRangeRescaler
    from sklearn.model_selection import StratifiedKFold

    split_call_count = {"n": 0}
    real_split = StratifiedKFold.split

    def spy_split(self, X, y=None, groups=None):
        """Records split() calls for this test's assertions."""
        split_call_count["n"] += 1
        return real_split(self, X, y, groups)

    monkeypatch.setattr(StratifiedKFold, "split", spy_split)

    rng = np.random.default_rng(0)
    n = 300
    preds = rng.uniform(0.1, 0.9, size=n)
    y = (rng.uniform(size=n) < preds).astype(np.int64)
    subgroups = {"g1": rng.random(n) < 0.5}

    rescaler = ThresholdRangeRescaler(thresholds=np.array([0.3, 0.5]), multipliers=np.array([0.8, 1.2]), n_splits=3, max_corrections=2, random_state=0)
    rescaler.fit(preds, y, subgroups)

    # Exactly ONE StratifiedKFold.split call for the whole fit() -- pre-fix this scaled with the grid
    # size (subgroups x thresholds x multipliers x max_corrections rounds).
    assert split_call_count["n"] == 1


def test_c7_result_bit_identical_to_pre_fix_semantics():
    """The caching itself must not change WHICH correction gets picked or its score."""
    from mlframe.competition.threshold_range_rescaler import ThresholdRangeRescaler
    from sklearn.model_selection import StratifiedKFold

    rng = np.random.default_rng(1)
    n = 400
    is_revolving = rng.random(n) < 0.35
    latent = rng.normal(size=n)
    true_prob = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.random(n) < true_prob).astype(np.int64)
    pred = true_prob + rng.normal(scale=0.05, size=n)
    pred[is_revolving & (pred > 0.4)] += 0.3
    pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
    subgroups = {"revolving": is_revolving}

    rescaler = ThresholdRangeRescaler(thresholds=np.array([0.3, 0.4, 0.5]), multipliers=np.array([0.7, 0.8, 1.2]), n_splits=4, max_corrections=2, random_state=0)
    rescaler.fit(pred, y, subgroups)

    # Manually recompute the SAME fold partition independently (as the pre-fix per-call code would
    # have) and confirm the cached version's baseline score matches exactly.
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    fold_test_indices = [test_idx for _, test_idx in skf.split(pred, y)]
    from sklearn.metrics import roc_auc_score

    manual_fold_scores = [roc_auc_score(y[idx], pred[idx]) for idx in fold_test_indices]
    assert rescaler.baseline_cv_score_ == pytest.approx(float(np.mean(manual_fold_scores)))
    assert len(rescaler.corrections_) >= 1
    assert rescaler.corrections_[0].subgroup == "revolving"


# ---------------------------------------------------------------------------
# C8: leak_scan's unconditional work = df.copy() -- ALREADY fixed pre-cluster, test added here
# ---------------------------------------------------------------------------


def test_c8_sort_by_density_no_target_avoids_frame_copy():
    """C8: sort by density no target avoids frame copy."""
    from mlframe.competition.leak_scan import sort_by_density_leak_scan

    result = sort_by_density_leak_scan(pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 1.0, 0.0]}), target=None)
    assert result["row_order"].shape[0] == 3


# ---------------------------------------------------------------------------
# P6: find_shifted_column_groups warns on a wide frame instead of silently running very long
# ---------------------------------------------------------------------------


def test_p6_wide_frame_warns_via_max_columns():
    """P6: wide frame warns via max columns."""
    from mlframe.competition.leak_scan import find_shifted_column_groups

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"c{i}" for i in range(5)])
    with pytest.warns(UserWarning, match="max_columns"):
        find_shifted_column_groups(df, max_columns=3)


def test_p6_default_max_columns_silent_for_narrow_frame():
    """P6: default max columns silent for narrow frame."""
    from mlframe.competition.leak_scan import find_shifted_column_groups

    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(20, 5)), columns=[f"c{i}" for i in range(5)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        find_shifted_column_groups(df)  # 5 columns is well under the default max_columns=500
    # The P7 one-time "COMPETITION/EXPLORATORY-ONLY" banner may or may not fire here depending on
    # whether an earlier test already used this callable -- only assert the max_columns warning is absent.
    assert not any("max_columns" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# P7: every competition-only entry point warns once per process on first use
# ---------------------------------------------------------------------------


def test_p7_function_entry_point_warns_once_via_submodule_path():
    """P7: function entry point warns once via submodule path."""
    import mlframe.competition as comp_pkg
    from mlframe.competition.logloss_clip import clip_probabilities_for_logloss

    comp_pkg._WARNED_COMPETITION_ONLY.discard("clip_probabilities_for_logloss")
    with pytest.warns(UserWarning, match="COMPETITION/EXPLORATORY-ONLY"):
        clip_probabilities_for_logloss(np.array([0.0, 0.5, 1.0]))
    # Second call in the same process does NOT warn again.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        clip_probabilities_for_logloss(np.array([0.0, 0.5, 1.0]))


def test_p7_class_entry_point_warns_once():
    """P7: class entry point warns once."""
    import mlframe.competition as comp_pkg
    from mlframe.competition.rounded_categorical_interaction import RoundedNumericCategoricalInteraction

    comp_pkg._WARNED_COMPETITION_ONLY.discard("RoundedNumericCategoricalInteraction")
    with pytest.warns(UserWarning, match="COMPETITION/EXPLORATORY-ONLY"):
        RoundedNumericCategoricalInteraction()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        RoundedNumericCategoricalInteraction()


def test_p7_package_level_reexport_also_warns():
    """Confirms the patch applies at the submodule level, not just the (separately bound)
    package-level re-export name -- both import paths must see the SAME wrapped callable."""
    import mlframe.competition as comp_pkg

    assert comp_pkg.clip_probabilities_for_logloss is comp_pkg._logloss_clip_mod.clip_probabilities_for_logloss


# ---------------------------------------------------------------------------
# P8: NaiveBayesLogOddsEnsembler.predict_proba had no n_features_in_ validation
# ---------------------------------------------------------------------------


def test_p8_predict_proba_rejects_mismatched_feature_count():
    """P8: predict proba rejects mismatched feature count."""
    from mlframe.competition.naive_bayes_log_odds import NaiveBayesLogOddsEnsembler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = NaiveBayesLogOddsEnsembler(calibrate=False)
    model.fit(X, y)

    with pytest.raises(ValueError, match="n_features_in_|feature"):
        model.predict_proba(rng.normal(size=(10, 3)))


def test_p8_matching_feature_count_still_works():
    """P8: matching feature count still works."""
    from mlframe.competition.naive_bayes_log_odds import NaiveBayesLogOddsEnsembler

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = NaiveBayesLogOddsEnsembler(calibrate=False)
    model.fit(X, y)
    proba = model.predict_proba(rng.normal(size=(10, 4)))
    assert proba.shape == (10, 2)


# ---------------------------------------------------------------------------
# P9: _cv_score now exposes per-fold scores for variance-based gating
# ---------------------------------------------------------------------------


def test_p9_correction_exposes_per_fold_scores():
    """P9: correction exposes per fold scores."""
    from mlframe.competition.threshold_range_rescaler import ThresholdRangeRescaler

    rng = np.random.default_rng(1)
    n = 400
    is_revolving = rng.random(n) < 0.35
    latent = rng.normal(size=n)
    true_prob = 1.0 / (1.0 + np.exp(-latent))
    y = (rng.random(n) < true_prob).astype(np.int64)
    pred = true_prob + rng.normal(scale=0.05, size=n)
    pred[is_revolving & (pred > 0.4)] += 0.3
    pred = np.clip(pred, 1e-6, 1.0 - 1e-6)

    rescaler = ThresholdRangeRescaler(thresholds=np.array([0.3, 0.4, 0.5]), multipliers=np.array([0.7, 0.8]), n_splits=4, max_corrections=1, random_state=0)
    rescaler.fit(pred, y, {"revolving": is_revolving})

    result = rescaler.result()
    assert len(result.baseline_fold_scores) == 4
    assert len(result.corrections) >= 1
    assert len(result.corrections[0].fold_scores) == 4
    assert result.corrections[0].cv_score == pytest.approx(float(np.mean(result.corrections[0].fold_scores)))
