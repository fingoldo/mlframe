"""The ``one_se_*`` tolerance band must be a standard error (fold-std / sqrt(k)), with the fold-std band kept behind ``*_foldstd``."""
from types import SimpleNamespace

import numpy as np
import pytest

from mlframe.feature_selection.wrappers.rfecv._one_se_band import band_half_width, fold_counts
from mlframe.feature_selection.wrappers.rfecv._stability_select import select_optimal_nfeatures_

NF = [0, 5, 10, 20, 40]
MEAN = [0.50, 0.80, 0.90, 0.895, 0.875]
STD = [0.0, 0.03, 0.03, 0.03, 0.03]


def _fake_rfecv(rule: str, k: int):
    """Fake fitted RFECV with 40 features and k identical folds of per-fold scores."""
    names = [f"f{i}" for i in range(40)]
    return SimpleNamespace(
        n_features_selection_rule=rule, mean_perf_weight=1.0, std_perf_weight=0.0, max_nfeatures=None, feature_names_in_=names,
        n_features_in_=40, _n_samples_fit_=5000, conduct_final_voting=False, verbose=0,
        selected_features_={n: names[:n] for n in NF}, _per_fold_scores={n: [0.9] * k for n in NF},
    )


def _pick(rule: str, k: int) -> int:
    """Return the feature count the rule selects on the fake RFECV."""
    fake = _fake_rfecv(rule, k)
    select_optimal_nfeatures_(fake, checked_nfeatures=NF, cv_mean_perf=MEAN, cv_std_perf=STD, feature_cost=0.0, smooth_perf=0)
    return int(fake.n_features_)


def test_band_half_width_shrinks_by_sqrt_k():
    """Band half width shrinks by sqrt k."""
    std = np.array([0.03, 0.06])
    np.testing.assert_allclose(band_half_width(std, np.array([9, 4])), [0.01, 0.03])
    np.testing.assert_allclose(band_half_width(std, np.array([9, 4]), "foldstd"), std)


def test_fold_counts_ignores_nan_folds_and_defaults_to_one():
    """Fold counts ignores nan folds and defaults to one."""
    fake = SimpleNamespace(_per_fold_scores={5: [0.1, np.nan, 0.3], 10: []})
    np.testing.assert_array_equal(fold_counts(fake, [5, 10, 20]), [2, 1, 1])


def test_one_se_max_uses_standard_error_band_not_fold_std():
    """One se max uses standard error band not fold std."""
    # best mean 0.90 @N=10; N=40 scores 0.875. Fold-std band floor = 0.87 admits N=40; SE band floor with k=9 = 0.89 excludes it.
    assert _pick("one_se_max_foldstd", k=9) == 40
    assert _pick("one_se_max", k=9) == 20


def test_se_band_equals_legacy_band_with_single_fold():
    """Se band equals legacy band with single fold."""
    assert _pick("one_se_max", k=1) == _pick("one_se_max_foldstd", k=1) == 40


def test_auto_resolves_to_se_band_and_reports_rule():
    """Auto resolves to se band and reports rule."""
    fake = _fake_rfecv("auto", k=9)
    select_optimal_nfeatures_(fake, checked_nfeatures=NF, cv_mean_perf=MEAN, cv_std_perf=STD, feature_cost=0.0, smooth_perf=0)
    assert fake.n_features_ == 20 and fake.resolved_n_features_rule_ == "one_se_max"


def test_one_se_min_se_band_is_narrower_so_never_smaller_than_foldstd():
    """One se min se band is narrower so never smaller than foldstd."""
    assert _pick("one_se_min", k=9) >= _pick("one_se_min_foldstd", k=9)


def test_unknown_rule_still_rejected():
    """Unknown rule still rejected."""
    with pytest.raises(ValueError):
        _pick("one_se_medium", k=3)
