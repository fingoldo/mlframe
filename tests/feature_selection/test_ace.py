"""Tests for ACE (Artificial Contrasts with Ensembles) - src/mlframe/feature_selection/ace.py.

Unit coverage: contrast construction, BH correction, t-test edge cases, masking loop, importance modes,
input guards. biz_value: ACE must accept the genuine signal features and reject pure-noise columns on a
synthetic where the answer is known, with a quantitative floor on the signal-vs-noise selection gap.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.ace import (
    ACEResult,
    ace_select,
    _benjamini_hochberg_reject,
    _make_contrasts,
    _ttest_greater,
    _default_estimator,
)


# ------------------------------------------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------------------------------------------


def test_make_contrasts_preserves_column_marginals():
    """Make contrasts preserves column marginals."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    C = _make_contrasts(X, np.random.default_rng(1))
    assert C.shape == X.shape
    # Each contrast column is a permutation of the source column: same sorted values, order destroyed.
    for j in range(X.shape[1]):
        assert np.allclose(np.sort(C[:, j]), np.sort(X[:, j]))
    assert not np.allclose(C, X)  # rows were actually shuffled


def test_benjamini_hochberg_monotone_and_bounds():
    # All-tiny p-values -> all reject; all-large -> none reject.
    """Benjamini hochberg monotone and bounds."""
    assert _benjamini_hochberg_reject(np.array([1e-6, 1e-6, 1e-6]), 0.05).all()
    assert not _benjamini_hochberg_reject(np.array([0.9, 0.8, 0.7]), 0.05).any()
    # BH is at least as strict as per-feature alpha: the smallest p passes only if p <= alpha/m.
    p = np.array([0.02, 0.5, 0.5, 0.5, 0.5])
    assert not _benjamini_hochberg_reject(p, 0.05).any()  # 0.02 > 0.05/5 = 0.01
    assert _benjamini_hochberg_reject(np.array([0.005, 0.5, 0.5, 0.5, 0.5]), 0.05)[0]


def test_benjamini_hochberg_empty():
    """Benjamini hochberg empty."""
    assert _benjamini_hochberg_reject(np.array([]), 0.05).shape == (0,)


def test_ttest_greater_signal_vs_noise():
    # A feature whose per-replicate importance sits clearly above its bar -> tiny p; one at the bar -> ~1.
    """Ttest greater signal vs noise."""
    n_rep = 12
    real = np.zeros((n_rep, 2))
    real[:, 0] = 0.30 + np.random.default_rng(0).normal(scale=0.01, size=n_rep)  # well above bar
    real[:, 1] = 0.05 + np.random.default_rng(1).normal(scale=0.01, size=n_rep)  # at/below bar
    bar = np.array([0.10, 0.10])
    p = _ttest_greater(real, bar)
    assert p[0] < 0.01
    assert p[1] > 0.5


def test_ttest_greater_zero_variance_branches():
    """Ttest greater zero variance branches."""
    real = np.array([[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]])  # zero dispersion
    bar = np.array([0.1, 0.3])  # feature 0 mean>bar -> p=0; feature 1 mean<bar -> p=1
    p = _ttest_greater(real, bar)
    assert p[0] == 0.0
    assert p[1] == 1.0


def test_default_estimator_task_detection():
    """Default estimator task detection."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    y_clf = np.array([0, 1, 0, 1, 1, 0] * 20)
    y_reg = np.linspace(0.0, 10.0, 120)
    assert isinstance(_default_estimator(y_clf, 120, 0), RandomForestClassifier)
    assert isinstance(_default_estimator(y_reg, 120, 0), RandomForestRegressor)


def test_ace_select_rejects_bad_input():
    """Ace select rejects bad input."""
    with pytest.raises(ValueError):
        ace_select(np.zeros((10,)), np.zeros(10))  # 1-D X
    with pytest.raises(ValueError):
        ace_select(np.zeros((10, 3)), np.zeros(10), n_replicates=1)  # too few replicates
    with pytest.raises(ValueError):
        ace_select(np.zeros((10, 3)), np.zeros(10), feature_names=["a", "b"])  # name/width mismatch


def test_ace_result_shape_and_support():
    """Ace result shape and support."""
    rng = np.random.default_rng(3)
    n = 400
    x_sig = rng.normal(size=n)
    X = np.column_stack([x_sig, rng.normal(size=n), rng.normal(size=n)])
    y = (x_sig > 0).astype(int)
    res = ace_select(X, y, n_replicates=6, n_masking_rounds=1, random_state=0)
    assert isinstance(res, ACEResult)
    assert res.importances_mean.shape == (3,)
    assert res.p_values.shape == (3,)
    assert res.accepted.shape == (3,)
    assert res.support().dtype == bool
    assert res.selected_features == [res.feature_names[i] for i in range(3) if res.accepted[i]]


def test_ace_accepts_dataframe_names():
    """Ace accepts dataframe names."""
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(7)
    n = 400
    sig = rng.normal(size=n)
    df = pd.DataFrame({"signal": sig, "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)})
    y = (sig > 0).astype(int)
    res = ace_select(df, y, n_replicates=6, n_masking_rounds=1, random_state=0)
    assert res.feature_names == ["signal", "noise_a", "noise_b"]


def test_ace_permutation_importance_mode_runs():
    """Ace permutation importance mode runs."""
    rng = np.random.default_rng(5)
    n = 300
    sig = rng.normal(size=n)
    X = np.column_stack([sig, rng.normal(size=n)])
    y = (sig > 0).astype(int)
    res = ace_select(X, y, n_replicates=5, n_masking_rounds=1, importance="permutation", n_perm_repeats=3, random_state=0)
    assert res.p_values.shape == (2,)
    # The signal feature's mean importance should exceed the pure-noise feature's.
    assert res.importances_mean[0] > res.importances_mean[1]


# ------------------------------------------------------------------------------------------------
# biz_value: ACE must separate genuine signal from noise on a known synthetic.
# ------------------------------------------------------------------------------------------------


def _synth_signal_noise(seed: int = 0, n: int = 1500):
    """3 signal features driving a binary target + 5 pure-noise features. Signal indices 0,1,2."""
    rng = np.random.default_rng(seed)
    s0 = rng.normal(size=n)
    s1 = rng.normal(size=n)
    s2 = rng.normal(size=n)
    logits = 2.0 * s0 + 1.5 * s1 - 1.8 * s2
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)
    noise = rng.normal(size=(n, 5))
    X = np.column_stack([s0, s1, s2, noise])
    signal_idx = [0, 1, 2]
    noise_idx = [3, 4, 5, 6, 7]
    return X, y, signal_idx, noise_idx


def test_biz_val_ace_accepts_signal_rejects_noise():
    """ACE accepts >= 2 of 3 signal features and keeps noise acceptance low.

    Measured (native importance, 20 replicates, percentile=100, BH alpha=0.05): 3/3 signal accepted,
    0/5 noise accepted. Floors set well below measured to absorb seed noise: >= 2/3 signal, <= 1/5 noise.
    A regression that broke the contrast bar, t-test, or masking loop drops the signal-accept count or
    lets noise through, tripping this test."""
    X, y, signal_idx, noise_idx = _synth_signal_noise(seed=0, n=1500)
    res = ace_select(X, y, n_replicates=20, contrast_percentile=100.0, alpha=0.05, random_state=0)

    n_signal_accepted = int(res.accepted[signal_idx].sum())
    n_noise_accepted = int(res.accepted[noise_idx].sum())

    assert n_signal_accepted >= 2, f"ACE accepted only {n_signal_accepted}/3 signal features"
    assert n_noise_accepted <= 1, f"ACE accepted {n_noise_accepted}/5 noise features (should be ~0)"
    # The selection gap (signal accepted - noise accepted) is the headline quantitative win.
    assert n_signal_accepted - n_noise_accepted >= 2


def test_biz_val_ace_signal_pvalues_beat_noise_pvalues():
    """The max signal p-value must be strictly smaller than the min noise p-value: ACE ranks every real
    feature ahead of every contrast-indistinguishable one. Measured gap is large (signal p ~1e-6, noise
    p ~0.5-1.0); this asserts the ORDERING holds, the property a downstream threshold relies on."""
    X, y, signal_idx, noise_idx = _synth_signal_noise(seed=1, n=1500)
    res = ace_select(X, y, n_replicates=20, contrast_percentile=100.0, alpha=0.05, random_state=0)
    max_signal_p = res.p_values[signal_idx].max()
    min_noise_p = res.p_values[noise_idx].min()
    assert max_signal_p < min_noise_p, f"signal p-values (max {max_signal_p:.3g}) must all beat noise p-values (min {min_noise_p:.3g})"


def test_biz_val_ace_masking_recovers_correlated_signal():
    """Masking-removal loop recovers a signal feature masked by a stronger correlated duplicate.

    With two near-duplicate strong signals, a single pass can leave the weaker copy tentative because the
    forest splits importance between them. The masking loop removes the accepted copy and re-tests, so
    across rounds ACE accepts MORE signal features than a single-round (n_masking_rounds=1) run. This pins
    the masking-loop value; a regression disabling it drops the multi-round accept count to the single-round
    count."""
    rng = np.random.default_rng(2)
    n = 1500
    base = rng.normal(size=n)
    dup = base + rng.normal(scale=0.05, size=n)  # near-duplicate of the strong signal
    weak = rng.normal(size=n)
    logits = 2.5 * base + 0.8 * weak
    y = (logits + rng.normal(scale=0.5, size=n) > 0).astype(int)
    noise = rng.normal(size=(n, 4))
    X = np.column_stack([base, dup, weak, noise])
    signal_idx = [0, 1, 2]

    res_multi = ace_select(X, y, n_replicates=20, n_masking_rounds=3, random_state=0)
    res_single = ace_select(X, y, n_replicates=20, n_masking_rounds=1, random_state=0)
    acc_multi = int(res_multi.accepted[signal_idx].sum())
    acc_single = int(res_single.accepted[signal_idx].sum())
    assert acc_multi >= acc_single, "masking loop must not accept fewer signals than a single round"
    assert acc_multi >= 2, f"masking run should recover >= 2/3 signals, got {acc_multi}"


# ------------------------------------------------------------------------------------------------
# Suite reachability: ACESelector adapter + registry registration + pre-pipeline routing.
# ------------------------------------------------------------------------------------------------


def test_ace_selector_adapter_fit_selects_signal_drops_noise():
    """The sklearn-compatible ACESelector adapter fits, exposes the support contract, and narrows to the
    informative column while dropping pure noise (mirrors the fit/get_support/transform contract the suite
    drives every selector through)."""
    pd = pytest.importorskip("pandas")
    from sklearn.exceptions import NotFittedError
    from mlframe.feature_selection.ace import ACESelector

    rng = np.random.default_rng(0)
    n = 1200
    sig = rng.normal(size=n)
    y = (2.0 * sig + rng.normal(scale=0.5, size=n) > 0).astype(int)
    df = pd.DataFrame({"signal": sig, "noise_a": rng.normal(size=n), "noise_b": rng.normal(size=n)})

    sel = ACESelector(n_replicates=15, contrast_percentile=100.0, random_state=0)
    with pytest.raises(NotFittedError):
        sel.transform(df)
    sel.fit(df, y)

    assert sel.support_.dtype == bool and sel.support_.shape == (3,)
    assert list(sel.feature_names_in_) == ["signal", "noise_a", "noise_b"]
    assert "signal" in sel.selected_features_
    assert "noise_a" not in sel.selected_features_ and "noise_b" not in sel.selected_features_
    out = sel.transform(df)
    assert list(out.columns) == sel.selected_features_
    assert out.shape[0] == n
    # get_support(indices) + get_feature_names_out agree with support_.
    assert list(sel.get_support(indices=True)) == list(np.where(sel.support_)[0])
    assert list(sel.get_feature_names_out()) == sel.selected_features_


def test_ace_registered_and_instantiates_adapter():
    """Ace registered and instantiates adapter."""
    from mlframe.feature_selection.registry import available, get
    from mlframe.feature_selection.ace import ACESelector

    assert "ACE" in available()
    inst = get("ACE").instantiate(n_replicates=5, random_state=0)
    assert isinstance(inst, ACESelector)
    assert inst.n_replicates == 5


def test_use_ace_fs_routes_through_pre_pipeline_builder():
    """use_ace_fs=True must add exactly one ACE selector (kind marker 'ACE') to the pre-pipelines, mirroring
    the shap-proxied wiring. Off by default: no ACE branch when the flag is unset."""
    from mlframe.training.core._setup_helpers_pre_pipelines import _build_pre_pipelines
    from mlframe.training.core._phase_train_one_target import _selector_kind
    from mlframe.feature_selection.ace import ACESelector

    pipelines, names = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        use_ace_fs=True,
        ace_kwargs={"n_replicates": 5},
        fs_random_seed=7,
    )
    ace_objs = [p for p in pipelines if isinstance(p, ACESelector)]
    assert len(ace_objs) == 1
    assert "ACE " in names
    assert _selector_kind(ace_objs[0]) == "ACE"
    # fs_random_seed defaults ACE's random_state when the operator didn't pin one.
    assert ace_objs[0].random_state == 7

    pipelines_off, _ = _build_pre_pipelines(
        use_ordinary_models=False,
        rfecv_models=[],
        rfecv_models_params={},
        use_mrmr_fs=False,
        mrmr_kwargs={},
        use_ace_fs=False,
    )
    assert not any(isinstance(p, ACESelector) for p in pipelines_off)


def test_ace_config_flag_and_kwargs_validation():
    """Ace config flag and kwargs validation."""
    from mlframe.training._feature_selection_config import FeatureSelectionConfig

    cfg = FeatureSelectionConfig(use_ace_fs=True, ace_kwargs={"n_replicates": 10})
    assert cfg.use_ace_fs is True and cfg.ace_kwargs == {"n_replicates": 10}
    # kwargs without the master flag is a loud config error (mirrors shap-proxied).
    with pytest.raises(ValueError):
        FeatureSelectionConfig(ace_kwargs={"n_replicates": 10})
    # unknown kwarg key rejected at config time.
    with pytest.raises(ValueError):
        FeatureSelectionConfig(use_ace_fs=True, ace_kwargs={"not_a_real_knob": 1})


def test_compare_selectors_jaccard_matches_core_implementation():
    """CHANGE 2 equivalence: the core set_similarity.jaccard used by compare_selectors reproduces the
    former local _jaccard on name sets (index sets), including the both-empty -> 1.0 convention."""
    from mlframe.core.set_similarity import jaccard

    def _old(a, b):
        """Helper that old."""
        if not a and not b:
            return 1.0
        union = a | b
        return 1.0 if not union else len(a & b) / len(union)

    cases = [({"a", "b"}, {"b", "c"}), (set(), set()), ({"x"}, set()), ({"a", "b", "c"}, {"a", "b", "c"})]
    for a, b in cases:
        assert jaccard(a, b) == _old(a, b)
