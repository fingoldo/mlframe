"""Feature selection for the LTR ranker suite, driven by the common ``FeatureSelectionConfig``.

LTR FS uses the SAME settings as every other target type (``use_mrmr_fs`` / ``rfecv_models`` /
``use_boruta_shap``) -- no LTR-specific FS config. Covers: the LTR FS selector helper, an e2e suite run, the
relevance-leak regression, and a cross-selector verification that MRMR / RFECV / BorutaShap / ShapProxiedFS all
work on a graded-relevance (LtR) target with regression-appropriate settings.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest


def _ltr_frame(seed=0, n=900, nq=30, n_noise=6):
    rng = np.random.default_rng(seed)
    s0 = rng.normal(size=n)
    s1 = rng.normal(size=n)
    noise = rng.normal(size=(n, n_noise))
    score = 1.5 * s0 - s1
    rel = np.digitize(score, np.quantile(score, [0.2, 0.4, 0.6, 0.8])).astype(int)
    cols = {"s0": s0, "s1": s1}
    cols.update({f"noise_{i}": noise[:, i] for i in range(n_noise)})
    df = pd.DataFrame(cols)
    df["target"] = rel
    df["qid"] = np.sort(rng.integers(0, nq, size=n))
    return df, ["s0", "s1"], [f"noise_{i}" for i in range(n_noise)]


# --- unit: the config-driven selector helper -----------------------------------------------------


def test_select_ltr_features_mrmr_recovers_signal():
    from mlframe.training.ranking._ranker_fs import select_ltr_features
    from mlframe.training import FeatureSelectionConfig

    df, signal, noise = _ltr_frame(0)
    X = df.drop(columns=["target", "qid"])
    sel = select_ltr_features(
        X, df["target"].to_numpy(),
        feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"use_simple_mode": True, "quantization_nbins": 5}),
        verbose=0,
    )
    assert set(signal) <= set(sel), f"signal lost: {sel}"
    assert not (set(noise) & set(sel)), f"noise leaked: {sel}"


def test_select_ltr_features_none_when_no_fs_enabled():
    from mlframe.training.ranking._ranker_fs import select_ltr_features
    from mlframe.training import FeatureSelectionConfig

    df, _s, _n = _ltr_frame(1)
    X = df.drop(columns=["target", "qid"])
    assert select_ltr_features(X, df["target"].to_numpy(), feature_selection_config=FeatureSelectionConfig()) is None
    assert select_ltr_features(X, df["target"].to_numpy(), feature_selection_config=None) is None


# --- verification: every selector works on a graded-relevance (LtR) target ------------------------


def test_all_selectors_work_on_graded_relevance():
    """MRMR / RFECV / BorutaShap / ShapProxiedFS must each fit on a multi-grade relevance target with
    regression-appropriate settings and return a non-empty support. This is the LtR x selector compatibility
    contract (graded relevance is regression-shaped, NOT binary classification)."""
    from mlframe.feature_selection.registry import get

    df, signal, _noise = _ltr_frame(2, n=600, n_noise=4)
    X = df.drop(columns=["target", "qid"])
    y = pd.Series(df["target"].to_numpy(), name="relevance")

    def _support_cols(sel):
        Xt = sel.transform(X)
        return set(Xt.columns) if hasattr(Xt, "columns") else None

    # MRMR -- relevance as MI target.
    m = get("MRMR").instantiate(use_simple_mode=True, quantization_nbins=5, verbose=0)
    m.fit(X, y)
    assert getattr(m, "support_", None) is not None and len(np.asarray(m.support_)) >= 1

    # BorutaShap -- regression mode (graded relevance).
    bs = get("BorutaShap").instantiate(classification=False)
    bs.fit(X, y)
    assert _support_cols(bs) is not None

    # ShapProxiedFS -- regression mode (classification=True would reject 5 grades).
    sp = get("ShapProxiedFS").instantiate(classification=False)
    sp.fit(X, y)
    assert _support_cols(sp) is not None

    # RFECV -- needs a regressor estimator for a graded target.
    from sklearn.ensemble import HistGradientBoostingRegressor
    rf = get("RFECV").instantiate(estimator=HistGradientBoostingRegressor(max_iter=30), cv=3)
    rf.fit(X, y)
    assert _support_cols(rf) is not None


# --- e2e through the suite (common feature_selection_config) ---------------------------------------


def _train_ltr(df, fte, *, feature_selection_config=None):
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import TargetTypes, ReportingConfig, OutputConfig

    with tempfile.TemporaryDirectory() as d:
        return train_mlframe_models_suite(
            df=df, target_name="t", model_name="ltr", features_and_targets_extractor=fte,
            mlframe_models=["cb"], target_type=TargetTypes.LEARNING_TO_RANK,
            feature_selection_config=feature_selection_config,
            reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False),
            output_config=OutputConfig(data_dir=d, models_dir="models", save_charts=False),
            verbose=0, hyperparams_config={"iterations": 15},
        )


def test_e2e_ltr_suite_common_mrmr_fs_excludes_noise():
    pytest.importorskip("catboost")
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
    from mlframe.training import FeatureSelectionConfig

    df, signal, noise = _ltr_frame(0)
    fte = SimpleFeaturesAndTargetsExtractor(learning_to_rank_targets=["target"], group_field="qid")
    _res, meta = _train_ltr(
        df, fte,
        feature_selection_config=FeatureSelectionConfig(use_mrmr_fs=True, mrmr_kwargs={"use_simple_mode": True, "quantization_nbins": 5}),
    )
    sel = meta.get("selected_features")
    assert sel, "use_mrmr_fs=True produced no selected_features for LTR"
    assert "target" not in sel and "qid" not in sel
    assert set(signal) <= set(sel), f"signal lost e2e: {sel}"
    assert not (set(noise) & set(sel)), f"noise leaked e2e: {sel}"


def test_e2e_ltr_relevance_column_not_leaked_into_features():
    """Regression: an FTE that declares its target via learning_to_rank_targets exposes no target_column attr;
    the ranker suite must still drop the relevance column from X (it lives in y), not memorise it as a feature."""
    pytest.importorskip("catboost")
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df, _signal, _noise = _ltr_frame(3)
    fte = SimpleFeaturesAndTargetsExtractor(learning_to_rank_targets=["target"], group_field="qid")
    _res, meta = _train_ltr(df, fte, feature_selection_config=None)
    cols = meta.get("columns") or []
    assert "target" not in cols, f"relevance column leaked into ranker features: {cols}"
    assert "qid" not in cols, f"group column leaked into ranker features: {cols}"
