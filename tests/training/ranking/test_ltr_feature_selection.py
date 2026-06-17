"""LTR-local feature selection for the ranker suite.

Covers the separate ``ranking._ranker_fs`` selector (it does NOT touch core MRMR) and its wiring into
``train_mlframe_ranker_suite`` via ``LearningToRankConfig.feature_selection``. Also pins the relevance-leak
regression: an FTE that declares the target via ``learning_to_rank_targets`` (no ``target_column`` attr) must
not let the relevance column leak into the ranker feature matrix.
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


# --- unit: the separate selector ------------------------------------------------------------------


def test_select_ltr_features_pointwise_recovers_signal():
    from mlframe.training.ranking._ranker_fs import select_ltr_features

    df, signal, noise = _ltr_frame(0)
    X = df.drop(columns=["target", "qid"])
    sel = select_ltr_features(X, df["target"].to_numpy(), df["qid"].to_numpy(), mrmr_kwargs={"quantization_nbins": 5})
    assert set(signal) <= set(sel), f"signal lost: {sel}"
    assert not (set(noise) & set(sel)), f"noise leaked into selection: {sel}"


def test_select_ltr_features_group_aware_recovers_signal():
    from mlframe.training.ranking._ranker_fs import select_ltr_features

    df, signal, noise = _ltr_frame(1)
    X = df.drop(columns=["target", "qid"])
    sel = select_ltr_features(
        X, df["target"].to_numpy(), df["qid"].to_numpy(),
        group_aware_mi=True, mrmr_kwargs={"quantization_nbins": 5},
    )
    assert set(signal) <= set(sel), f"group-aware signal lost: {sel}"


def test_group_aware_relevance_ranks_signal_above_noise():
    """biz_value: per-query group-aware MI must score the generating features above every noise column."""
    from mlframe.training.ranking._ranker_fs import group_aware_relevance

    df, signal, noise = _ltr_frame(2)
    cols = [c for c in df.columns if c not in ("target", "qid")]
    rel = group_aware_relevance(cols, df[cols].to_numpy(np.float64), df["target"].to_numpy(np.float64), df["qid"].to_numpy())
    min_signal = min(rel[c] for c in signal)
    max_noise = max(rel[c] for c in noise)
    assert min_signal > max_noise, f"group-aware MI failed to separate signal from noise: signal_min={min_signal:.4f} noise_max={max_noise:.4f}"


def test_select_ltr_features_fallback_on_degenerate_input():
    """All-constant features must not crash; selector falls back to the candidate columns."""
    from mlframe.training.ranking._ranker_fs import select_ltr_features

    X = pd.DataFrame({"a": np.ones(50), "b": np.ones(50)})
    sel = select_ltr_features(X, np.arange(50) % 3)
    assert set(sel) <= {"a", "b"}


# --- e2e through the suite ------------------------------------------------------------------


def _train_ltr(df, fte, ranking_config, **kw):
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.configs import TargetTypes, ReportingConfig, OutputConfig

    with tempfile.TemporaryDirectory() as d:
        return train_mlframe_models_suite(
            df=df, target_name="t", model_name="ltr", features_and_targets_extractor=fte,
            mlframe_models=["cb"], target_type=TargetTypes.LEARNING_TO_RANK, ranking_config=ranking_config,
            reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False),
            output_config=OutputConfig(data_dir=d, models_dir="models", save_charts=False),
            verbose=0, hyperparams_config={"iterations": 15}, **kw,
        )


def test_e2e_ltr_suite_feature_selection_excludes_noise():
    pytest.importorskip("catboost")
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor
    from mlframe.training._model_configs_behavior import LearningToRankConfig

    df, signal, noise = _ltr_frame(0)
    fte = SimpleFeaturesAndTargetsExtractor(learning_to_rank_targets=["target"], group_field="qid")
    _res, meta = _train_ltr(
        df, fte,
        LearningToRankConfig(feature_selection=True, fs_mrmr_kwargs={"use_simple_mode": True, "quantization_nbins": 5}),
    )
    sel = meta.get("selected_features")
    assert sel, "feature_selection=True produced no selected_features"
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
    _res, meta = _train_ltr(df, fte, ranking_config=None)  # FS off -> exercises the plain column-drop path
    cols = meta.get("columns") or []
    assert "target" not in cols, f"relevance column leaked into ranker features: {cols}"
    assert "qid" not in cols, f"group column leaked into ranker features: {cols}"
