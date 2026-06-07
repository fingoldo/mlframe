"""Suite-side wiring of LTR panel rendering.

Verifies that ``train_mlframe_ranker_suite`` automatically calls
``render_multi_target_panels`` per (flavor, split) when the new
``plot_file`` / ``plot_outputs`` / ``ltr_panels`` kwargs are supplied,
and that legacy callers (none of those kwargs set) see no panel files
written -- back-compat preserved.

Kept tiny on purpose (~50 queries × 5 docs, 50 boost iters) so the
test runs in a few seconds even with three rankers.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import TargetTypes
from mlframe.training.extractors import FeaturesAndTargetsExtractor
from mlframe.training.ranking.ranker_suite import train_mlframe_ranker_suite


class _MiniRankFTE(FeaturesAndTargetsExtractor):
    def __init__(self):
        super().__init__(group_field="qid")

    def build_targets(self, df):
        rel = df["relevance"]
        if hasattr(rel, "to_numpy"):
            rel = rel.to_numpy()
        return {TargetTypes.LEARNING_TO_RANK: {"relevance": np.asarray(rel)}}


@pytest.fixture
def mini_search_data():
    """50 queries × 5 docs = 250 rows. Strong signal in feature 0."""
    rng = np.random.default_rng(42)
    n_q, n_per = 50, 5
    n = n_q * n_per
    qid = np.repeat(np.arange(n_q), n_per)
    X = rng.standard_normal((n, 3)).astype(np.float32)
    score = 1.5 * X[:, 0] + 0.3 * rng.standard_normal(n)
    y = np.clip(np.round(score + 1.5), 0, 3).astype(int)
    df = pd.DataFrame(X, columns=["f0", "f1", "f2"])
    df["qid"] = qid
    df["relevance"] = y
    return df


class TestLTRPanelWiring:
    def test_panels_emitted_per_flavor_per_split(self, mini_search_data, tmp_path):
        """With plot_outputs + ltr_panels set, every (flavor, split)
        combo MUST drop a panel file at the expected path."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            models, _ = train_mlframe_ranker_suite(
                df=mini_search_data,
                target_name="relevance",
                model_name="wt",
                features_and_targets_extractor=_MiniRankFTE(),
                mlframe_models=["cb", "xgb", "lgb"],
                use_mlframe_ensembles=True,
                iterations=50,
                early_stopping_rounds=10,
                plot_file=str(tmp_path / "ltr"),
                plot_outputs="matplotlib[png]",
                ltr_panels="NDCG_K MRR_DIST",
                verbose=0,
            )
        # Per-flavor val + test files.
        for flavor in ["cb", "xgb", "lgb"]:
            for split in ["val", "test"]:
                expected = tmp_path / f"ltr_wt_{flavor}_{split}_ltr_panels.png"
                assert expected.exists(), f"missing {expected.name}"
                assert expected.stat().st_size > 5000
        # Ensemble files.
        for split in ["val", "test"]:
            expected = tmp_path / f"ltr_wt_ensemble_{split}_ltr_panels.png"
            assert expected.exists(), f"missing ensemble {expected.name}"

    def test_no_panels_when_kwargs_omitted(self, mini_search_data, tmp_path):
        """Legacy back-compat: caller doesn't pass plot_outputs /
        ltr_panels -- no panel files appear under tmp_path."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_ranker_suite(
                df=mini_search_data,
                target_name="relevance",
                model_name="wt",
                features_and_targets_extractor=_MiniRankFTE(),
                mlframe_models=["cb"],
                use_mlframe_ensembles=False,
                iterations=50,
                early_stopping_rounds=10,
                # no plot_file / plot_outputs / ltr_panels
                verbose=0,
            )
        assert not list(tmp_path.glob("*_ltr_panels.*"))

    def test_no_panels_when_only_plot_file_set(self, mini_search_data, tmp_path):
        """plot_file alone is not enough -- need the DSL too. Caller
        opt-in surface is symmetric across the 3 fields."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_ranker_suite(
                df=mini_search_data,
                target_name="relevance",
                model_name="wt",
                features_and_targets_extractor=_MiniRankFTE(),
                mlframe_models=["cb"],
                use_mlframe_ensembles=False,
                iterations=50,
                early_stopping_rounds=10,
                plot_file=str(tmp_path / "ltr"),
                # plot_outputs / ltr_panels NOT set
                verbose=0,
            )
        assert not list(tmp_path.glob("*_ltr_panels.*"))

    def test_dual_backend_emits_both(self, mini_search_data, tmp_path):
        """Multi-backend DSL emits both .matplotlib.png and .plotly.html."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_ranker_suite(
                df=mini_search_data,
                target_name="relevance",
                model_name="wt",
                features_and_targets_extractor=_MiniRankFTE(),
                mlframe_models=["cb"],
                use_mlframe_ensembles=False,
                iterations=50,
                early_stopping_rounds=10,
                plot_file=str(tmp_path / "ltr"),
                plot_outputs="matplotlib[png] + plotly[html]",
                ltr_panels="NDCG_K",
                verbose=0,
            )
        assert (tmp_path / "ltr_wt_cb_val_ltr_panels.matplotlib.png").exists()
        assert (tmp_path / "ltr_wt_cb_val_ltr_panels.plotly.html").exists()
