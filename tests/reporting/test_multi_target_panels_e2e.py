"""End-to-end smoke for multi-target panel auto-emission through the
real ``train_mlframe_models_suite`` path with an actual model fit.

Auto-dispatch unit tests already cover the dispatcher in isolation
(``tests/reporting/test_auto_dispatch.py``); this test verifies the
full chain: ``ReportingConfig.{plot_outputs,multiclass_panels,multilabel_panels}``
-> ``train_and_evaluate_model`` -> ``_compute_split_metrics``
-> ``report_model_perf`` -> ``render_multi_target_panels`` ->
file at ``{plot_file}_<split>_{multiclass,multilabel}_panels.png``.

Kept tiny (200 rows, lgb-only, 30 iterations) so each test runs in
a few seconds on CPU.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import (
    OutputConfig,
    ReportingConfig,
    TargetTypes,
)
from mlframe.training.core import train_mlframe_models_suite
from tests.training.shared import SimpleFeaturesAndTargetsExtractor


pytestmark = pytest.mark.slow


# ----------------------------------------------------------------------------
# Tiny synthetic fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def mc_dataset():
    """200 rows, 4 features, 3 well-separated classes."""
    rng = np.random.default_rng(42)
    n, _K = 200, 3
    X = rng.standard_normal((n, 4))
    # Plant signal: class assignment depends on f0+f1.
    score = X[:, 0] - 0.5 * X[:, 1] + 0.3 * rng.standard_normal(n)
    y = np.digitize(score, [-0.5, 0.5]).astype(np.int64)  # 3 buckets
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["test_target"] = y
    return df


@pytest.fixture
def ml_dataset():
    """200 rows, 4 features, 3 binary labels (correlated to features)."""
    rng = np.random.default_rng(42)
    n, K = 200, 3
    X = rng.standard_normal((n, 4)).astype(np.float32)
    y = np.zeros((n, K), dtype=np.int8)
    y[:, 0] = (X[:, 0] > 0).astype(np.int8)
    y[:, 1] = (X[:, 1] > 0.3).astype(np.int8)
    y[:, 2] = (X[:, 0] + X[:, 1] > 0.5).astype(np.int8)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    # Pack labels as object column with list cells (per FTE contract).
    df["test_target"] = list(y)
    return df


# ----------------------------------------------------------------------------
# E2E
# ----------------------------------------------------------------------------


class TestMultiTargetPanelE2E:
    def test_multiclass_panels_auto_emitted(self, mc_dataset, tmp_path):
        """Default ReportingConfig (multiclass_panels = full template +
        plot_outputs = matplotlib) emits panel files per (model, split)
        through the real suite + report_model_perf path."""
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="test_target",
            target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
        )
        # Pin matplotlib only for speed (skip plotly subplot HTML).
        reporting = ReportingConfig(
            plot_outputs="matplotlib[png]",
            multiclass_panels="CONFUSION PR_F1",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_models_suite(
                df=mc_dataset,
                target_name="test_target",
                model_name="mc_e2e",
                features_and_targets_extractor=fte,
                target_type=TargetTypes.MULTICLASS_CLASSIFICATION,
                mlframe_models=["xgb"],
                hyperparams_config={"iterations": 30},
                reporting_config=reporting,
                output_config=OutputConfig(
                    data_dir=str(tmp_path),
                    models_dir="models",
                ),
                use_mlframe_ensembles=False,
                verbose=0,
            )
        # Multiclass panel files must appear in tmp_path/charts/...
        # The exact subdirectory layout depends on plot_file resolution
        # inside core.py (build_paths -> data_dir/charts/...). Search
        # recursively for the suffix.
        # Smart-naming policy: single backend × single fmt -> base.fmt;
        # otherwise base.<backend>.<fmt>. Match either.
        emitted = list(tmp_path.rglob("*_multiclass_panels.png")) + list(tmp_path.rglob("*_multiclass_panels.*.png"))
        assert emitted, (
            "no multiclass panel file emitted -- ReportingConfig "
            "did not flow through to render_multi_target_panels. "
            f"Files in tmp_path: {sorted(p.name for p in tmp_path.rglob('*'))}"
        )
        # At least one file must be non-trivially sized.
        assert any(f.stat().st_size > 5000 for f in emitted)

    def test_multilabel_panels_auto_emitted(self, ml_dataset, tmp_path):
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="test_target",
            target_type=TargetTypes.MULTILABEL_CLASSIFICATION,
        )
        reporting = ReportingConfig(
            plot_outputs="matplotlib[png]",
            multilabel_panels="PR_F1 COOCCURRENCE",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_mlframe_models_suite(
                df=ml_dataset,
                target_name="test_target",
                model_name="ml_e2e",
                features_and_targets_extractor=fte,
                target_type=TargetTypes.MULTILABEL_CLASSIFICATION,
                mlframe_models=["xgb"],
                hyperparams_config={"iterations": 30},
                reporting_config=reporting,
                output_config=OutputConfig(
                    data_dir=str(tmp_path),
                    models_dir="models",
                ),
                use_mlframe_ensembles=False,
                verbose=0,
            )
        emitted = list(tmp_path.rglob("*_multilabel_panels.png")) + list(tmp_path.rglob("*_multilabel_panels.*.png"))
        assert emitted, (
            "no multilabel panel file emitted -- ReportingConfig "
            "did not flow through to render_multi_target_panels. "
            f"Files in tmp_path: {sorted(p.name for p in tmp_path.rglob('*'))}"
        )
