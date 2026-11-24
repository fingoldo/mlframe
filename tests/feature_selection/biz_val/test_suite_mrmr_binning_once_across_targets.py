"""One ``train_mlframe_models_suite`` call with MRMR feature selection and targets of two types must not bin a feature again when its binning does not depend on the target (PLAN-09).

MRMR level: ``tests/feature_selection/mrmr/test_mrmr_binning_once_per_target.py`` pins that an unsupervised ``binning_scheme`` bins each feature column once across two fits in one process (the edge memo
keys on content). This test pins the same property at the level of the whole suite, which may fit MRMR per target in other threads or processes, on a copy or a cast of the frame, or after a transform.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins_columns as cols
from mlframe.training import FeatureSelectionConfig, OutputConfig
from mlframe.training.configs import ReportingConfig
from mlframe.training.core import train_mlframe_models_suite
from tests.training.shared import SimpleFeaturesAndTargetsExtractor
from tests.feature_selection.binning_cases import binning_kwargs

pytestmark = pytest.mark.timeout(1800)


def _frame(n: int = 600, p: int = 8, seed: int = 0) -> pd.DataFrame:
    """Numeric features, a regression target ``target`` on top of them; the extractor adds a binary target of the second type."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, p)), columns=[f"f{i}" for i in range(p)])
    X["target"] = (X["f0"] + 0.5 * X["f1"] * X["f2"] + 0.3 * rng.standard_normal(n)).astype(float)
    return X


def _count_uncached(monkeypatch) -> list:
    """The column-edge computations that really run (a memo hit does not reach the uncached builder), each recorded as a copy of the column."""
    calls: list = []
    real = cols._compute_col_edges_uncached

    def counting(col, *args, **kwargs):
        """Record one real computation (a copy of the column) and delegate."""
        calls.append(np.array(col, copy=True))
        return real(col, *args, **kwargs)

    monkeypatch.setattr(cols, "_compute_col_edges_uncached", counting)
    return calls


def _run_suite(df: pd.DataFrame, strategy: str):
    """One suite call: a regression target and a binary target on the same frame, MRMR selection with the given strategy and no feature engineering (only the binning of the raw columns is counted)."""
    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True, extra_targets="mixed_reg_bin")
    mrmr = {"verbose": 0, "random_seed": 0, "n_workers": 1, "fe_max_steps": 0, **binning_kwargs(strategy)}
    with tempfile.TemporaryDirectory() as d:
        return train_mlframe_models_suite(
            df=df,
            target_name="t",
            model_name="binning_once",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            hyperparams_config={"iterations": 5},
            reporting_config=ReportingConfig(show_perf_chart=False, show_fi=False),
            use_ordinary_models=True,
            use_mlframe_ensembles=False,
            output_config=OutputConfig(data_dir=d, models_dir="models", save_charts=False),
            verbose=0,
            feature_selection_config=FeatureSelectionConfig(mrmr=mrmr),
        )


def test_unsupervised_strategy_bins_each_feature_once_across_the_targets_of_one_suite_call(monkeypatch):
    """Two targets of different types, the default strategy: every feature column is binned once, however many targets the call trains."""
    df = _frame()
    cols.clear_edge_memo()
    calls = _count_uncached(monkeypatch)
    result, _meta = _run_suite(df, "rice_capped")
    assert len(result) >= 2, f"the call must train two target types, got {list(result)}"
    # The suite hands MRMR a train split, not the frame it was given, so the columns are compared with each other: with no feature engineering the computations are the feature columns and each target's own
    # column (distinct per target), so a repeated content is a feature binned again.
    n_features = df.shape[1] - 1
    assert len(calls) >= n_features, "the suite must bin the feature columns at least once"
    keys = [hash(np.ascontiguousarray(a).tobytes()) for a in calls]
    assert len(keys) == len(set(keys)), f"{len(keys) - len(set(keys))} binning(s) repeated an earlier one across the targets ({len(keys)} computed for {n_features} features and 2 targets)"


def test_supervised_strategy_rebins_for_another_target_contrast(monkeypatch):
    """Contrast: a strategy that depends on the target (``mdlp``) must bin a feature again for the second target, so a count of zero repeats would mean the counter is blind."""
    df = _frame()
    cols.clear_edge_memo()
    calls = _count_uncached(monkeypatch)
    _run_suite(df, "mdlp")
    keys = [hash(np.ascontiguousarray(a).tobytes()) for a in calls]
    assert len(keys) > len(set(keys)), "a supervised strategy should re-bin each feature for the second target"
