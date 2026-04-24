"""Fix B — Hypothesis-driven continuous leaf-value sampling for the
combo fuzz suite.

Complement to the pairwise (``test_fuzz_suite.py``) and 3-wise
(``test_fuzz_3way_suite.py``) suites. Those enumerate discrete axis
values from hand-curated lists; this test runs a much smaller number
of random configurations where the "leaf" numeric values (``n_rows``,
``fillna_value``, ``test_size``, ``cat_feature_count``) are drawn
continuously by Hypothesis. Benefit: when a combo fails, Hypothesis
automatically shrinks to the minimum failing input (smallest n_rows,
smallest test_size, etc.), giving us a tight repro instead of the
~600-row synthetic the pairwise suite produces.

This file is the default Hypothesis ``max_examples=20`` — enough to
probe the continuous space without exploding wall-clock. Raise
``MLFRAME_HYPOTHESIS_EXAMPLES`` for deeper sweeps.
"""
from __future__ import annotations

import os
from datetime import timedelta

import pytest

from hypothesis import HealthCheck, given, settings, strategies as st

from ._fuzz_combo import FuzzCombo, build_frame_for_combo
from .shared import SimpleFeaturesAndTargetsExtractor
from .test_fuzz_suite import (
    _assert_prediction_invariants,
    _common_init_for_combo,
    _config_for_models,
    _configs_for_combo,
    _custom_pre_pipelines_for_combo,
    _outlier_detector_for_combo,
    _skip_if_deps_missing,
)


_MAX_EXAMPLES = int(os.environ.get("MLFRAME_HYPOTHESIS_EXAMPLES", "20"))


# Continuous strategies for the leaf axes Hypothesis drives. Discrete
# axes (input_type, models, target_type) come from ``st.sampled_from``
# with small value sets — pairwise coverage is the job of the other
# two suites; here we explore the continuous neighbourhood around a
# few discrete points.
_leaf_strategy = st.fixed_dictionaries({
    "n_rows": st.integers(min_value=120, max_value=1500),
    "cat_feature_count": st.integers(min_value=0, max_value=5),
    "null_fraction_cats": st.floats(min_value=0.0, max_value=0.4, allow_nan=False),
    "test_size": st.floats(min_value=0.05, max_value=0.35, allow_nan=False),
    "fillna_value": st.one_of(st.none(), st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)),
    "iterations": st.integers(min_value=3, max_value=15),
})

_discrete_strategy = st.fixed_dictionaries({
    "input_type": st.sampled_from(["pandas", "polars_utf8", "polars_enum", "polars_nullable"]),
    "models": st.sampled_from([("cb",), ("xgb",), ("lgb",), ("hgb",), ("linear",), ("cb", "xgb"), ("lgb", "xgb")]),
    "target_type": st.sampled_from(["binary_classification", "regression"]),
    "categorical_encoding_cfg": st.sampled_from(["ordinal", "onehot"]),
    "scaler_name_cfg": st.sampled_from(["standard", "robust", None]),
    "inject_inf_nan": st.booleans(),
    "inject_label_leak": st.booleans(),
})


def _combo_from_hyp(leaf: dict, discrete: dict, seed: int) -> FuzzCombo:
    """Assemble a FuzzCombo from Hypothesis-drawn leaves + discrete set."""
    return FuzzCombo(
        models=tuple(sorted(discrete["models"])),
        input_type=discrete["input_type"],
        n_rows=leaf["n_rows"],
        cat_feature_count=leaf["cat_feature_count"],
        null_fraction_cats=leaf["null_fraction_cats"],
        use_mrmr_fs=False,
        weight_schemas=("uniform",),
        target_type=discrete["target_type"],
        auto_detect_cats=True,
        align_polars_categorical_dicts=True,
        seed=seed,
        inject_inf_nan=leaf.get("inject_inf_nan", False) or discrete["inject_inf_nan"],
        iterations=leaf["iterations"],
        fillna_value_cfg=leaf["fillna_value"],
        test_size_cfg=leaf["test_size"],
        scaler_name_cfg=discrete["scaler_name_cfg"],
        categorical_encoding_cfg=discrete["categorical_encoding_cfg"],
        inject_label_leak=discrete["inject_label_leak"],
    )


@settings(
    max_examples=_MAX_EXAMPLES,
    deadline=timedelta(seconds=180),
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
@given(leaf=_leaf_strategy, discrete=_discrete_strategy, seed=st.integers(0, 99999))
def test_hypothesis_leaf_sampling(tmp_path_factory, leaf, discrete, seed):
    """Single Hypothesis test: run the suite under a combo whose
    continuous leaves were drawn by Hypothesis. Uses tmp_path_factory
    so each example gets a fresh tmpdir without the function_scoped
    warning."""
    combo = _combo_from_hyp(leaf, discrete, seed)
    _skip_if_deps_missing(combo.models)
    tmp_path = tmp_path_factory.mktemp(f"hyp_{seed}_{leaf['n_rows']}")

    df, target_col, _ = build_frame_for_combo(combo)
    fte = SimpleFeaturesAndTargetsExtractor(
        target_column=target_col,
        regression=(combo.target_type == "regression"),
    )
    outlier_detector = _outlier_detector_for_combo(combo)
    custom_pre = _custom_pre_pipelines_for_combo(combo)

    from mlframe.training.core import train_mlframe_models_suite

    trained, _meta = train_mlframe_models_suite(
        df=df,
        target_name=combo.short_id(),
        model_name=combo.short_id(),
        features_and_targets_extractor=fte,
        mlframe_models=list(combo.models),
        hyperparams_config=_config_for_models(
            combo.models, combo.n_rows,
            iterations=combo.iterations,
            early_stopping_rounds=combo.early_stopping_rounds_cfg,
        ),
        init_common_params=_common_init_for_combo(combo),
        use_ordinary_models=True,
        use_mlframe_ensembles=False,
        outlier_detector=outlier_detector,
        custom_pre_pipelines=custom_pre,
        data_dir=str(tmp_path),
        models_dir="models",
        verbose=0,
        use_mrmr_fs=False,
        **_configs_for_combo(combo),
    )
    # Same invariants the pairwise suite asserts.
    _assert_prediction_invariants(trained, _meta, combo)
    # Basic crash-free: we got here, so train completed.
