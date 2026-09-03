"""Regression: for a non-polars strategy with no pre_pipeline, the pipeline cache key must still
discriminate by actual X content -- it must NOT degenerate into a constant string independent of
the data.

Live incident: ``compute_model_pipeline_cache_key`` passed ``train_df=None`` to
``_compute_pipeline_cache_key`` for any non-polars strategy (``train_df_polars if
strategy.supports_polars else None``), on the theory that "pandas frames don't reach this branch
typed-distinct enough to need the suffix -- handled upstream in split_features." With no named
pre_pipeline, ``_compute_pipeline_cache_key``'s target-discriminator is ALSO gated off (by design,
to preserve legitimate same-X-multi-target cache sharing -- see
test_pipeline_cache_key_target_discrimination.py). The two gates together left ZERO content
discriminator for the extremely common "plain tree-model fit, no cat/text/embedding, no FS" case:
two ENTIRELY UNRELATED ``train_mlframe_models_suite`` calls (different X shape -- 11 columns vs 3,
different targets) produced the IDENTICAL key
``"lgb_tier(False, False)_kindpd_feats69dda5e0bd93b362"``, and ``PipelineCache`` served the first
call's fitted/cached artefacts to the second -- reproduced end-to-end as a ``LightGBMError: number
of features in data (3) is not the same as it was in training data (11)`` at predict time
(test_registered_composite_model_keys_suite.py::
test_lgb_string_key_dispatch_unaffected_by_gated_outlier_registration).

Fix: fall back to the pandas ``train_df`` (already available on ``ctx`` as ``train_df_pd``) when
the strategy isn't polars-native, so the existing, already backend-agnostic ``_dtype_suffix`` logic
folds a real schema fingerprint into the key for pandas paths too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.core._phase_train_one_target import _compute_pipeline_cache_key
from mlframe.training.core._phase_train_one_target_cache_helpers import compute_model_pipeline_cache_key
from mlframe.training.strategies import LinearModelStrategy

pytest.importorskip("sklearn")


def _frame(n_cols: int, n_rows: int = 20, seed: int = 0) -> pd.DataFrame:
    """A plain all-numeric pandas frame with ``n_cols`` columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"f{i}": rng.standard_normal(n_rows).astype(np.float32) for i in range(n_cols)})


def test_different_schema_no_pre_pipeline_pandas_frames_get_different_keys():
    """Two unrelated pandas frames (different column counts) with no pre_pipeline must NOT collide."""
    strategy = LinearModelStrategy()
    assert not strategy.supports_polars, "this test targets the non-polars-strategy branch"

    key_11 = compute_model_pipeline_cache_key(
        strategy=strategy,
        pre_pipeline_name=None,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        train_df_polars=None,
        train_df_pd=_frame(11),
        cur_target_name="target_a",
        current_train_target=np.array([1.0, 2.0, 3.0]),
        _compute_pipeline_cache_key=_compute_pipeline_cache_key,
    )
    key_3 = compute_model_pipeline_cache_key(
        strategy=strategy,
        pre_pipeline_name=None,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        train_df_polars=None,
        train_df_pd=_frame(3),
        cur_target_name="target_b",
        current_train_target=np.array([9.0, 9.0, 9.0, 9.0, 9.0]),
        _compute_pipeline_cache_key=_compute_pipeline_cache_key,
    )
    assert key_11 != key_3, f"cache key collided across pandas frames with different schemas: {key_11!r} == {key_3!r}"


def test_same_schema_no_pre_pipeline_pandas_frames_still_share_the_slot():
    """The legitimate optimization this fix must not break: SAME X schema, no pre_pipeline ->
    the frame IS target-independent (imp/scale/encode doesn't look at y), so different targets
    over an otherwise-identical schema still share one cache slot."""
    strategy = LinearModelStrategy()
    same_shape_a = _frame(5, seed=1)
    same_shape_b = _frame(5, seed=1)  # identical columns AND dtypes (schema-equal, not object-equal)

    key_a = compute_model_pipeline_cache_key(
        strategy=strategy,
        pre_pipeline_name=None,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        train_df_polars=None,
        train_df_pd=same_shape_a,
        cur_target_name="target_a",
        current_train_target=np.array([1.0, 2.0]),
        _compute_pipeline_cache_key=_compute_pipeline_cache_key,
    )
    key_b = compute_model_pipeline_cache_key(
        strategy=strategy,
        pre_pipeline_name=None,
        cat_features=None,
        text_features=None,
        embedding_features=None,
        train_df_polars=None,
        train_df_pd=same_shape_b,
        cur_target_name="target_b",
        current_train_target=np.array([9.0, 9.0, 9.0]),
        _compute_pipeline_cache_key=_compute_pipeline_cache_key,
    )
    assert key_a == key_b, "same-schema, no-pre_pipeline pandas frames must still share the cache slot across targets"
