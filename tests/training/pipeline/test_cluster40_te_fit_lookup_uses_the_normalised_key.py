"""Cluster 40 regression: fit-time val/test lookup normalises entity ids like the predict-time replay (audit 2026-11-20, finding 40-F25)."""

import numpy as np
import pandas as pd

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._target_encoding_composite_fe import apply_target_encoding_composite_fe, replay_target_encoding_composite_fe


def test_a_float_id_in_val_finds_the_entity_the_train_side_stored_under_an_int():
    """Train ids are ints, val ids the same entities as floats (3 vs 3.0): the val rows get the entity encoding, not the prior, and equal the replay's."""
    rng = np.random.default_rng(0)
    n_train, n_val = 300, 60
    train_ids = rng.integers(0, 6, n_train)
    val_ids = rng.integers(0, 6, n_val)
    group_ids = np.array(list(train_ids) + [float(v) for v in val_ids], dtype=object)
    y = (train_ids * 2.0 + rng.normal(0, 0.1, n_train)).astype(float)
    ts = np.arange(n_train + n_val, dtype=float)
    df = pd.DataFrame({"cat_col": rng.choice(list("AB"), n_train + n_val)})
    cfg = PreprocessingExtensionsConfig(two_step_target_encode_columns=["cat_col"])
    metadata: dict = {}
    _train, val, _ = apply_target_encoding_composite_fe(
        df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True), None, cfg,
        group_ids, ts, y, np.arange(n_train), np.arange(n_train, n_train + n_val), None, metadata=metadata,
    )
    out_col = metadata["two_step_target_encode_out_col"]
    prior = metadata["two_step_target_encode_global_prior"]
    assert not np.isclose(val[out_col].to_numpy(), prior).all(), "every val row fell back to the prior: the float ids missed the int-keyed lookup"
    replayed = replay_target_encoding_composite_fe(df.iloc[n_train:].reset_index(drop=True), metadata, np.array(val_ids))
    np.testing.assert_allclose(val[out_col].to_numpy(), replayed[out_col].to_numpy())
