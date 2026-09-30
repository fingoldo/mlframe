"""Invariants of narrowing the suite's split to one target's labelled rows."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mlframe.training.core._target_labels import label_mask
from mlframe.training.core._target_rows import SPLIT_INDEX_FIELDS, build_target_rows, mask_signature


@st.composite
def _mask_and_splits(draw):
    """A label mask (all, none, or random) over n rows and a split of them, with OD dropping some train/val rows."""
    n = draw(st.integers(min_value=1, max_value=200))
    kind = draw(st.sampled_from(["all", "none", "random"]))
    if kind == "random":
        mask = np.array(draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype=bool)
    else:
        mask = np.full(n, kind == "all")
    rng = np.random.default_rng(draw(st.integers(0, 2**31 - 1)))
    parts = np.array_split(rng.permutation(n), 4)
    train, val, test, calib = (np.sort(p) for p in parts)
    splits = {"train_idx": train, "val_idx": val, "test_idx": test}
    if draw(st.booleans()):
        splits["calib_idx"] = calib
    if draw(st.booleans()):  # outlier detection kept a subset of train and val, in order
        splits["filtered_train_idx"] = train[rng.random(train.size) < 0.8]
        splits["filtered_val_idx"] = val[rng.random(val.size) < 0.8]
    return mask, splits


@settings(max_examples=300, deadline=None)
@given(_mask_and_splits())
def test_narrowed_splits_keep_exactly_the_labelled_rows_in_order(case):
    mask, splits = case
    rows = build_target_rows(mask, splits)
    for name in SPLIT_INDEX_FIELDS:
        if name not in splits:
            assert name not in rows.idx and name not in rows.pos
            continue
        split_idx = splits[name]
        np.testing.assert_array_equal(split_idx[rows.pos[name]], rows.idx[name])
        assert mask[rows.idx[name]].all()
        np.testing.assert_array_equal(rows.idx[name], split_idx[mask[split_idx]])
        assert np.all(np.diff(rows.pos[name]) > 0)
        assert rows.n_labelled[name] == int(mask[split_idx].sum()) and rows.n_total[name] == split_idx.size
    if "filtered_train_idx" in splits:
        assert np.isin(rows.idx["filtered_train_idx"], rows.idx["train_idx"]).all()


def test_equal_masks_share_one_narrowing_and_fully_labelled_targets_get_none():
    y_a = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    y_c = np.array([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0])
    splits = {"train_idx": np.array([0, 1, 2, 3]), "val_idx": np.array([4]), "test_idx": np.array([5])}
    masks = {name: label_mask(values) for name, values in {"a": y_a, "b": y_a * 10, "c": y_c, "full": np.arange(6.0)}.items()}
    assert masks["full"] is None
    assert mask_signature(masks["a"]) == mask_signature(masks["b"])
    assert mask_signature(masks["a"]) != mask_signature(masks["c"])
    rows_a = build_target_rows(masks["a"], splits)
    np.testing.assert_array_equal(rows_a.idx["train_idx"], [0, 2, 3])
    assert rows_a.idx["val_idx"].size == 0 and rows_a.labelled_share("val_idx") == 0.0
    assert rows_a.labelled_share("calib_idx") is None


@pytest.mark.parametrize("n", [7, 8, 9])
def test_the_signature_tells_apart_masks_that_pack_to_the_same_bytes(n):
    """``packbits`` pads to whole bytes, so a mask of 7 falses and one of 8 falses pack alike; the length must count."""
    assert mask_signature(np.zeros(n, bool)) != mask_signature(np.zeros(n + 1, bool))
    assert mask_signature(np.zeros(n, bool)) == mask_signature(np.zeros(n, bool))
