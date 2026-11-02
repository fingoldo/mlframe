"""A numeric ``invalid_row_fill`` is the meta-feature of rows a group does not cover, not just the fallback for an all-invalid group."""

from __future__ import annotations

from sklearn.linear_model import LinearRegression

from mlframe.training.composite import GroupedBlockStacker

from .test_biz_val_grouped_block_stacking import _make_grouped_block_dataset


def _fitted(fill):
    """Fit the stacker on a block dataset with the given invalid_row_fill."""
    X, y, feature_groups = _make_grouped_block_dataset(n=800, n_groups=3, cols_per_group=3, seed=2)
    return GroupedBlockStacker(
        feature_groups=feature_groups,
        submodel_factory=lambda: LinearRegression(),
        meta_estimator=LinearRegression(),
        invalid_row_fill=fill,
        n_splits=3,
        random_state=0,
    ).fit(X, y)


def test_numeric_fill_is_used_for_every_group_even_when_valid_rows_exist():
    """With invalid_row_fill=-7.0 each group's fill value is -7.0, not the group's OOF mean."""
    stacker = _fitted(-7.0)
    assert set(stacker.group_fill_values_.values()) == {-7.0}


def test_group_mean_fill_stays_the_oof_mean():
    """The default keeps filling with the group's OOF mean (negative control)."""
    stacker = _fitted("group_mean")
    assert all(v != -7.0 for v in stacker.group_fill_values_.values())
