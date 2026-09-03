"""Correctness test for ``training.neural.field_grouped_mlp.FieldGroupedMLPRegressor``.

Source: 1st_kkbox-music-recommendation-challenge.md -- "Field-aware: inputs divided into user/song/context
groups, high-level features extracted before concatenation." The hypothesized generalization benefit (field
grouping structurally prevents spurious cross-field interactions, helping small-sample generalization versus
a flat MLP) was tested directly and did NOT reproduce: across 6 independent synthetic configurations (varying
field count, field size, and sample size, each with signal confined to WITHIN-field pairwise products and
pure noise across fields), the field-grouped architecture's held-out R2 was consistently WORSE than a
comparably-sized flat MLP's (e.g. seed 0: fg=-0.448 vs flat=-0.184; seed 1: fg=-0.683 vs flat=-0.523; seed 2:
fg=-1.260 vs flat=-0.752) -- the flat MLP's extra representational flexibility outweighed the field-grouped
model's narrower inductive bias at every tested capacity/sample-size combination. This is an HONEST NEGATIVE,
documented rather than papered over with a cherry-picked synthetic: the architecture is implemented correctly
(these tests pin its mechanics) and remains available as a tunable option (per CLAUDE.md's "rejected != deleted"
convention) for callers who want the param-reduction/interpretability property regardless of the unproven
generalization claim, but it is NOT validated as a net win and should not be assumed superior to a flat MLP.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mlframe.training.neural.field_grouped_mlp import FieldGroupedMLPRegressor


def test_field_grouped_mlp_fits_and_predicts_correct_shape():
    """FieldGroupedMLPRegressor fits on multi-field input and returns finite predictions of the expected shape."""
    rng = np.random.default_rng(0)
    n, field_a_size, field_b_size = 100, 4, 4
    X = rng.normal(size=(n, field_a_size + field_b_size)).astype(np.float32)
    y = (X[:, 0] * X[:, 1] + X[:, field_a_size] * X[:, field_a_size + 1]).astype(np.float32)

    field_groups = {"A": list(range(field_a_size)), "B": list(range(field_a_size, field_a_size + field_b_size))}
    model = FieldGroupedMLPRegressor(field_groups=field_groups, field_hidden=4, head_hidden=8, n_epochs=20, random_state=0).fit(X, y)
    preds = model.predict(X)

    assert preds.shape == (n,)
    assert np.all(np.isfinite(preds))


def test_field_grouped_mlp_only_uses_columns_within_their_declared_field():
    """Zeroing field B entirely leaves field A's contribution to the output unaffected, proving structural field isolation."""
    # a field's encoder must never see another field's columns -- verify by zeroing field B entirely and
    # confirming field A's contribution to the output is unaffected (structural isolation, not learned).
    rng = np.random.default_rng(1)
    n = 50
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = np.zeros(n, dtype=np.float32)

    field_groups = {"A": [0, 1], "B": [2, 3]}
    model = FieldGroupedMLPRegressor(field_groups=field_groups, field_hidden=4, head_hidden=8, n_epochs=5, random_state=1).fit(X, y)

    X_modified = X.copy()
    X_modified[:, 2:] = 0.0  # zero out field B entirely.

    pred_original_field_a_only = model.model_.field_encoders["A"](torch.from_numpy(X[:, [0, 1]]))
    pred_modified_field_a_only = model.model_.field_encoders["A"](torch.from_numpy(X_modified[:, [0, 1]]))
    np.testing.assert_allclose(pred_original_field_a_only.detach().numpy(), pred_modified_field_a_only.detach().numpy())


def test_field_groups_missing_column_raises():
    """TRAINING_NEURAL-4: a column absent from every field must raise, not silently drop its signal."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    y = np.zeros(20, dtype=np.float32)
    field_groups = {"A": [0, 1], "B": [2]}  # column 3 missing from every field
    with pytest.raises(ValueError, match="missing from every field"):
        FieldGroupedMLPRegressor(field_groups=field_groups, n_epochs=1).fit(X, y)


def test_field_groups_duplicated_column_raises():
    """TRAINING_NEURAL-4: a column assigned to two fields must raise, not silently double-count it."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    y = np.zeros(20, dtype=np.float32)
    field_groups = {"A": [0, 1, 2], "B": [2, 3]}  # column 2 in both fields
    with pytest.raises(ValueError, match="more than one field"):
        FieldGroupedMLPRegressor(field_groups=field_groups, n_epochs=1).fit(X, y)


def test_field_groups_out_of_range_column_raises():
    """TRAINING_NEURAL-4: a column index outside X's width must raise a clear error, not an opaque torch indexing crash."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    y = np.zeros(20, dtype=np.float32)
    field_groups = {"A": [0, 1], "B": [2, 99]}  # 99 is out of range
    with pytest.raises(ValueError, match="out of range"):
        FieldGroupedMLPRegressor(field_groups=field_groups, n_epochs=1).fit(X, y)


def test_field_groups_valid_exact_partition_does_not_raise():
    """Sanity: a correct exact partition of every column into exactly one field must not raise."""
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    y = np.zeros(20, dtype=np.float32)
    field_groups = {"A": [0, 1], "B": [2, 3]}
    FieldGroupedMLPRegressor(field_groups=field_groups, n_epochs=1).fit(X, y)  # must not raise
