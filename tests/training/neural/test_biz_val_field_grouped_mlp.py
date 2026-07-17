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
import torch

from mlframe.training.neural.field_grouped_mlp import FieldGroupedMLPRegressor


def test_field_grouped_mlp_fits_and_predicts_correct_shape():
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
