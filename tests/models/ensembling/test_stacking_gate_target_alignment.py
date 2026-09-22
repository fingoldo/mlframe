"""The blend-weight fit must be handed the target of the split its gate predictions come from.

`score_ensemble` passed the whole-frame `target_arr` while `_gate_preds_for_check` is an OOF/calib/val subset, so every
member failed the per-member length check, `_saw_preds` stayed empty, no `res["_stacking_gate"]` was stamped and the
advertised NNLS/Caruana blend was a uniform mean on every run - with nothing in the log saying so.
"""

from __future__ import annotations

import logging
import types

import numpy as np
import pytest

from mlframe.models.ensembling.score_flavours import run_stacking_aware_gate
from mlframe.models.ensembling.score_gate import resolve_gate_target_arr


def _bed(n_sub: int = 300):
    rng = np.random.default_rng(0)
    y = (rng.random(n_sub) < 0.4).astype(np.float64)
    good = np.clip(y * 0.8 + rng.normal(0, 0.1, n_sub) + 0.1, 0, 1)
    weak = np.clip(rng.random(n_sub), 0, 1)
    return y, [good, weak]


def _run(target, preds, **kw):
    res: dict = {}
    weights = run_stacking_aware_gate(
        enable_stacking_aware_gate=True,
        _gate_preds_for_check=preds,
        target_arr=target,
        level_models_and_predictions=[types.SimpleNamespace(), types.SimpleNamespace()],
        _ensemble_member_tags=["good", "weak"],
        stacking_gate_min_weight=0.01,
        use_nnls_weights=True,
        res=res,
        verbose=False,
        **kw,
    )
    return weights, res


def test_the_aligned_target_produces_weights():
    y, preds = _bed()
    weights, res = _run(y, preds)
    assert weights is not None, "the weight fit must run when the target matches the gate rows"
    assert res["_stacking_gate"]["applied_to_blend"] is True
    assert weights[0] > weights[1], "the informative member must outweigh the noise member"


def test_a_whole_frame_target_is_reported_instead_of_silently_disabling_the_fit(caplog):
    """The regression's own shape: gate rows are a subset of the frame. The fit still cannot run, but it now says so."""
    y, preds = _bed()
    full_frame_target = np.concatenate([y, y])
    with caplog.at_level(logging.WARNING, logger="mlframe.models.ensembling"):
        weights, res = _run(full_frame_target, preds)
    assert weights is None
    assert "blend-weight fit skipped" in caplog.text
    assert "_stacking_gate" not in res


@pytest.mark.parametrize(
    "split, expected",
    [("oof", "train"), ("train", "train"), ("train-coarse", "train"), ("val", "val"), ("val-coarse", "val"), ("test", "test"), ("test-coarse", "test")],
)
def test_every_gate_source_maps_to_its_own_split(split, expected):
    arrays = {k: np.full(3, float(i)) for i, k in enumerate(("train", "val", "test"))}
    got = resolve_gate_target_arr(
        split,
        train_target_arr=arrays["train"],
        val_target_arr=arrays["val"],
        test_target_arr=arrays["test"],
        level_models_and_predictions=[],
    )
    np.testing.assert_array_equal(got, arrays[expected])


def test_the_calib_source_reads_the_slice_stamped_on_the_members():
    member = types.SimpleNamespace(calib_target=np.array([1.0, 0.0, 1.0]))
    got = resolve_gate_target_arr("calib", level_models_and_predictions=[member])
    np.testing.assert_array_equal(got, member.calib_target)


def test_an_unknown_source_resolves_to_nothing_rather_than_to_the_wrong_rows():
    assert resolve_gate_target_arr(None, train_target_arr=np.zeros(3), level_models_and_predictions=[]) is None
