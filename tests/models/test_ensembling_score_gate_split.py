"""Wave 12a monolith-split sensor for ``mlframe.models.ensembling.score``.

Carve pattern: three pure helpers extracted to ``_ensembling_score_gate``:
- ``select_gate_source_split`` (OOF vs val/test/train preference + coarse-gate threshold flip)
- ``catastrophic_drop_kn`` (K>2 absolute-MAE catastrophic drop)
- ``catastrophic_drop_k2`` (K=2 single-member dropout w/ early-return signal)

All three are byte-equivalent to their pre-carve inline blocks: same branches, same logger.warning / .info calls, same ``res`` dict stamping, same threshold mutations returned via tuple unpacking. Existing 200+ ensembling tests remain GREEN.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def parent_module():
    from mlframe.models.ensembling import score as _ensembling_score

    return _ensembling_score


@pytest.fixture(scope="module")
def gate_sibling():
    from mlframe.models.ensembling import score_gate as _ensembling_score_gate

    return _ensembling_score_gate


def test_gate_helpers_resolve(gate_sibling):
    assert hasattr(gate_sibling, "select_gate_source_split")
    assert hasattr(gate_sibling, "catastrophic_drop_kn")
    assert hasattr(gate_sibling, "catastrophic_drop_k2")


def test_facade_loc_budget(parent_module):
    path = Path(parent_module.__file__)
    n_lines = len(path.read_text(encoding="utf-8").splitlines())
    assert n_lines < 800, f"facade is {n_lines} LOC, expected < 800 after Wave 12 gate carve"


def test_score_ensemble_still_callable(parent_module):
    assert hasattr(parent_module, "score_ensemble") and callable(parent_module.score_ensemble)


def test_select_gate_source_split_prefers_oof(gate_sibling):
    """When members carry OOF preds, the helper picks 'oof' regardless of fallback config."""
    n = 50
    rng = np.random.default_rng(0)
    members = [
        SimpleNamespace(
            oof_preds=rng.standard_normal(n),
            val_preds=rng.standard_normal(n),
            test_preds=rng.standard_normal(n),
            train_preds=rng.standard_normal(n),
        )
        for _ in range(3)
    ]
    preds, label, coarse, mae, std, mae_r, std_r = gate_sibling.select_gate_source_split(
        level_models_and_predictions=members,
        require_oof_for_gate=True,
        coarse_gate_max_mae_relative=5.0,
        coarse_gate_max_std_relative=5.0,
        max_mae=0.0,
        max_std=0.0,
        max_mae_relative=2.5,
        max_std_relative=2.5,
        verbose=False,
    )
    assert label == "oof"
    assert preds is not None and len(preds) == 3
    assert coarse is False
    assert mae == 0.0 and std == 0.0
    assert mae_r == 2.5 and std_r == 2.5


def test_select_gate_source_split_coarse_fallback_flips_thresholds(gate_sibling):
    """No OOF + require_oof_for_gate=True -> coarse path picks val_preds + flips thresholds."""
    n = 50
    rng = np.random.default_rng(0)
    members = [SimpleNamespace(val_preds=rng.standard_normal(n), test_preds=None, train_preds=None) for _ in range(3)]
    preds, label, coarse, mae, std, mae_r, std_r = gate_sibling.select_gate_source_split(
        level_models_and_predictions=members,
        require_oof_for_gate=True,
        coarse_gate_max_mae_relative=4.0,
        coarse_gate_max_std_relative=3.5,
        max_mae=0.05,
        max_std=0.06,
        max_mae_relative=2.5,
        max_std_relative=2.5,
        verbose=False,
    )
    assert label == "val-coarse"
    assert coarse is True
    assert mae == 0.0 and std == 0.0
    assert mae_r == 4.0 and std_r == 3.5


def test_catastrophic_drop_k2_early_return_path(gate_sibling):
    """K=2 with one catastrophic member returns ``early_return=True`` and stamps res sentinels."""
    n = 50
    rng = np.random.default_rng(1)
    target = rng.standard_normal(n)
    good = target + rng.standard_normal(n) * 0.05
    bad = target + 100.0
    members = [SimpleNamespace(), SimpleNamespace()]
    tags = ["good_model", "bad_model"]
    short_tags = ["g", "b"]
    res = {}
    out_lvl, out_tags, out_short, out_name, early = gate_sibling.catastrophic_drop_k2(
        level_models_and_predictions=members,
        _gate_preds_for_check=[good, bad],
        _gate_source_split="oof",
        _ensemble_member_tags=tags,
        _ensemble_short_tags=short_tags,
        ensemble_name="[g+b] ensemble",
        train_target_arr=target,
        val_target_arr=None,
        test_target_arr=None,
        k2_catastrophic_mae_ratio=20.0,
        verbose=False,
        res=res,
    )
    assert early is True
    assert res["_reason"] == "k2_catastrophic_dropout"
    assert res["_n_members"] == 1
    assert res["_dropped_member"] == "bad_model"
    assert res["_kept_member"] == "good_model"
    assert len(out_lvl) == 1
    assert out_tags == ["good_model"]
    assert out_short == ["g"]


def test_catastrophic_drop_kn_drops_outlier(gate_sibling):
    """K>2 with one catastrophic member slices it out and stamps ``_kn_catastrophic_dropped``."""
    n = 60
    rng = np.random.default_rng(2)
    target = rng.standard_normal(n)
    good_a = target + rng.standard_normal(n) * 0.02
    good_b = target + rng.standard_normal(n) * 0.04
    good_c = target + rng.standard_normal(n) * 0.06
    bad = target + 500.0
    preds = [good_a, good_b, good_c, bad]
    members = [SimpleNamespace() for _ in range(4)]
    tags = ["a", "b", "c", "bad"]
    short_tags = ["a", "b", "c", "bad"]
    res = {}
    out_lvl, out_preds, out_tags, out_short = gate_sibling.catastrophic_drop_kn(
        level_models_and_predictions=members,
        _gate_preds_for_check=preds,
        _gate_source_split="oof",
        _ensemble_member_tags=tags,
        _ensemble_short_tags=short_tags,
        train_target_arr=target,
        val_target_arr=None,
        test_target_arr=None,
        k2_catastrophic_mae_ratio=20.0,
        verbose=False,
        res=res,
    )
    assert "_kn_catastrophic_dropped" in res
    assert out_tags == ["a", "b", "c"]
    assert out_short == ["a", "b", "c"]
    assert len(out_lvl) == 3
    assert len(out_preds) == 3
