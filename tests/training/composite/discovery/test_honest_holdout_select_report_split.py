"""The rows a gate selects on must not be the rows the honest number is reported from.

The carve promises a holdout no decision ever saw, but the drop gate, the honest-OOF ranking key and the cross-target
budget all read it, and the gain stamped on each survivor is then measured on those same rows. That makes the reported
number a maximum over survivors of a comparison made on the very rows it reports -- the winner's curse the carve exists
to remove, and it grows as the gates reject more candidates. The holdout is therefore halved: one half decides, the
other reports.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlframe.training.composite.discovery._honest_holdout import carve_screening_holdout, split_holdout_select_report


def test_the_two_halves_are_disjoint_and_cover_the_holdout():
    """Every holdout row goes to exactly one role, so neither half is scored on the other's rows."""
    holdout = np.arange(500, 900)
    select, report = split_holdout_select_report(holdout, random_state=0)
    assert np.intersect1d(select, report).size == 0, "a row used to select must not also carry the reported number"
    np.testing.assert_array_equal(np.union1d(select, report), holdout)
    assert min(select.size, report.size) >= 50


def test_the_split_is_deterministic_for_a_seed():
    """Two carves with the same seed agree, so a rerun reports the same honest number."""
    holdout = np.arange(1000, 1400)
    a_sel, a_rep = split_holdout_select_report(holdout, random_state=7)
    b_sel, b_rep = split_holdout_select_report(holdout, random_state=7)
    np.testing.assert_array_equal(a_sel, b_sel)
    np.testing.assert_array_equal(a_rep, b_rep)
    assert not np.array_equal(a_sel, split_holdout_select_report(holdout, random_state=8)[0])


def test_a_holdout_too_small_to_halve_keeps_both_roles_on_all_rows():
    """Below two usable halves there is nothing to split; both roles keep the whole holdout, as before."""
    holdout = np.arange(60)
    select, report = split_holdout_select_report(holdout, random_state=0)
    np.testing.assert_array_equal(select, holdout)
    np.testing.assert_array_equal(report, holdout)


def test_no_holdout_leaves_both_roles_unset():
    """With the feature disabled there is no holdout at all and neither role gets rows."""
    assert split_holdout_select_report(None, random_state=0) == (None, None)


def test_the_carve_publishes_both_halves_on_the_instance():
    """``carve_screening_holdout`` stamps the two halves so the gates and the re-score can each read their own."""
    disc = SimpleNamespace(config=SimpleNamespace(honest_holdout_frac=0.3, random_state=0))
    train_idx = np.arange(2000)
    screen_idx, holdout_idx = carve_screening_holdout(disc, train_idx)

    assert holdout_idx is not None and holdout_idx.size
    assert np.intersect1d(screen_idx, holdout_idx).size == 0, "the screening pool must stay disjoint from the holdout"
    select, report = disc.honest_holdout_select_idx_, disc.honest_holdout_report_idx_
    assert np.intersect1d(select, report).size == 0
    np.testing.assert_array_equal(np.union1d(select, report), np.sort(holdout_idx))
    assert np.intersect1d(screen_idx, report).size == 0, "the reported rows must be unseen by screening as well"


def test_the_exported_rmse_gain_is_measured_on_the_report_half():
    """The RMSE gate decides on the select half; the gain every consumer reads comes from the report half.

    The first fix moved only the MI re-score to the report half. ``honest_holdout_rmse_gain`` - the number the cross-target
    budget ranks by and the ship floor tests - was still stamped by the gate on the rows it had just used to decide. This
    spies on the gate: the select-half call must decide, a second call on the report rows must record, and the exported
    fields must be the recording call's, with the gate's own number kept apart as ``selection_holdout_rmse_gain``.
    """
    import warnings

    import pandas as pd

    from mlframe.training.composite import CompositeTargetDiscovery
    from mlframe.training.composite.discovery import _honest_rmse_gate as gate_mod
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    rng = np.random.default_rng(0)
    n = 2000
    base = rng.uniform(50.0, 150.0, n)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame({"base": base, "x1": x1, "x2": x2, "y": 0.9 * base + 4.0 * np.sin(x1) + 2.0 * x2 + rng.normal(0.0, 1.0, n)})

    calls: list[dict] = []
    real_gate = gate_mod.apply_honest_rmse_gate

    def _spy(self, df_, target_col, kept_specs, usable_features, screen_idx, holdout_idx, y_full, **kwargs):
        """Run the real gate, then note which rows it read, in which mode, and what it stamped."""
        out = real_gate(self, df_, target_col, kept_specs, usable_features, screen_idx, holdout_idx, y_full, **kwargs)
        calls.append({"rows": np.asarray(holdout_idx), "record_only": bool(kwargs.get("record_only", False)),
                      "gains": {s.name: s.honest_holdout_rmse_gain for s in out}})
        return out

    cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, base_candidates=["base"], transforms=["linear_residual", "diff"])
    disc = CompositeTargetDiscovery(cfg)
    import unittest.mock as mock

    with mock.patch.object(gate_mod, "apply_honest_rmse_gate", _spy), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disc.fit(df, "y", ["base", "x1", "x2"], np.arange(n))

    select, report = disc.honest_holdout_select_idx_, disc.honest_holdout_report_idx_
    assert np.intersect1d(select, report).size == 0
    decides = [c for c in calls if not c["record_only"]]
    records = [c for c in calls if c["record_only"]]
    assert len(decides) == 1, f"expected one deciding call on the select half, got {len(decides)}"
    assert len(records) == 1, "no call re-scored the survivors on the report half, so the exported gain is the gate's own"
    decide, record = decides[0], records[0]
    np.testing.assert_array_equal(decide["rows"], select)
    np.testing.assert_array_equal(record["rows"], report)

    specs = disc.export_specs()
    assert specs, "the fixture must ship at least one spec for the assertion to mean anything"
    for s in specs:
        assert s["honest_holdout_rmse_gain"] == record["gains"][s["name"]], "the exported gain is not the report half's"
        assert s["selection_holdout_rmse_gain"] == decide["gains"][s["name"]], "the gate's own number was lost"
        assert s["honest_holdout_rmse_gain"] != s["selection_holdout_rmse_gain"], "two different row sets gave the same number"
