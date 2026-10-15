"""The mix-offset hook adds a ``__mixoff`` companion target only when given inputs and when the gate accepts a base."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mlframe.feature_selection.filters.mrmr import MRMR
from mlframe.training.configs import TargetTypes
from mlframe.training.core._phase_composite_discovery import _add_mix_offset_targets


def _inputs(seed, n=1600):
    """A frame, a target with a strong linear term in ``u``, and the selector and gate used by the hook."""
    rng = np.random.default_rng(seed)
    u, x2, x3, nz = (rng.standard_normal(n) for _ in range(4))
    y = 3.0 * u + x2 * x3 + 0.1 * rng.standard_normal(n)
    X = pd.DataFrame({"u": u, "x2": x2, "x3": x3, "nz": nz})
    return X, y, {
        "X": X,
        "train_idx": np.arange(1000),
        "selector": MRMR(verbose=0, n_jobs=1, random_state=seed, fe_target_mix_enable=True, fe_target_mix_max_bases=1, fe_max_steps=1),
        "gate_regressor": HistGradientBoostingRegressor(max_iter=100, random_state=0),
    }


def test_accepted_base_adds_a_companion_target_with_metadata():
    """A linear base the gate accepts yields ``<name>__mixoff`` on every row and a metadata record of its beta."""
    _X, y, inputs = _inputs(0)
    tbt = {TargetTypes.REGRESSION: {"t": y}}
    meta = {}
    out = _add_mix_offset_targets(tbt, meta, inputs)
    assert "t__mixoff" in out[TargetTypes.REGRESSION]
    assert out[TargetTypes.REGRESSION]["t__mixoff"].shape == y.shape
    rec = meta["mix_offset_targets"][str(TargetTypes.REGRESSION)]["t__mixoff"]
    assert rec["source"] == "t" and abs(rec["beta"] - 3.0) < 0.3


def test_noise_target_gets_no_companion():
    """Pure noise against the frame gives no companion and no metadata."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.standard_normal(1600), "b": rng.standard_normal(1600)})
    y = rng.standard_normal(1600)
    inputs = {"X": X, "train_idx": np.arange(1000), "selector": MRMR(verbose=0, n_jobs=1, random_state=1, fe_target_mix_enable=True, fe_target_mix_max_bases=1, fe_max_steps=1), "gate_regressor": HistGradientBoostingRegressor(max_iter=50, random_state=0)}
    meta = {}
    out = _add_mix_offset_targets({TargetTypes.REGRESSION: {"t": y}}, meta, inputs)
    assert list(out[TargetTypes.REGRESSION]) == ["t"]
    assert "mix_offset_targets" not in meta


def test_predict_finds_the_record_only_for_its_companion_target():
    """The predict-side lookup returns the record of ``<name>__mixoff`` under its target type and nothing for any other name."""
    from mlframe.feature_selection.filters._target_mix_target import mix_offset_record as _mix_offset_record

    rec = {"source": "t", "beta": 3.0, "recipe": None}
    meta = {"mix_offset_targets": {str(TargetTypes.REGRESSION): {"t__mixoff": rec}}}
    assert _mix_offset_record(meta, TargetTypes.REGRESSION, "t__mixoff") is rec
    assert _mix_offset_record(meta, TargetTypes.REGRESSION, "t") is None
    assert _mix_offset_record({}, TargetTypes.REGRESSION, "t__mixoff") is None
    assert _mix_offset_record(None, TargetTypes.REGRESSION, "t__mixoff") is None


def test_flag_defaults_off_and_the_inputs_helper_returns_none_when_off():
    """The config flag is True by default; the helper returns None only when it is switched off."""
    from mlframe.training._composite_target_discovery_config import CompositeTargetDiscoveryConfig
    from mlframe.training.core._main_train_suite_phases import _mix_offset_inputs_for_discovery

    assert CompositeTargetDiscoveryConfig().mix_offset_enabled is True
    df = pd.DataFrame({"a": [1.0, 2.0]})
    assert _mix_offset_inputs_for_discovery(CompositeTargetDiscoveryConfig(mix_offset_enabled=False), df, None, None, np.array([0, 1]), np.array([], dtype=np.int64), np.array([], dtype=np.int64)) is None


def test_flag_on_with_a_partition_returns_the_hook_inputs():
    """With the flag set and the splits covering the table, the helper returns the full frame, the training rows and a selector."""
    from mlframe.training._composite_target_discovery_config import CompositeTargetDiscoveryConfig
    from mlframe.training.core._main_train_suite_phases import _mix_offset_inputs_for_discovery

    cfg = CompositeTargetDiscoveryConfig(mix_offset_enabled=True)
    full = pd.DataFrame({"a": np.arange(6.0)})
    tr, va, te = np.array([0, 2, 4]), np.array([1]), np.array([3, 5])
    out = _mix_offset_inputs_for_discovery(cfg, full.iloc[tr], full.iloc[va], full.iloc[te], tr, va, te)
    assert out is not None
    assert np.array_equal(out["X"]["a"].to_numpy(), full["a"].to_numpy())
    assert np.array_equal(out["train_idx"], tr)
    assert out["selector"].fe_target_mix_enable is True
