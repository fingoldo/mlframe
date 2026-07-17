"""Unit + biz_value for the AR-failsafe (lag_predict) floor in honest group-OOF selection.

Honest-OOF selection ranks specs by their production predict-T -> invert-to-y reconstruction RMSE on the group-disjoint
holdout and floors them against the raw-y honest-OOF baseline. On a strong-AR sequential target that floor is WRONG: the
model actually deployed is the ``lag_predict`` failsafe (``y_hat = y_prev``), which can be dramatically better than the
raw-y model on the out-of-range holdout wells. A composite spec that beats the (weak) raw-y model but loses to the lag
failsafe is worthless -- yet the raw-only floor keeps it (prod incident: ensemble 13.30 vs lag_predict 11.58 floor).

This module pins the fix: ``honest_oof_reconstruction_rmse`` records ``self._honest_oof_lag_rmse`` (the AR-failsafe RMSE
on the SAME holdout rows) and the selector floors against ``min(raw, lag)``. The lag column is detected by provenance
(name ``{target}_prev`` / ``_lag_1`` / ..., the SAME probe ``_dummy_baseline_regression`` uses), NEVER by a marginal
correlation match, so a contemporaneous near-copy of y that is not a lag is not exempted.

Cost: one narrow column gather + an O(n) RMSE on the already-bounded (<=30k) holdout sample, negligible next to the
per-spec tiny-model fits honest-OOF already runs -- a trivial helper, no dispatcher / njit warranted.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite import CompositeSpec, CompositeTargetDiscovery
from mlframe.training.composite.discovery._causal_lag import (
    CAUSAL_LAG_SUFFIXES,
    causal_lag_predict_rmse,
    detect_causal_lag_column,
)
from mlframe.training.composite.discovery._honest_oof_select import honest_oof_reconstruction_rmse
from mlframe.training.configs import CompositeTargetDiscoveryConfig

# Raw-y tiny model sees ONLY x1 (uninformative): it cannot recover the per-well level, so on the upper-tail holdout it
# predicts ~global mean and is WEAK. The level-carrying columns are bases / the lag, never plain features -- exactly the
# prod setup where the causal lag is dropped as a "leakage candidate" and the raw model is left blind to the AR signal.
_FEATS = ["x1"]


def _ar_frame(n_groups: int = 40, per: int = 80, seed: int = 1, phi: float = 0.9, noise: float = 1.0):
    """Group-sequential AR(1) target. Each well hovers (mean-reverting) around its own level; ``y_prev`` is the exact
    causal lag so ``lag_predict`` is near-perfect (RMSE ~ noise). ``base_partial`` ~= 0.5*level (a partial level proxy):
    a ``diff`` spec on it beats the blind raw-y model but its reconstruction on the out-of-range upper-tail wells still
    loses badly to the lag failsafe."""
    rng = np.random.default_rng(seed)
    level = rng.uniform(0.0, 50.0, n_groups)
    groups = np.repeat(np.arange(n_groups), per)
    y = np.empty(groups.size, dtype=np.float64)
    y_prev = np.empty(groups.size, dtype=np.float64)
    idx = 0
    for g in range(n_groups):
        prev = float(level[g])
        for _ in range(per):
            y_prev[idx] = prev
            cur = level[g] + phi * (prev - level[g]) + rng.normal(0.0, noise)
            y[idx] = cur
            prev = cur
            idx += 1
    x1 = rng.normal(size=groups.size)
    base_partial = 0.5 * level[groups] + rng.normal(0.0, 0.2, groups.size)
    base_full = level[groups] + rng.normal(0.0, 0.2, groups.size)
    df = pd.DataFrame({"x1": x1, "base_partial": base_partial, "base_full": base_full, "y_prev": y_prev, "y": y.astype(np.float64)})
    return df, groups.astype(np.int64), y.astype(np.float64), level


def _split_upper_tail(groups, level, n_holdout_wells: int = 8):
    holdout_wells = set(np.argsort(level)[-n_holdout_wells:].tolist())
    hmask = np.array([g in holdout_wells for g in groups])
    return np.nonzero(~hmask)[0], np.nonzero(hmask)[0]


def _spec(name, transform_name, base_column, params):
    return CompositeSpec(
        name=name,
        target_col="y",
        transform_name=transform_name,
        base_column=base_column,
        fitted_params=dict(params),
        mi_gain=1.0,
        mi_y=0.0,
        mi_t=1.0,
        valid_domain_frac=1.0,
        n_train_rows=100,
    )


def _disc(groups, holdout_idx, **cfg_kw):
    cfg = CompositeTargetDiscoveryConfig(
        enabled=True,
        random_state=0,
        tiny_model_n_estimators=40,
        yscale_holdout_gate_sample_n=30_000,
        **cfg_kw,
    )
    disc = CompositeTargetDiscovery(cfg)
    disc._group_ids_for_rerank = groups
    disc.honest_holdout_idx_ = holdout_idx
    return disc


@pytest.fixture(autouse=True)
def _silence_lgbm_feature_name_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        yield


class TestCausalLagHelperUnit:
    def test_detect_finds_prev_suffix(self):
        df = pd.DataFrame({"TVT": [1.0], "TVT_prev": [0.0], "x": [0.0]})
        assert detect_causal_lag_column(df, "TVT") == "TVT_prev"

    def test_detect_returns_none_when_absent(self):
        df = pd.DataFrame({"TVT": [1.0], "x": [0.0]})
        assert detect_causal_lag_column(df, "TVT") is None

    def test_detect_respects_suffix_priority(self):
        # _prev is first in CAUSAL_LAG_SUFFIXES, so it wins over _lag when both are present.
        assert CAUSAL_LAG_SUFFIXES[0] == "_prev"
        df = pd.DataFrame({"y": [1.0], "y_prev": [0.0], "y_lag": [9.0]})
        assert detect_causal_lag_column(df, "y") == "y_prev"

    def test_detect_polars_schema(self):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"y": [1.0, 2.0], "y_lag_1": [0.0, 1.0]})
        assert detect_causal_lag_column(df, "y") == "y_lag_1"

    def test_detect_empty_target(self):
        df = pd.DataFrame({"y_prev": [0.0]})
        assert detect_causal_lag_column(df, "") is None

    def test_lag_rmse_over_finite_rows(self):
        y = np.array([1.0, 2.0, 3.0, 4.0] * 20, dtype=np.float64)
        lag = y + 0.5
        assert math.isclose(causal_lag_predict_rmse(lag, y), 0.5, rel_tol=1e-9)

    def test_lag_rmse_ignores_nonfinite(self):
        y = np.arange(100, dtype=np.float64)
        lag = y.copy()
        lag[0] = np.nan
        lag[1] = np.inf
        assert causal_lag_predict_rmse(lag, y) == 0.0  # remaining 98 rows are exact

    def test_lag_rmse_nan_when_too_few_rows(self):
        assert math.isnan(causal_lag_predict_rmse(np.zeros(10), np.ones(10)))

    def test_lag_rmse_nan_on_shape_mismatch(self):
        assert math.isnan(causal_lag_predict_rmse(np.zeros(60), np.ones(50)))


class TestHonestOofLagFloor:
    def test_lag_rmse_recorded_when_lag_column_present(self):
        df, groups, y, level = _ar_frame()
        screen_idx, holdout_idx = _split_upper_tail(groups, level)
        disc = _disc(groups, holdout_idx)
        spec = _spec("y-diff-partial", "diff", "base_partial", {})
        honest_oof_reconstruction_rmse(disc, df, "y", [spec], _FEATS, screen_idx, holdout_idx, y)
        assert math.isfinite(disc._honest_oof_lag_rmse), "lag floor must be measured when a lag column exists"
        assert math.isfinite(disc._honest_oof_raw_rmse)

    def test_lag_rmse_nan_when_no_lag_column(self):
        df, groups, y, level = _ar_frame()
        df = df.drop(columns=["y_prev"])  # remove the causal lag -> non-AR path, floor reduces to raw
        screen_idx, holdout_idx = _split_upper_tail(groups, level)
        disc = _disc(groups, holdout_idx)
        spec = _spec("y-diff-partial", "diff", "base_partial", {})
        honest_oof_reconstruction_rmse(disc, df, "y", [spec], _FEATS, screen_idx, holdout_idx, y)
        assert math.isnan(disc._honest_oof_lag_rmse), "no lag column -> no lag floor"

    def test_biz_val_lag_floor_buries_spec_that_beats_raw_but_loses_to_lag(self):
        """The core contract. On the strong-AR synthetic the ordering is lag << spec << raw:
        * raw-y honest-OOF is WEAK (blind to level, predicts ~mean on out-of-range wells) -- measured ~20+;
        * the ``diff`` spec beats raw but its reconstruction on unseen wells is moderate -- measured ~10-13;
        * the AR failsafe (lag_predict) is near-perfect -- measured ~1.
        With the raw-only floor the spec passes (spec < raw*1.05); with the min(raw, lag) floor it is buried
        (spec >> lag*1.05). Asserts the exact 13-vs-11.58-class incident is now caught."""
        df, groups, y, level = _ar_frame()
        screen_idx, holdout_idx = _split_upper_tail(groups, level)
        disc = _disc(groups, holdout_idx)
        spec = _spec("y-diff-partial", "diff", "base_partial", {})
        res = honest_oof_reconstruction_rmse(disc, df, "y", [spec], _FEATS, screen_idx, holdout_idx, y)

        raw = disc._honest_oof_raw_rmse
        lag = disc._honest_oof_lag_rmse
        spec_recon = res[spec.name]
        # The three quantities genuinely separate: lag is the tightest, raw the loosest, spec in between.
        assert lag < 3.0, f"AR failsafe must be near-perfect on this synthetic; got {lag:.3f}"
        assert raw > 8.0, f"raw-y model must be weak (blind to level); got {raw:.3f}"
        assert lag * 2.0 < spec_recon < raw, f"spec must beat raw but lose to lag; lag={lag:.3f} spec={spec_recon:.3f} raw={raw:.3f}"

        tol = 1.05
        floor_raw_only = raw
        floor_with_lag = min(raw, lag)
        # Raw-only floor would KEEP the spec (the bug); the lag floor REJECTS it (the fix).
        assert spec_recon < floor_raw_only * tol, "raw-only floor keeps the spec (the pre-fix bug)"
        assert spec_recon >= floor_with_lag * tol, "min(raw, lag) floor buries the spec (the fix)"

    def test_biz_val_full_rerank_drops_spec_only_when_lag_present(self):
        """End-to-end through ``_tiny_model_rerank`` with the confounding structural / near-copy gates disabled so the
        ONLY difference between the two runs is the honest-OOF floor: the ``diff`` spec is REJECTED when the lag column
        is present (floor = lag) and RETAINED when it is absent (floor = raw)."""
        gates_off = dict(
            structural_fragility_gate_enabled=False,
            per_bin_n_bins=0,
            use_wilcoxon_gate=False,
        )
        df, groups, y, level = _ar_frame()
        screen_idx, holdout_idx = _split_upper_tail(groups, level)
        spec_a = _spec("y-diff-partial", "diff", "base_partial", {})

        disc_lag = _disc(groups, holdout_idx, **gates_off)
        kept_lag = disc_lag._tiny_model_rerank(
            kept_specs=[spec_a],
            df=df,
            target_col="y",
            usable_features=_FEATS,
            train_idx=screen_idx,
            y_full=y,
        )
        assert "y-diff-partial" not in [s.name for s in kept_lag], "lag floor must reject a spec that loses to lag_predict"

        df_nolag = df.drop(columns=["y_prev"])
        spec_b = _spec("y-diff-partial", "diff", "base_partial", {})
        disc_nolag = _disc(groups, holdout_idx, **gates_off)
        kept_nolag = disc_nolag._tiny_model_rerank(
            kept_specs=[spec_b],
            df=df_nolag,
            target_col="y",
            usable_features=_FEATS,
            train_idx=screen_idx,
            y_full=y,
        )
        assert "y-diff-partial" in [s.name for s in kept_nolag], "without a lag column the raw-only floor retains the spec"
