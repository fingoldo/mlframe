"""Unit + biz_value for the MEASURED achievable-ceiling composite-discovery precheck (``core/_achievable_ceiling.py``).

The precheck replaces the crude ``lag1_autocorr >= 0.99`` heuristic with a MEASUREMENT on a bounded group-disjoint holdout:
raw-y tiny-model RMSE, lag_predict RMSE, and an OPTIMISTIC achievable-composite RMSE (best linear-residual base). When
even the optimistic composite cannot beat ``min(raw, lag)`` by a margin AND the failsafe already carries the target, the
verdict is SKIP.

Core pins:
* the ceiling measurement returns finite RMSEs + a well-formed verdict dict;
* SKIP on a strong-AR synthetic (lag near-perfect, composites can't beat it);
* PROCEED on a genuinely-composable synthetic (base explains level, residual learnable, lag weak);
* the FOOTGUN: ``extreme_ar_group_aware_skip=False`` must STILL yield SKIP on the strong-AR synthetic (the measured
  precheck is orthogonal to the legacy flag), pinned both at the precheck-function level and end-to-end through the phase;
* degenerate inputs (no lag column, no groups, tiny n) never crash and PROCEED (never a low-confidence skip).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.training.configs import CompositeTargetDiscoveryConfig, TargetTypes
from mlframe.training.core._achievable_ceiling import (
    _abs_corr,
    _ols_alpha_beta,
    measure_achievable_ceiling,
    run_achievable_ceiling_precheck,
)
from mlframe.training.core._phase_composite_discovery import run_composite_target_discovery

_VERDICT_KEYS = {
    "raw_rmse",
    "lag_rmse",
    "best_composite_rmse",
    "headroom_vs_min",
    "decision",
    "reason",
    "method",
}


@pytest.fixture(autouse=True)
def _silence_lgbm_feature_name_warning():
    """Silence lgbm feature name warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        warnings.filterwarnings("ignore", message=".*Grid size.*")
        yield


def _ar_frame(n_groups: int = 40, per: int = 200, seed: int = 1, phi: float = 0.95, noise: float = 1.0):
    """Group-sequential AR(1) target. Each well hovers (mean-reverting) around its own level; ``y_prev`` is the exact
    causal lag so ``lag_predict`` is near-perfect (RMSE ~ noise) and a raw tree fed ``y_prev`` is near-perfect too. A
    residual composite ``T = y - y_prev`` is pure noise -> the optimistic composite cannot beat the lag failsafe."""
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
    df = pd.DataFrame({"x1": rng.normal(size=groups.size), "y_prev": y_prev})
    return df, groups.astype(np.int64), y


def _composable_frame(n_groups: int = 30, per: int = 300, seed: int = 2):
    """``y = 40*base + resid(x1) + noise`` with a WIDE-range base and a WEAK noisy lag. A shallow booster underfits the
    wide linear ramp (staircase error), while the linear-residual composite captures ``40*base`` EXACTLY and learns the
    clean residual -> the optimistic composite crushes both the raw model and the (weak) lag failsafe."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per)
    base = rng.uniform(-200.0, 200.0, groups.size)
    x1 = rng.normal(size=groups.size)
    resid = 3.0 * np.sin(x1 * 2.0) + 0.5 * x1
    y = 40.0 * base + resid + rng.normal(0.0, 1.0, groups.size)
    y_prev = y + rng.normal(0.0, 50.0, groups.size)  # deliberately weak lag
    df = pd.DataFrame({"x1": x1, "base": base, "y_prev": y_prev})
    return df, groups.astype(np.int64), y


def _random_frame(n: int = 4000, seed: int = 3, n_groups: int = 20):
    """Random frame."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.standard_normal((n, 4)), columns=["a", "b", "c", "d"])
    y = rng.standard_normal(n)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    return df, groups.astype(np.int64), y


class TestAchievableCeilingUnit:
    """Groups tests covering achievable ceiling unit."""
    def test_verdict_schema_well_formed(self):
        """Verdict schema well formed."""
        df, g, y = _ar_frame()
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["x1", "y_prev"], y_train=y, group_ids_train=g)
        assert _VERDICT_KEYS.issubset(v.keys()), f"verdict missing keys: {_VERDICT_KEYS - set(v.keys())}"
        assert v["decision"] in ("proceed", "skip")
        assert v["method"] == "measured_achievable_ceiling"
        assert isinstance(v["reason"], str) and v["reason"]

    def test_measurement_returns_finite_rmses(self):
        """Measurement returns finite rmses."""
        df, g, y = _ar_frame()
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["x1", "y_prev"], y_train=y, group_ids_train=g)
        assert np.isfinite(v["raw_rmse"]), "raw baseline RMSE must be finite on a well-posed frame"
        assert np.isfinite(v["lag_rmse"]), "lag RMSE must be finite when a lag column exists"
        assert np.isfinite(v["best_composite_rmse"]), "optimistic composite RMSE must be finite here"

    def test_skip_when_composite_cannot_beat_floor(self):
        """Skip when composite cannot beat floor."""
        df, g, y = _ar_frame()
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["x1", "y_prev"], y_train=y, group_ids_train=g)
        assert v["decision"] == "skip", v["reason"]

    def test_proceed_when_composite_clearly_wins(self):
        """Proceed when composite clearly wins."""
        df, g, y = _composable_frame()
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["x1", "base", "y_prev"], y_train=y, group_ids_train=g)
        assert v["decision"] == "proceed", v["reason"]
        assert v["best_composite_rmse"] < v["floor_rmse"], "composite must beat the floor on a composable target"

    def test_degenerate_no_lag_column_proceeds_with_nan_lag(self):
        """Degenerate no lag column proceeds with nan lag."""
        df, g, y = _random_frame()
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["a", "b", "c", "d"], y_train=y, group_ids_train=g)
        assert np.isnan(v["lag_rmse"]), "no causal-lag column -> lag RMSE is NaN"
        assert v["decision"] == "proceed", "a weak/random floor must never trigger a low-confidence skip"

    def test_degenerate_no_groups_uses_random_split(self):
        """Degenerate no groups uses random split."""
        df, _, y = _random_frame()
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["a", "b", "c", "d"], y_train=y, group_ids_train=None)
        assert np.isfinite(v["raw_rmse"])
        assert v["decision"] == "proceed"

    def test_degenerate_tiny_n_proceeds(self):
        """Degenerate tiny n proceeds."""
        df, g, y = _random_frame(n=150, n_groups=5)
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["a", "b", "c", "d"], y_train=y, group_ids_train=g)
        assert v["decision"] == "proceed"
        assert "insufficient" in v["reason"].lower()

    def test_helpers_ols_and_corr(self):
        """Helpers ols and corr."""
        rng = np.random.default_rng(0)
        base = rng.normal(size=500)
        y = 3.0 * base + 7.0 + rng.normal(0.0, 1e-6, 500)
        alpha, beta = _ols_alpha_beta(base, y)
        assert abs(alpha - 3.0) < 1e-2 and abs(beta - 7.0) < 1e-2
        assert _abs_corr(base, y) > 0.999
        assert _abs_corr(base, np.zeros_like(base)) == 0.0  # constant -> 0


class TestAchievableCeilingBizValue:
    """Groups tests covering achievable ceiling biz value."""
    def test_biz_val_strong_ar_skips_with_quantified_headroom(self):
        """Strong-AR: raw+lag near-perfect, the optimistic composite cannot beat them -> SKIP. Quantified: the lag is
        near-perfect vs a large target spread, and the composite gets no material headroom over the floor."""
        df, g, y = _ar_frame()
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["x1", "y_prev"], y_train=y, group_ids_train=g)
        assert v["decision"] == "skip", v["reason"]
        # The failsafe is near-perfect relative to the target spread (measured headroom ~ -0.6%, floor/std ~ 0.07).
        assert v["floor_rmse"] < 0.2 * float(np.std(y)), "the AR failsafe must be a strong floor on this synthetic"
        assert v["headroom_vs_min"] < 0.02, f"composite must gain < margin over the floor; got {v['headroom_vs_min']:.3f}"
        assert v["lag_rmse"] < 3.0, f"lag_predict must be near-perfect; got {v['lag_rmse']:.3f}"

    def test_biz_val_strong_ar_skips_even_with_extreme_ar_flag_off(self):
        """FOOTGUN PIN. Setting ``extreme_ar_group_aware_skip=False`` disables ONLY the legacy autocorr heuristic; the
        MEASURED ceiling precheck is orthogonal and STILL returns a SKIP verdict on the strong-AR synthetic."""
        df, g, y = _ar_frame()
        cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, extreme_ar_group_aware_skip=False)
        v = run_achievable_ceiling_precheck(
            config=cfg,
            df=df,
            target_col="y",
            feature_cols=["x1", "y_prev"],
            y_train=y,
            group_ids_train=g,
        )
        assert v is not None, "the measured precheck must run regardless of extreme_ar_group_aware_skip"
        assert v["decision"] == "skip", f"extreme_ar_group_aware_skip=False must NOT disable the measured ceiling skip; got {v['reason']}"

    def test_biz_val_composable_proceeds_with_large_headroom(self):
        """Composable: the optimistic composite crushes both the raw model (staircase underfit of the wide linear base)
        and the weak lag. Quantified: headroom >= 0.5 (measured ~0.95), and the composite beats the floor by >5x."""
        df, g, y = _composable_frame()
        v = measure_achievable_ceiling(df=df, target_col="y", feature_cols=["x1", "base", "y_prev"], y_train=y, group_ids_train=g)
        assert v["decision"] == "proceed", v["reason"]
        assert v["headroom_vs_min"] >= 0.5, f"composable target must show large headroom; got {v['headroom_vs_min']:.3f}"
        assert v["best_composite_rmse"] * 5.0 < v["floor_rmse"], "optimistic composite must beat the floor by >5x here"
        assert v["best_base"] == "base", f"the wide-range level carrier must be the chosen base; got {v['best_base']}"


class TestAchievableCeilingConfigWiring:
    """Groups tests covering achievable ceiling config wiring."""
    def test_precheck_disabled_returns_none(self):
        """Precheck disabled returns none."""
        df, g, y = _ar_frame()
        cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, composite_achievable_ceiling_precheck=False)
        v = run_achievable_ceiling_precheck(
            config=cfg,
            df=df,
            target_col="y",
            feature_cols=["x1", "y_prev"],
            y_train=y,
            group_ids_train=g,
        )
        assert v is None, "composite_achievable_ceiling_precheck=False must disable the measured precheck"

    def test_precheck_margin_is_configurable(self):
        """A punishing margin (99%) forces a SKIP even on a target the default margin would PROCEED, proving the
        margin flows from config into the decision."""
        df, g, y = _composable_frame()
        cfg = CompositeTargetDiscoveryConfig(
            enabled=True,
            random_state=0,
            composite_achievable_ceiling_margin=0.99,
            composite_achievable_ceiling_strong_floor_frac=1.0,
        )
        v = run_achievable_ceiling_precheck(
            config=cfg,
            df=df,
            target_col="y",
            feature_cols=["x1", "base", "y_prev"],
            y_train=y,
            group_ids_train=g,
        )
        assert v["decision"] == "skip", f"a 99% margin must force a skip; got {v['reason']}"


class TestAchievableCeilingPhaseIntegration:
    """End-to-end footgun pin through ``run_composite_target_discovery``: the phase must record the verdict in metadata,
    skip discovery, and add ZERO composite specs on the strong-AR target -- even with ``extreme_ar_group_aware_skip=False``."""

    def _run_phase(self, cfg, df, groups, y):
        """Run phase."""
        target_by_type = {TargetTypes.REGRESSION: {"y": y}}
        metadata: dict = {}
        n = len(df)
        val = df.iloc[: n // 5]
        run_composite_target_discovery(
            composite_target_discovery_config=cfg,
            target_by_type=target_by_type,
            mlframe_models=["lgb"],
            metadata=metadata,
            filtered_train_df=df,
            filtered_train_idx=np.arange(n),
            train_df_pd=df,
            val_df_pd=val,
            test_df_pd=val,
            train_idx=np.arange(n),
            val_idx=np.arange(n // 5),
            test_idx=np.arange(n // 5),
            baseline_diagnostics_config=None,
            cat_features=[],
            verbose=False,
            group_ids=groups,
            split_config={"use_groups": True},
        )
        return metadata

    def test_biz_val_phase_skips_strong_ar_with_flag_off(self):
        """Biz val phase skips strong ar with flag off."""
        df, groups, y = _ar_frame()
        cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, extreme_ar_group_aware_skip=False)
        metadata = self._run_phase(cfg, df, groups, y)

        verdict = metadata.get("composite_precheck_verdict", {}).get(str(TargetTypes.REGRESSION), {}).get("y")
        assert verdict is not None, "the phase must record composite_precheck_verdict in metadata"
        assert verdict["decision"] == "skip", f"phase must SKIP the strong-AR target even with extreme_ar_group_aware_skip=False; got {verdict['reason']}"
        specs = metadata.get("composite_target_specs", {}).get(str(TargetTypes.REGRESSION), {}).get("y", [])
        assert not specs, f"no composite specs must be added when the ceiling precheck skips; got {specs}"

    def test_biz_val_phase_proceeds_on_composable_target(self):
        """Biz val phase proceeds on composable target."""
        df, groups, y = _composable_frame()
        cfg = CompositeTargetDiscoveryConfig(enabled=True, random_state=0, extreme_ar_group_aware_skip=False)
        metadata = self._run_phase(cfg, df, groups, y)

        verdict = metadata.get("composite_precheck_verdict", {}).get(str(TargetTypes.REGRESSION), {}).get("y")
        assert verdict is not None
        assert verdict["decision"] == "proceed", f"phase must PROCEED on a composable target; got {verdict['reason']}"
