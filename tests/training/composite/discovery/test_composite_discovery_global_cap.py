"""``max_total_composite_targets`` caps composite targets ACROSS the whole run, not per base target.

Pre-fix, discovery wrote every accepted spec straight into ``target_by_type`` inside the per-target
loop with no overall ceiling -- a production run with 5 heavy-tailed base targets accepted ~10 specs
from EACH, landing 47 composite targets in one suite (each trained cb+xgb+lgb with the full post-fit
diagnostics suite). The cap is applied globally, AFTER every base target's own discovery has run, by
sorting all accepted specs by their honest-holdout OOS RMSE gain (relative to the raw-y baseline) and
keeping the best-scoring ones -- so a strong target isn't starved by an equal per-target share.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mlframe.training.configs import CompositeTargetDiscoveryConfig, TargetTypes
from mlframe.training.core._phase_composite_discovery import run_composite_target_discovery


def _strong_ar_target(base: np.ndarray, rng: np.random.Generator, noise_scale: float) -> np.ndarray:
    """y = 0.95 * base + small extra signal + noise -- a near-copy AR-lag base composite discovery accepts easily."""
    x_extra = rng.normal(size=base.shape[0])
    return 0.95 * base + 0.4 * x_extra + rng.normal(scale=noise_scale, size=base.shape[0])


def _run(cfg: CompositeTargetDiscoveryConfig, targets: dict, feats_df: pd.DataFrame):
    """Run discovery over ``targets`` ({name: y array}) against the same feature frame for every target."""
    full_idx = np.arange(len(feats_df))
    metadata: dict = {}
    target_by_type = {TargetTypes.REGRESSION: dict(targets)}
    return run_composite_target_discovery(
        composite_target_discovery_config=cfg,
        target_by_type=target_by_type,
        mlframe_models=None,
        metadata=metadata,
        filtered_train_df=feats_df,
        filtered_train_idx=full_idx,
        train_df_pd=feats_df,
        val_df_pd=feats_df,
        test_df_pd=feats_df,
        train_idx=full_idx,
        val_idx=full_idx,
        test_idx=full_idx,
        baseline_diagnostics_config=None,
        cat_features=None,
        verbose=False,
    )


def _synthetic_two_targets(seed: int = 0, n: int = 1500):
    """Two independent strong-AR-lag base targets sharing one feature frame, both easy composite accepts."""
    rng = np.random.default_rng(seed)
    base_a = rng.normal(loc=10.0, scale=3.0, size=n)
    base_b = rng.normal(loc=-5.0, scale=2.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    df = pd.DataFrame({"base_a": base_a, "base_b": base_b, "x1": x1, "x2": x2})
    y_a = _strong_ar_target(base_a, rng, noise_scale=0.3)
    y_b = _strong_ar_target(base_b, rng, noise_scale=0.3)
    return df, {"target_a": y_a.astype(np.float64), "target_b": y_b.astype(np.float64)}


class TestGlobalCompositeCap:
    """Groups tests for: TestGlobalCompositeCap."""

    def test_default_config_has_a_real_global_cap(self):
        """0 would mean no cap at all, defeating the point of a global budget."""
        cfg = CompositeTargetDiscoveryConfig()
        assert cfg.max_total_composite_targets is not None
        assert cfg.max_total_composite_targets > 0

    def test_none_disables_the_cap_keeping_every_discovered_spec(self):
        """The escape hatch: an explicit None must not drop anything the per-target discovery accepted.

        ``min_honest_gain_to_train`` is the other, independent reason a spec is not trained; switched off here so the
        assertion measures the cap alone (its own test is below).
        """
        feats_df, targets = _synthetic_two_targets()
        cfg = CompositeTargetDiscoveryConfig(enabled=True, max_total_composite_targets=None, min_honest_gain_to_train=None)
        target_by_type, metadata = _run(cfg, targets, feats_df)

        n_discovered = sum(len(v) for tt_specs in metadata["composite_target_specs"].values() for v in tt_specs.values())
        n_in_target_by_type = len(target_by_type[TargetTypes.REGRESSION]) - len(targets)
        assert n_discovered > 0, "both targets are strong-AR near-copies; discovery must accept at least one spec each"
        assert n_in_target_by_type == n_discovered

    def test_gain_floor_drops_specs_that_do_not_beat_raw(self, caplog):
        """A spec whose honest-holdout RMSE gain is at or below the floor is not trained: the cap alone only picks the
        best N, so on a run where few specs help it still shipped specs that lose to raw y (12 of 25 in production)."""
        feats_df, targets = _synthetic_two_targets()
        no_floor = CompositeTargetDiscoveryConfig(enabled=True, max_total_composite_targets=None, min_honest_gain_to_train=None)
        tbt_all, _ = _run(no_floor, targets, feats_df)
        n_all = len(tbt_all[TargetTypes.REGRESSION]) - len(targets)

        floored = CompositeTargetDiscoveryConfig(enabled=True, max_total_composite_targets=None, min_honest_gain_to_train=0.5)
        with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_composite_discovery"):
            tbt_floor, _ = _run(floored, targets, feats_df)
        n_floor = len(tbt_floor[TargetTypes.REGRESSION]) - len(targets)
        assert n_floor < n_all, "a 50%-of-baseline floor must drop specs this DGP cannot clear"
        assert "min_honest_gain_to_train" in caplog.text

    def test_cap_keeps_only_the_global_budget_across_both_targets(self, caplog):
        """A cap smaller than the total discovered across BOTH targets must still leave target_by_type
        with exactly the capped count -- not the capped count PER target (which would be 2x too many)."""
        feats_df, targets = _synthetic_two_targets()
        # This test isolates the global CAP. The fixture's discovered gains are noise-level (relative honest gains
        # of +0.6%..+1.4%, each about a quarter of its own paired standard error on this small synthetic), so the
        # noise-aware ship floor -- tested in test_discovery_gates_audit_2026_09_20.py -- would drop every one of them
        # before the cap could act. min_honest_gain_z=0 keeps that orthogonal gate out of the way; the constant
        # min_honest_gain_to_train floor still applies, and the production default stays 2.
        cfg_uncapped = CompositeTargetDiscoveryConfig(enabled=True, max_total_composite_targets=None, min_honest_gain_z=0.0)
        _, metadata_uncapped = _run(cfg_uncapped, targets, feats_df)
        n_discovered = sum(len(v) for tt_specs in metadata_uncapped["composite_target_specs"].values() for v in tt_specs.values())
        assert n_discovered >= 2, "need at least 2 discovered specs across both targets to prove the cap bites"

        cap = 1
        cfg_capped = CompositeTargetDiscoveryConfig(enabled=True, max_total_composite_targets=cap, min_honest_gain_z=0.0)
        with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_composite_discovery"):
            target_by_type, _ = _run(cfg_capped, targets, feats_df)
        n_in_target_by_type = len(target_by_type[TargetTypes.REGRESSION]) - len(targets)
        assert n_in_target_by_type == cap
        text = " ".join(r.getMessage() for r in caplog.records)
        assert "global cap" in text

    def test_cap_keeps_the_best_scoring_spec_not_an_arbitrary_one(self):
        """With cap=1 and two targets of very different noise (hence very different honest RMSE gain),
        the kept spec must come from the CLEANER (higher-gain) target, not whichever discovery visited
        first -- proving the selection is by score, not by processing order."""
        rng = np.random.default_rng(1)
        n = 1500
        base_clean = rng.normal(loc=10.0, scale=3.0, size=n)
        base_noisy = rng.normal(loc=-5.0, scale=2.0, size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        feats_df = pd.DataFrame({"base_clean": base_clean, "base_noisy": base_noisy, "x1": x1, "x2": x2})
        y_clean = _strong_ar_target(base_clean, rng, noise_scale=0.05)  # near-noiseless -- high honest gain
        y_noisy = _strong_ar_target(base_noisy, rng, noise_scale=3.0)  # heavy noise -- low/no honest gain
        targets = {"target_clean": y_clean.astype(np.float64), "target_noisy": y_noisy.astype(np.float64)}

        cfg = CompositeTargetDiscoveryConfig(enabled=True, max_total_composite_targets=1)
        target_by_type, _metadata = _run(cfg, targets, feats_df)
        kept_names = set(target_by_type[TargetTypes.REGRESSION]) - set(targets)
        assert len(kept_names) == 1, f"expected exactly 1 composite kept under cap=1, got {kept_names}"
        kept_name = next(iter(kept_names))
        assert kept_name.startswith("target_clean-"), f"cap=1 kept '{kept_name}' -- expected the CLEAN target's spec (higher honest gain), not the noisy one"
