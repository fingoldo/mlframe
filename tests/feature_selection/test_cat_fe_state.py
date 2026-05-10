"""Smoke tests for ``CatFEConfig`` / ``CatFEState`` dataclasses.

These verify the persistence contract that the rest of cat-FE relies
on: configs round-trip through pickle (joblib uses pickle under the
hood) and through ``sklearn.base.clone`` (GridSearchCV / Pipeline
re-introspection).
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, fields

import pytest

from mlframe.feature_selection.filters.cat_fe_state import (
    CatFEConfig,
    CatFEState,
)


class TestCatFEConfigDefaults:
    """Defaults match the plan v3 contract -- this test pins them so a
    future innocent-looking edit doesn't silently change opt-in
    behaviour."""

    def test_enabled_by_default(self):
        """2026-05-11: default flipped to enable=True per mlframe rule
        'Accuracy / performance over legacy'. To get legacy behaviour
        explicitly pass ``CatFEConfig(enable=False)``."""
        cfg = CatFEConfig()
        assert cfg.enable is True, (
            "Default flipped to enable=True since 2026-05-11; "
            "see CHANGELOG. Legacy: CatFEConfig(enable=False)."
        )

    def test_full_npermutations_nonzero_by_default(self):
        """SB4: zero is an anti-statistical trap. The default keeps
        users from accidentally surfacing pairs with no FWER guarantee.
        2026-05-11: default reduced from 100 to 50 for conservative
        on-by-default tuning; still well above zero."""
        cfg = CatFEConfig()
        assert cfg.full_npermutations >= 50

    def test_fwer_correction_default_is_none(self):
        """2026-05-11: default flipped to 'none'. Rationale: with the
        on-by-default ``full_npermutations=50`` perm budget, BH-FDR/WY
        on 28+ pair families mathematically can't reject ANYTHING
        (min raw p=1/51=0.02; BH-corrected = 0.02 * m/k > 0.05 for
        any reasonable m). Users who want strict FWER control bump
        permutations to 500-1000 AND set fwer_correction='bh_fdr' or
        'westfall_young'."""
        cfg = CatFEConfig()
        assert cfg.fwer_correction == "none"

    def test_select_on_default_is_synergy(self):
        cfg = CatFEConfig()
        assert cfg.select_on == "synergy"

    def test_include_numeric_default_is_false(self):
        """SM9: discretized noisy floats produce spurious aliasing.
        Opt-in only."""
        cfg = CatFEConfig()
        assert cfg.include_numeric is False

    def test_min_interaction_information_default_is_none(self):
        """``None`` -> resolved at fit time to -3/sqrt(n) (B4)."""
        cfg = CatFEConfig()
        assert cfg.min_interaction_information is None

    def test_max_combined_nbins_default_is_none(self):
        """``None`` -> resolved at fit time via Paninski formula (SM5)."""
        cfg = CatFEConfig()
        assert cfg.max_combined_nbins is None

    def test_top_k_pairs_reasonable(self):
        """2026-05-11: default reduced from 64 to 32 for the
        now-on-by-default conservative tuning."""
        cfg = CatFEConfig()
        assert cfg.top_k_pairs == 32

    def test_max_kway_order_pairs_only(self):
        """Default ``2`` = pairs only. K-way greedy requires explicit opt-in."""
        cfg = CatFEConfig()
        assert cfg.max_kway_order == 2

    def test_emit_diagnostics_on_by_default(self):
        """E4: diagnostics are cheap and load-bearing for debugging."""
        cfg = CatFEConfig()
        assert cfg.emit_diagnostics is True


class TestPersistence:
    """Configs and state must survive pickle (joblib) and clone (sklearn
    GridSearchCV / Pipeline)."""

    def test_config_pickle_round_trip(self):
        cfg = CatFEConfig(
            enable=True,
            top_k_pairs=128,
            max_kway_order=3,
            backend="cpu",
            n_folds_stability=5,
        )
        restored = pickle.loads(pickle.dumps(cfg))
        assert restored == cfg

    def test_state_pickle_round_trip(self):
        st = CatFEState()
        st.dropped_singleton_nbins.append("col_const")
        st.diagnostics["mul(a,b)"] = {"II": 0.05, "joint_MI": 0.07}
        restored = pickle.loads(pickle.dumps(st))
        assert restored.dropped_singleton_nbins == ["col_const"]
        assert restored.diagnostics["mul(a,b)"] == {"II": 0.05, "joint_MI": 0.07}

    def test_config_asdict_serializable(self):
        """``asdict`` needed by introspection / repr / external logging."""
        cfg = CatFEConfig(enable=True)
        d = asdict(cfg)
        assert isinstance(d, dict)
        assert d["enable"] is True
        # Round-trip through dict
        cfg2 = CatFEConfig(**d)
        assert cfg2 == cfg

    def test_field_count_pinned(self):
        """If a future PR adds a field without thinking about BC, this
        test forces an explicit acknowledgement: bump the expected
        count, update the v3 plan and ``__setstate__`` defaults at the
        same time."""
        # Count current fields; pin via assert. New field => deliberate update.
        cfg_field_count = len(fields(CatFEConfig))
        state_field_count = len(fields(CatFEState))
        # Plan v3 SB8 lists 14 core knobs + extensions. Keep the
        # pin loose (just floor + ceiling) so reasonable refactors
        # don't churn the test, but a careless +5 or -3 trips it.
        assert 18 <= cfg_field_count <= 45, \
            f"CatFEConfig has {cfg_field_count} fields; if intentional update the pin"
        assert 4 <= state_field_count <= 12, \
            f"CatFEState has {state_field_count} fields; if intentional update the pin"


class TestPostInitValidation:
    """Tier 1.2: __post_init__ validation catches misconfig at
    construction time (not deep in fit())."""

    def test_zero_top_k_rejected(self):
        with pytest.raises(ValueError, match="top_k_pairs"):
            CatFEConfig(top_k_pairs=0)

    def test_negative_top_k_rejected(self):
        with pytest.raises(ValueError, match="top_k_pairs"):
            CatFEConfig(top_k_pairs=-5)

    def test_invalid_max_kway_order_rejected(self):
        with pytest.raises(ValueError, match="max_kway_order"):
            CatFEConfig(max_kway_order=1)

    def test_negative_permutations_rejected(self):
        with pytest.raises(ValueError, match="full_npermutations"):
            CatFEConfig(full_npermutations=-1)
        with pytest.raises(ValueError, match="shortlist_npermutations"):
            CatFEConfig(shortlist_npermutations=-1)

    def test_negative_marginal_floor_rejected(self):
        with pytest.raises(ValueError, match="marginal_floor"):
            CatFEConfig(marginal_floor=-0.1)

    def test_tiny_max_combined_nbins_rejected(self):
        with pytest.raises(ValueError, match="max_combined_nbins"):
            CatFEConfig(max_combined_nbins=2)

    def test_fold_prevalence_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="min_fold_prevalence"):
            CatFEConfig(min_fold_prevalence=1.5)
        with pytest.raises(ValueError, match="min_fold_prevalence"):
            CatFEConfig(min_fold_prevalence=-0.1)

    def test_negative_anti_redundancy_beta_rejected(self):
        with pytest.raises(ValueError, match="anti_redundancy_beta"):
            CatFEConfig(anti_redundancy_beta=-0.5)

    def test_min_n_samples_below_2_rejected(self):
        with pytest.raises(ValueError, match="min_n_samples"):
            CatFEConfig(min_n_samples=1)

    def test_min_class_count_below_1_rejected(self):
        with pytest.raises(ValueError, match="min_class_count"):
            CatFEConfig(min_class_count=0)

    def test_shortlist_perms_exceeding_full_rejected(self):
        with pytest.raises(ValueError, match="shortlist_npermutations"):
            CatFEConfig(shortlist_npermutations=200, full_npermutations=50)

    def test_n_folds_stability_negative_rejected(self):
        with pytest.raises(ValueError, match="n_folds_stability"):
            CatFEConfig(n_folds_stability=-1)

    def test_valid_config_accepted(self):
        # Sanity: well-formed config passes
        cfg = CatFEConfig(
            enable=True, top_k_pairs=16,
            max_kway_order=3, min_n_samples=100,
            full_npermutations=200, shortlist_npermutations=10,
        )
        assert cfg.top_k_pairs == 16


class TestConfigSemantics:
    """Spot-check that the literal-typed fields actually accept their
    documented values (catches typos in the dataclass declaration)."""

    @pytest.mark.parametrize("value", ["none", "bonferroni", "bh_fdr", "westfall_young"])
    def test_fwer_correction_accepts_documented_values(self, value):
        cfg = CatFEConfig(fwer_correction=value)
        assert cfg.fwer_correction == value

    @pytest.mark.parametrize("value", ["synergy", "redundancy", "absolute"])
    def test_select_on_accepts_documented_values(self, value):
        cfg = CatFEConfig(select_on=value)
        assert cfg.select_on == value

    @pytest.mark.parametrize("value", ["auto", "cpu", "gpu"])
    def test_backend_accepts_documented_values(self, value):
        cfg = CatFEConfig(backend=value)
        assert cfg.backend == value

    @pytest.mark.parametrize("value", ["clip", "sentinel", "raise"])
    def test_unknown_strategy_accepts_documented_values(self, value):
        cfg = CatFEConfig(unknown_strategy=value)
        assert cfg.unknown_strategy == value

    @pytest.mark.parametrize("value", ["and", "or", "permutation_primary"])
    def test_gate_logic_accepts_documented_values(self, value):
        cfg = CatFEConfig(gate_logic=value)
        assert cfg.gate_logic == value
