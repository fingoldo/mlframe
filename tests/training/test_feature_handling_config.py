"""
TDD pack for the phase-A surface of ``mlframe.training.feature_handling``:

  * ``FeatureHandlingConfig`` top-level + 6 sub-configs.
  * ``TextHandlerSpec`` / ``CatHandlerSpec`` with discriminated-union
    params (round-3 R2-5 lock).
  * Compat matrix + early validation with difflib suggestions
    (round-2 U-R2-29).
  * Auto-derived memory budgets, cgroup-aware (round-3 R2-1).
  * Preset factories.
  * Effective handler chain assembly (defaults + per_model override
    + append).

The tests are the contract -- the implementation in phases B-G reads
these fields by attribute, so any behaviour change must update the
tests first.
"""

from __future__ import annotations

from unittest import mock

import pytest
from pydantic import ValidationError

from mlframe.training.feature_handling import (
    AutoDeriveConfig,
    Axis,
    CacheConfig,
    CatHandlerSpec,
    FeatureHandlingConfig,
    FrozenEmbeddingParams,
    HashingParams,
    LearnableEmbeddingParams,
    LoggingConfig,
    MemoryConfig,
    ModelHandlingOverride,
    NoParams,
    PricingConfig,
    ReproConfig,
    TargetEncodeParams,
    TextDetectionConfig,
    TextHandlerSpec,
    TfidfParams,
    cb_native_only,
    embedding_only,
    register_model_axis_support,
    tfidf_only,
    validate_handler_for_model,
)


# =====================================================================
# 1. Per-method TypedDict params + extra="forbid"
# =====================================================================

class TestTypedDictParamsExtraForbid:
    """Round-2 U-R2-25: ``params: Dict[str, Any]`` was a typo amnesty
    zone. Each TypedDict now uses ``extra="forbid"`` so misspelled
    fields raise immediately.
    """

    def test_tfidf_params_typo_raises(self):
        with pytest.raises(ValidationError, match="max_feature"):
            TfidfParams(max_feature=5000)  # typo of max_features

    def test_tfidf_params_wrong_arity_tuple_raises(self):
        with pytest.raises(ValidationError):
            TfidfParams(ngram_range=(1,))  # tuple length 1

    def test_hashing_params_typo_raises(self):
        with pytest.raises(ValidationError, match="n_feature"):
            HashingParams(n_feature=2**16)

    def test_frozen_embedding_params_typo_raises(self):
        with pytest.raises(ValidationError, match="poll"):
            FrozenEmbeddingParams(poll="cls")

    def test_target_encode_params_typo_raises(self):
        with pytest.raises(ValidationError, match="smothing"):
            TargetEncodeParams(kind="target_mean", smothing=10.0)

    def test_learnable_extends_frozen_inherits_pool(self):
        p = LearnableEmbeddingParams(pool="cls", finetune_lr_mult=0.05)
        assert p.pool == "cls"
        assert p.finetune_lr_mult == 0.05


# =====================================================================
# 2. HandlerSpec discriminated union — kind-method match validation
# =====================================================================

class TestHandlerSpecMethodKindMatch:
    """Round-3 R2-5: Pydantic Union[TfidfParams, HashingParams, ...]
    silent mismatch — without the ``model_post_init`` check, a
    ``method="hashing"`` with ``params=TfidfParams()`` could route to
    the wrong class. Lock the validator.
    """

    def test_text_method_matches_tfidf_params(self):
        s = TextHandlerSpec(method="tfidf", params=TfidfParams(max_features=1000))
        assert s.method == "tfidf"
        assert s.params.kind == "tfidf"

    def test_text_method_mismatch_raises(self):
        with pytest.raises(ValidationError, match="match"):
            TextHandlerSpec(method="hashing", params=TfidfParams())

    def test_cat_method_matches_target_mean_params(self):
        s = CatHandlerSpec(
            method="target_mean",
            params=TargetEncodeParams(kind="target_mean", smoothing=20.0),
        )
        assert s.method == "target_mean"
        assert s.params.smoothing == 20.0

    def test_cat_method_mismatch_target_kind_raises(self):
        with pytest.raises(ValidationError, match="match"):
            CatHandlerSpec(
                method="target_mean",
                params=TargetEncodeParams(kind="woe"),
            )

    def test_native_method_no_params_default(self):
        s = TextHandlerSpec(method="native")
        assert s.method == "native"

    def test_drop_method_no_params_default(self):
        s = TextHandlerSpec(method="drop")
        assert s.method == "drop"

    def test_method_requiring_params_with_no_params_raises(self):
        # method='tfidf' + default NoParams(kind='drop') is inconsistent
        with pytest.raises(ValidationError):
            TextHandlerSpec(method="tfidf")


# =====================================================================
# 3. Compat matrix + early validation with difflib
# =====================================================================

class TestCompatMatrixValidation:
    """Round-2 U-R2-29: every misconfig must fail BEFORE fit with a
    readable message that includes valid options + close-match
    suggestion.
    """

    def test_typo_method_yields_difflib_suggestion(self):
        with pytest.raises(ValueError, match="tfidf"):
            validate_handler_for_model("xgb", Axis.TEXT, "tdfif")

    def test_completely_wrong_method_lists_valid_options(self):
        with pytest.raises(ValueError, match="valid methods"):
            validate_handler_for_model("xgb", Axis.TEXT, "zzzz_unknown")

    def test_unknown_model_kind_raises(self):
        with pytest.raises(ValueError, match="register_model_axis_support"):
            validate_handler_for_model("not_a_model", Axis.TEXT, "tfidf")

    def test_native_text_only_for_cb(self):
        validate_handler_for_model("cb", Axis.TEXT, "native")  # ok
        with pytest.raises(ValueError, match="valid methods"):
            validate_handler_for_model("xgb", Axis.TEXT, "native")

    def test_as_embedding_feature_only_for_cb(self):
        # cb supports it
        validate_handler_for_model("cb", Axis.TEXT, "frozen_text_embedding", output="as_embedding_feature")
        # xgb doesn't
        with pytest.raises(ValueError, match="as_embedding_feature"):
            validate_handler_for_model(
                "xgb", Axis.TEXT, "frozen_text_embedding", output="as_embedding_feature",
            )

    def test_learnable_text_embedding_neural_only(self):
        # neural model -> ok
        validate_handler_for_model("mlp", Axis.TEXT, "learnable_text_embedding")
        # non-neural -> "requires a neural model" message (cross-cutting
        # rule wins over matrix lookup so users see the actionable
        # explanation, not just "method not in valid set").
        with pytest.raises(ValueError, match="neural"):
            validate_handler_for_model("xgb", Axis.TEXT, "learnable_text_embedding")
        with pytest.raises(ValueError, match="neural"):
            validate_handler_for_model("cb", Axis.TEXT, "learnable_text_embedding")

    def test_cat_embedding_neural_only_via_matrix(self):
        # mlp/recurrent/tabnet support cat="embedding"
        validate_handler_for_model("mlp", Axis.CAT, "embedding")
        validate_handler_for_model("tabnet", Axis.CAT, "embedding")
        # xgb does NOT
        with pytest.raises(ValueError, match="valid methods"):
            validate_handler_for_model("xgb", Axis.CAT, "embedding")

    def test_register_model_axis_support_idempotent(self):
        register_model_axis_support("__test_model__", Axis.TEXT, ["tfidf", "drop"])
        # Re-registering same set is a no-op.
        register_model_axis_support("__test_model__", Axis.TEXT, ["tfidf", "drop"])
        # Now validates against new entry.
        validate_handler_for_model("__test_model__", Axis.TEXT, "tfidf")

    def test_register_model_axis_support_conflict_raises(self):
        register_model_axis_support("__test_model_2__", Axis.TEXT, ["tfidf"])
        with pytest.raises(ValueError, match="already registered"):
            register_model_axis_support("__test_model_2__", Axis.TEXT, ["hashing"])


# =====================================================================
# 4. FHC validate_against_models — combined error reporting
# =====================================================================

class TestFhcValidateAgainstModels:
    def test_happy_path_single_model(self):
        fhc = FeatureHandlingConfig(
            default_text=[TextHandlerSpec(method="tfidf", params=TfidfParams())],
        )
        fhc.validate_against_models(["xgb"])  # should not raise

    def test_combined_error_lists_all_mismatches(self):
        # mlp: cat="native" not allowed (mlp cat must be embedding/ordinal/onehot/drop)
        # xgb: text="learnable_text_embedding" not allowed (neural-only)
        fhc = FeatureHandlingConfig(
            per_model={
                "mlp": ModelHandlingOverride(cat=[CatHandlerSpec(method="native")]),
                "xgb": ModelHandlingOverride(text=[
                    TextHandlerSpec(
                        method="learnable_text_embedding",
                        params=LearnableEmbeddingParams(),
                    )
                ]),
            },
        )
        with pytest.raises(ValueError) as excinfo:
            fhc.validate_against_models(["mlp", "xgb"])
        msg = str(excinfo.value)
        assert "mlp" in msg
        assert "xgb" in msg
        assert "incompatible handler/model combinations" in msg


# =====================================================================
# 5. Auto-derived memory budgets (cgroup-aware)
# =====================================================================

class TestAutoDeriveMemory:
    """Round-3 R2-1 + plan §1.1 + user-confirmed sub-configs nesting.

    Mock psutil.virtual_memory at construction time so tests are
    deterministic across machines (round-3 R2-1c). We mock the
    module-level psutil call inside ``system.py`` so the validator
    sees the mocked value.
    """

    @pytest.mark.parametrize(
        "total_gb,expected_budget,expected_cache,expected_reserve",
        [
            # 16 GB box: budget=11.2, cache=3.36, reserve=2.0 (capped to min)
            (16, 11.2, 16 * 0.7 * 0.3, 2.0),
            # 32 GB: cache=6.72, reserve=3.2 (10% of total)
            (32, 22.4, 32 * 0.7 * 0.3, 3.2),
            # 64 GB
            (64, 44.8, 64 * 0.7 * 0.3, 6.4),
        ],
    )
    def test_auto_derive_scales_with_total_ram(
        self, total_gb, expected_budget, expected_cache, expected_reserve,
    ):
        with mock.patch(
            "mlframe.training.feature_handling.system.psutil.virtual_memory"
        ) as mock_vm, mock.patch(
            "mlframe.training.feature_handling.system._read_cgroup_memory_limit_bytes",
            return_value=None,
        ):
            mock_vm.return_value = mock.MagicMock(total=int(total_gb * 1e9))
            fhc = FeatureHandlingConfig()
            assert abs(fhc.memory.budget_gb - expected_budget) < 0.01
            assert abs(fhc.cache.ram_max_gb - expected_cache) < 0.01
            assert abs(fhc.cache.ram_reserve_gb - expected_reserve) < 0.01

    def test_auto_derive_with_cgroup_limit_uses_min(self):
        """Inside a 4 GB container on a 64 GB host, budget derives
        from the cgroup limit, not host RAM."""
        with mock.patch(
            "mlframe.training.feature_handling.system.psutil.virtual_memory"
        ) as mock_vm, mock.patch(
            "mlframe.training.feature_handling.system._read_cgroup_memory_limit_bytes",
            return_value=int(4 * 1e9),
        ):
            mock_vm.return_value = mock.MagicMock(total=int(64 * 1e9))
            fhc = FeatureHandlingConfig()
            # 4 GB total -> 0.7 fraction = 2.8 GB; reserve floor 2.0;
            # cache_ram = 2.8 * 0.3 = 0.84
            assert abs(fhc.memory.budget_gb - 2.8) < 0.01
            assert abs(fhc.cache.ram_reserve_gb - 2.0) < 0.01

    def test_explicit_budget_skips_auto_derive(self):
        fhc = FeatureHandlingConfig(memory=MemoryConfig(budget_gb=99.0))
        assert fhc.memory.budget_gb == 99.0

    def test_invalid_budget_on_tiny_machine_raises(self):
        """1 GB total + keep_free_min=2 -> ram_max + reserve > total
        -> ValueError at construction."""
        with mock.patch(
            "mlframe.training.feature_handling.system.psutil.virtual_memory"
        ) as mock_vm, mock.patch(
            "mlframe.training.feature_handling.system._read_cgroup_memory_limit_bytes",
            return_value=None,
        ):
            mock_vm.return_value = mock.MagicMock(total=int(1 * 1e9))
            with pytest.raises(ValidationError, match="invalid memory budget"):
                FeatureHandlingConfig()

    def test_resolved_property(self):
        with mock.patch(
            "mlframe.training.feature_handling.system.psutil.virtual_memory"
        ) as mock_vm, mock.patch(
            "mlframe.training.feature_handling.system._read_cgroup_memory_limit_bytes",
            return_value=None,
        ):
            mock_vm.return_value = mock.MagicMock(total=int(32 * 1e9))
            fhc = FeatureHandlingConfig()
            res = fhc.resolved
            assert "memory_budget_gb" in res
            assert "cache_ram_max_gb" in res
            assert "cache_ram_reserve_gb" in res

    def test_env_override_takes_precedence(self, monkeypatch):
        """``MLFRAME_MEMORY_BUDGET_GB`` env wins over psutil + cgroup."""
        monkeypatch.setenv("MLFRAME_MEMORY_BUDGET_GB", "20.0")
        # detect_memory_limit_bytes returns 20 GB -> budget = 20 * 0.7 = 14
        fhc = FeatureHandlingConfig()
        assert abs(fhc.memory.budget_gb - 14.0) < 0.01


# =====================================================================
# 6. Sub-config extra="forbid"
# =====================================================================

class TestSubConfigExtraForbid:
    @pytest.mark.parametrize(
        "config_cls,bad_kwargs",
        [
            (CacheConfig, {"persistance": "off"}),  # typo
            (MemoryConfig, {"budgett_gb": 16.0}),
            (PricingConfig, {"capacity_usd": 5.0}),
            (LoggingConfig, {"verbose_logging": True}),  # actually shouldn't exist
            (ReproConfig, {"determinitic_torch": True}),  # typo
            (TextDetectionConfig, {"text_min_chars": 30}),
            (AutoDeriveConfig, {"memory_budget_pct": 0.7}),
        ],
    )
    def test_typo_raises(self, config_cls, bad_kwargs):
        with pytest.raises(ValidationError):
            config_cls(**bad_kwargs)


# =====================================================================
# 7. ModelHandlingOverride append/replace semantics
# =====================================================================

class TestEffectiveSpecsAppendReplace:
    def test_replace_replaces_defaults(self):
        fhc = FeatureHandlingConfig(
            default_text=[TextHandlerSpec(method="tfidf", params=TfidfParams())],
            per_model={
                "cb": ModelHandlingOverride(text=[
                    TextHandlerSpec(method="native"),
                ]),
            },
        )
        # cb uses override (native), not default (tfidf)
        cb_specs = fhc._effective_text_specs("cb")
        assert [s.method for s in cb_specs] == ["native"]
        # xgb has no override -> default tfidf
        xgb_specs = fhc._effective_text_specs("xgb")
        assert [s.method for s in xgb_specs] == ["tfidf"]

    def test_append_extends_defaults(self):
        fhc = FeatureHandlingConfig(
            default_text=[TextHandlerSpec(method="tfidf", params=TfidfParams())],
            per_model={
                "linear": ModelHandlingOverride(text_append=[
                    TextHandlerSpec(method="hashing", params=HashingParams()),
                ]),
            },
        )
        linear_specs = fhc._effective_text_specs("linear")
        assert [s.method for s in linear_specs] == ["tfidf", "hashing"]

    def test_replace_plus_append(self):
        fhc = FeatureHandlingConfig(
            default_text=[TextHandlerSpec(method="tfidf", params=TfidfParams())],
            per_model={
                "mlp": ModelHandlingOverride(
                    text=[TextHandlerSpec(method="frozen_text_embedding", params=FrozenEmbeddingParams())],
                    text_append=[TextHandlerSpec(method="hashing", params=HashingParams())],
                ),
            },
        )
        mlp_specs = fhc._effective_text_specs("mlp")
        assert [s.method for s in mlp_specs] == ["frozen_text_embedding", "hashing"]


# =====================================================================
# 8. Preset factories
# =====================================================================

class TestPresetFactories:
    def test_tfidf_only(self):
        fhc = tfidf_only(max_features=10000, ngram_range=(1, 3))
        assert len(fhc.default_text) == 1
        assert fhc.default_text[0].method == "tfidf"
        assert fhc.default_text[0].params.max_features == 10000
        assert fhc.default_text[0].params.ngram_range == (1, 3)
        # Phase Q (2026-05-09) flipped the universal cat default from
        # "native" (works only for cb/xgb/lgb/tabnet) to "ordinal"
        # (works for ALL model kinds). The cat=native restriction caused
        # validate_against_models() to reject HGB/MLP/linear/etc users
        # of the preset; "ordinal" passes validation everywhere.
        assert fhc.default_cat[0].method == "ordinal"

    def test_cb_native_only(self):
        fhc = cb_native_only()
        # default text drops; cb override flips to native
        assert fhc.default_text[0].method == "drop"
        assert fhc.per_model["cb"].text[0].method == "native"
        # validate_against_models with cb-only must succeed
        fhc.validate_against_models(["cb"])

    def test_embedding_only(self):
        fhc = embedding_only(provider=None, pool="cls")
        assert fhc.default_text[0].method == "frozen_text_embedding"
        assert fhc.default_text[0].params.pool == "cls"


# =====================================================================
# 9. apply_to_columns behaviour
# =====================================================================

class TestApplyToColumns:
    def test_apply_to_columns_explicit_list(self):
        s = TextHandlerSpec(
            method="tfidf",
            params=TfidfParams(),
            apply_to_columns=["desc", "title"],
        )
        assert s.apply_to_columns == ["desc", "title"]

    def test_apply_to_columns_default_none(self):
        s = TextHandlerSpec(method="tfidf", params=TfidfParams())
        assert s.apply_to_columns is None


# =====================================================================
# 10. describe() basic shape
# =====================================================================

class TestDescribe:
    def test_describe_short_returns_dict(self):
        fhc = FeatureHandlingConfig()
        d = fhc.describe(short=True)
        assert "mode" in d
        assert "cache_persistence" in d
        assert "resolved_memory_gb" in d

    def test_describe_verbose_includes_subconfigs(self):
        fhc = FeatureHandlingConfig()
        d = fhc.describe(short=False)
        assert "text_detection" in d
        assert "cache" in d
        assert "memory" in d
        assert "repro" in d
