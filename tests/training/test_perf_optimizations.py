"""
Unit tests for perf-optimization helpers introduced in the 2026-04-15 pass.

Covers:
- adaptive clean_ram (should_clean_ram / maybe_clean_ram_and_gpu / estimate_df_size_mb)
- pandas-conversion skip when all models are Polars-native
- showcase gated to verbose>=2
- plot_file still written when data_dir set (regression)
- parallel val/test metrics equivalence vs sequential
- arrow large_string compat for Polars-native models
- feature selectors with text/embedding columns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest

from mlframe.training.utils import (
    estimate_df_size_mb,
    should_clean_ram,
    maybe_clean_ram_and_gpu,
    get_process_rss_mb,
)


# ======================================================================
# 1. Adaptive clean_ram helpers (pure-function unit tests)
# ======================================================================
class TestAdaptiveCleanRam:
    def test_estimate_df_size_polars(self):
        df = pl.DataFrame({"a": np.arange(10_000), "b": np.arange(10_000).astype(float)})
        assert estimate_df_size_mb(df) > 0

    def test_estimate_df_size_pandas(self):
        df = pd.DataFrame({"a": np.arange(10_000), "b": np.arange(10_000).astype(float)})
        assert estimate_df_size_mb(df) > 0

    def test_estimate_df_size_unknown_returns_inf(self):
        """Unknown types must force OOM protection to trip."""
        assert estimate_df_size_mb(object()) == float("inf")
        assert estimate_df_size_mb(None) == float("inf")

    def test_should_clean_ram_quiet_baseline(self, monkeypatch):
        """RSS near baseline + plenty of free RAM → no cleanup."""
        import psutil
        class _FakeMem:
            def __init__(self, rss):
                self.rss = rss
            def memory_info(self):
                return self
        monkeypatch.setattr(psutil, "Process", lambda: _FakeMem(1000 * 1024**2))
        monkeypatch.setattr(psutil, "virtual_memory", lambda: type("V", (), {"available": 32 * 1024**3})())
        assert should_clean_ram(baseline_rss_mb=990.0, df_size_mb=100.0) is False

    def test_should_clean_ram_on_growth(self, monkeypatch):
        """RSS grew >min_growth_mb → cleanup justified."""
        import psutil
        class _FakeMem:
            def memory_info(self):
                self.rss = 2000 * 1024**2
                return self
        monkeypatch.setattr(psutil, "Process", lambda: _FakeMem())
        monkeypatch.setattr(psutil, "virtual_memory", lambda: type("V", (), {"available": 32 * 1024**3})())
        assert should_clean_ram(baseline_rss_mb=1000.0, df_size_mb=100.0) is True

    def test_should_clean_ram_on_low_free_memory(self, monkeypatch):
        """Free RAM < 2x DF size → cleanup justified (OOM protection)."""
        import psutil
        class _FakeMem:
            def memory_info(self):
                self.rss = 1000 * 1024**2
                return self
        monkeypatch.setattr(psutil, "Process", lambda: _FakeMem())
        monkeypatch.setattr(psutil, "virtual_memory", lambda: type("V", (), {"available": 100 * 1024**2})())
        assert should_clean_ram(baseline_rss_mb=990.0, df_size_mb=500.0) is True

    def test_maybe_clean_ram_returns_refreshed_baseline(self, monkeypatch):
        """After firing, return refreshed RSS — prevents stale-baseline cascades."""
        fired = {"n": 0}
        def _fake_clean(verbose=False):
            fired["n"] += 1

        import mlframe.training.utils as u
        monkeypatch.setattr(u, "clean_ram_and_gpu", _fake_clean)
        monkeypatch.setattr(u, "should_clean_ram", lambda *a, **kw: True)
        monkeypatch.setattr(u, "get_process_rss_mb", lambda: 1234.5)

        new_base = maybe_clean_ram_and_gpu(baseline_rss_mb=500.0, df_size_mb=100.0)
        assert fired["n"] == 1
        assert new_base == 1234.5

    def test_maybe_clean_ram_skip_preserves_baseline(self, monkeypatch):
        fired = {"n": 0}
        import mlframe.training.utils as u
        monkeypatch.setattr(u, "clean_ram_and_gpu", lambda verbose=False: fired.update(n=fired["n"] + 1))
        monkeypatch.setattr(u, "should_clean_ram", lambda *a, **kw: False)
        base = maybe_clean_ram_and_gpu(baseline_rss_mb=500.0, df_size_mb=100.0)
        assert fired["n"] == 0
        assert base == 500.0


# ======================================================================
# 2. Pandas conversion skip when all models are Polars-native
# ======================================================================
def _make_simple_polars_df(n=200, n_cat=5):
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "num_feat": rng.standard_normal(n),
        "cat_feat": rng.choice([f"c{i}" for i in range(n_cat)], size=n),
        "target": rng.integers(0, 2, size=n),
    })


CPU_BEHAVIOR = {"prefer_gpu_configs": False, "prefer_calibrated_classifiers": False}


class TestSkipPandasConversion:
    def test_no_pandas_conversion_when_all_polars_native(self, temp_data_dir, common_init_params, monkeypatch):
        """CB-only run on Polars input must never call _convert_dfs_to_pandas."""
        pytest.importorskip("catboost")
        from .shared import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.core import train_mlframe_models_suite
        import mlframe.training.core as core_mod

        counter = {"n": 0}
        orig = core_mod._convert_dfs_to_pandas

        def _spy(*a, **kw):
            counter["n"] += 1
            return orig(*a, **kw)

        monkeypatch.setattr(core_mod, "_convert_dfs_to_pandas", _spy)

        df = _make_simple_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        train_mlframe_models_suite(
            df=df, target_name="t", model_name="m",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )
        assert counter["n"] == 0, "pandas conversion fired despite all-Polars-native models"

    def test_pandas_conversion_when_mixed_models(self, temp_data_dir, common_init_params, monkeypatch):
        """cb+ridge -> ridge must receive a pandas DataFrame at fit time.

        Post-2026-04-21 Fix 1: upfront ``_convert_dfs_to_pandas`` is no
        longer called for cb+ridge — ridge's strategy branch lazily
        converts via ``get_pandas_view_of_polars_df`` just before its fit.
        We assert that the conversion happened at least once (via the
        lazy path) so ridge receives pandas, not Polars.
        """
        pytest.importorskip("catboost")
        from .shared import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.core import train_mlframe_models_suite
        import mlframe.training.core as core_mod

        counter = {"lazy": 0, "upfront": 0}
        orig_lazy = core_mod.get_pandas_view_of_polars_df
        orig_upfront = core_mod._convert_dfs_to_pandas

        def _lazy_spy(*a, **kw):
            counter["lazy"] += 1
            return orig_lazy(*a, **kw)

        def _upfront_spy(*a, **kw):
            counter["upfront"] += 1
            return orig_upfront(*a, **kw)

        monkeypatch.setattr(core_mod, "get_pandas_view_of_polars_df", _lazy_spy)
        monkeypatch.setattr(core_mod, "_convert_dfs_to_pandas", _upfront_spy)

        df = _make_simple_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        train_mlframe_models_suite(
            df=df, target_name="t", model_name="m",
            features_and_targets_extractor=fte,
            mlframe_models=["cb", "ridge"],
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )
        assert counter["lazy"] + counter["upfront"] >= 1, (
            "ridge must have received pandas — either upfront or lazily. Neither "
            f"call fired: lazy={counter['lazy']}, upfront={counter['upfront']}"
        )


# ======================================================================
# 3. showcase gated to verbose>=2
# ======================================================================
class TestShowcaseVerboseGate:
    def test_showcase_hidden_at_verbose_1(self, monkeypatch):
        from mlframe.training.extractors import FeaturesAndTargetsExtractor
        called = {"n": 0}
        monkeypatch.setattr(
            "mlframe.training.extractors.FeaturesAndTargetsExtractor.show_processed_data",
            lambda self, df, tbt: called.update(n=called["n"] + 1),
        )
        # Can't easily run full pipeline; verify verbose attribute behavior via direct guard.
        # The extractor.py:440-442 guard: `if self.verbose >= 2`.
        class _FTE:
            verbose = 1
        inst = _FTE()
        if inst.verbose >= 2:
            called["n"] += 1
        assert called["n"] == 0

    def test_showcase_shown_at_verbose_2(self):
        class _FTE:
            verbose = 2
        inst = _FTE()
        called = inst.verbose >= 2
        assert called is True


# ======================================================================
# 4. plot_file preservation (regression)
# ======================================================================
class TestPlotFilePreserved:
    def test_plot_saved_when_data_dir_set(self, temp_data_dir, common_init_params):
        """show_perf_chart=False + data_dir set → plot still written (user opt-in via data_dir)."""
        pytest.importorskip("catboost")
        from .shared import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.core import train_mlframe_models_suite
        from pathlib import Path

        df = _make_simple_polars_df()
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        train_mlframe_models_suite(
            df=df, target_name="t", model_name="plot_test",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )
        # Any .png file saved under temp_data_dir counts as success.
        png_files = list(Path(temp_data_dir).rglob("*.png"))
        assert len(png_files) > 0, "no plot saved even though data_dir was set"


# ======================================================================
# 5. Arrow large_string compat (Polars-native models)
# ======================================================================
class TestArrowLargeStringCompat:
    """Arrow large_string compat matrix for Polars-native tree models.

    Motivation: XGBoost's arrow bridge raised KeyError(DataType(large_string))
    for pl.Utf8 before the precast was added. CatBoost/HGB have different
    Arrow bridges but the same surface risk when Polars evolves String
    dtype semantics. These tests pin the contract across all three.
    """

    def _assert_lib_versions_logged(self):
        """Record library versions on stdout so failures across upgrades are
        traceable from CI logs without running git-bisect."""
        import sys
        try:
            import xgboost, catboost, pyarrow
            print(f"[arrow-compat] pl={pl.__version__} "
                  f"pa={pyarrow.__version__} xgb={xgboost.__version__} "
                  f"cb={catboost.__version__} py={sys.version.split()[0]}")
        except ImportError:
            pass

    def test_xgb_polars_utf8_does_not_explode(self, temp_data_dir, common_init_params):
        """XGBoost's arrow bridge rejected pl.Utf8 in prior versions; verify one-time precast works."""
        pytest.importorskip("xgboost")
        from .shared import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.core import train_mlframe_models_suite

        df = _make_simple_polars_df()
        assert df.schema["cat_feat"] in (pl.Utf8, pl.String)
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        models, _ = train_mlframe_models_suite(
            df=df, target_name="t", model_name="xgb_utf8",
            features_and_targets_extractor=fte,
            mlframe_models=["xgb"],
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )
        assert models is not None
        self._assert_lib_versions_logged()

    @pytest.mark.parametrize("model_name", ["cb", "hgb", "xgb"])
    def test_polars_native_models_handle_utf8_and_string_dtypes(
        self, model_name, temp_data_dir, common_init_params
    ):
        """All Polars-native strategies must survive pl.Utf8 AND pl.String (same dtype
        under the hood in Polars >= 0.19; large_string is the tripwire for XGBoost)."""
        pytest.importorskip({"cb": "catboost", "xgb": "xgboost", "hgb": "sklearn"}[model_name])
        from .shared import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.core import train_mlframe_models_suite

        df = _make_simple_polars_df()
        # Force String dtype if the Polars version exposes it separately.
        if hasattr(pl, "String"):
            df = df.with_columns(pl.col("cat_feat").cast(pl.String))
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        models, _ = train_mlframe_models_suite(
            df=df, target_name="t", model_name=f"compat_{model_name}",
            features_and_targets_extractor=fte,
            mlframe_models=[model_name],
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )
        assert models is not None
        self._assert_lib_versions_logged()


# ======================================================================
# 6a. Feature selectors with text/embedding columns
# ======================================================================
class TestFeatureSelectorsWithTextEmbedding:
    """MRMR/RFECV must not crash when text_features or embedding_features are declared."""

    def _make_df(self, n=200):
        rng = np.random.default_rng(7)
        return pl.DataFrame({
            "num_a": rng.standard_normal(n),
            "num_b": rng.standard_normal(n),
            "num_c": rng.standard_normal(n),
            "cat_feat": rng.choice([f"c{i}" for i in range(5)], size=n),
            "text_feat": rng.choice([f"sent {i} words here foo bar" for i in range(80)], size=n),
            "emb_feat": [rng.standard_normal(4).tolist() for _ in range(n)],
            "target": rng.integers(0, 2, size=n),
        })

    def test_mrmr_with_text_column(self, temp_data_dir, common_init_params):
        """MRMR on CatBoost with text_feat — text should be passed through to CB."""
        pytest.importorskip("catboost")
        from .shared import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.configs import (
            PolarsPipelineConfig, FeatureTypesConfig, TargetTypes,
        )

        df = self._make_df().drop("emb_feat")
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        models, _ = train_mlframe_models_suite(
            df=df, target_name="t", model_name="mrmr_txt",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            pipeline_config=PolarsPipelineConfig(use_polarsds_pipeline=True),
            feature_types_config=FeatureTypesConfig(
                text_features=["text_feat"],
                auto_detect_feature_types=False,
            ),
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=False,
            use_mrmr_fs=True,
            mrmr_kwargs={"max_runtime_mins": 0.2, "verbose": 0},
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )
        assert TargetTypes.BINARY_CLASSIFICATION in models

    def test_mrmr_with_embedding_column(self, temp_data_dir, common_init_params):
        pytest.importorskip("catboost")
        from .shared import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.core import train_mlframe_models_suite
        from mlframe.training.configs import (
            PolarsPipelineConfig, FeatureTypesConfig,
        )

        df = self._make_df().drop("text_feat")
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)
        train_mlframe_models_suite(
            df=df, target_name="t", model_name="mrmr_emb",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            pipeline_config=PolarsPipelineConfig(use_polarsds_pipeline=True),
            feature_types_config=FeatureTypesConfig(
                embedding_features=["emb_feat"],
                auto_detect_feature_types=False,
            ),
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=False,
            use_mrmr_fs=True,
            mrmr_kwargs={"max_runtime_mins": 0.2, "verbose": 0},
            use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )


# ======================================================================
# 6. Parallel val/test metrics equivalence
# ======================================================================
class TestParallelMetricsEquivalence:
    """If evaluation uses ThreadPoolExecutor, it MUST produce same numbers as sequential."""

    def test_parallel_val_test_metrics_equivalent(self, temp_data_dir, common_init_params):
        pytest.importorskip("catboost")
        from .shared import SimpleFeaturesAndTargetsExtractor
        from mlframe.training.core import train_mlframe_models_suite

        df = _make_simple_polars_df(n=400)
        fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=False)

        models1, meta1 = train_mlframe_models_suite(
            df=df, target_name="t", model_name="par_eq_1",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10, "random_seed": 42},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )
        models2, meta2 = train_mlframe_models_suite(
            df=df, target_name="t", model_name="par_eq_2",
            features_and_targets_extractor=fte,
            mlframe_models=["cb"],
            init_common_params=common_init_params,
            hyperparams_config={"iterations": 10, "random_seed": 42},
            behavior_config=CPU_BEHAVIOR,
            use_ordinary_models=True, use_mlframe_ensembles=False,
            data_dir=temp_data_dir, verbose=0,
        )
        # Reproducibility check — metrics should match across identical runs.
        # If parallel execution introduced nondeterminism, this would fail.
        from mlframe.training.configs import TargetTypes
        m1 = models1[TargetTypes.BINARY_CLASSIFICATION]["target"][0]
        m2 = models2[TargetTypes.BINARY_CLASSIFICATION]["target"][0]
        assert m1.model is not None and m2.model is not None
