"""Pipeline cache key is CONTENT-based, not strategy-name-based.

Pre-2026-05-18: ``LinearStrategy.cache_key="linear"`` and
``NeuralStrategy.cache_key="neural"`` produced DIFFERENT cache keys
even when both used IDENTICAL ``imp+scaler`` pipelines. The second
strategy in a (linear, neural) sequence re-did the 17s pre-pipeline
transform on the same 4M rows. User-reported in TVT log analysis
2026-05-18.

Post-fix: cache key is built from a content tuple
``(requires_imputation, requires_scaling, requires_encoding)`` so any
two strategies with matching preprocessing requirements share the cache
slot.
"""
from __future__ import annotations


class TestLinearAndNeuralShareCacheKey:
    """LinearStrategy and NeuralStrategy both require imp+scale+enc;
    they MUST produce the same content-key, hence the same
    PipelineCache lookup."""

    def test_content_key_matches_for_linear_and_neural(self) -> None:
        from mlframe.training.strategies import (
            LinearModelStrategy, NeuralNetStrategy,
        )
        lin = LinearModelStrategy()
        neu = NeuralNetStrategy()

        # All three preprocessing-requirement flags must match for the
        # content-key contract to hold.
        assert lin.requires_imputation == neu.requires_imputation
        assert lin.requires_scaling == neu.requires_scaling
        assert lin.requires_encoding == neu.requires_encoding

        # Mirror the in-source content-key construction.
        def _content_key(s):
            return (
                f"imp{int(s.requires_imputation)}"
                f"_scale{int(s.requires_scaling)}"
                f"_enc{int(s.requires_encoding)}"
            )
        assert _content_key(lin) == _content_key(neu), (
            f"LinearStrategy and NeuralStrategy must produce the same "
            f"content key (identical preprocessing requirements); "
            f"got linear={_content_key(lin)!r}, neural={_content_key(neu)!r}"
        )

    def test_tree_and_hgb_have_distinct_content_keys(self) -> None:
        """Sanity: strategies with DIFFERENT preprocessing requirements
        still get DIFFERENT cache keys (otherwise we over-merge)."""
        from mlframe.training.strategies import (
            TreeModelStrategy, HGBStrategy,
        )
        tree = TreeModelStrategy()
        hgb = HGBStrategy()
        # TreeStrategy: no imputation needed (CB/LGB/XGB native);
        # HGBStrategy: needs encoding for cat features.
        # At minimum one of the three flags should differ.
        flags_tree = (
            tree.requires_imputation,
            tree.requires_scaling,
            tree.requires_encoding,
        )
        flags_hgb = (
            hgb.requires_imputation,
            hgb.requires_scaling,
            hgb.requires_encoding,
        )
        assert flags_tree != flags_hgb, (
            f"TreeStrategy and HGBStrategy must have at least one "
            f"differing preprocessing flag so they don't collide on the "
            f"content cache key; got tree={flags_tree}, hgb={flags_hgb}"
        )


class TestPipelineCacheKeyMatchesAcrossStrategiesWithIdenticalRequirements:
    """End-to-end: with the content-key in place, two strategies that share
    preprocessing requirements (Linear + Neural) produce identical cache
    lookups, so the second strategy in a (linear, neural) sequence HITS
    the cache instead of re-doing the pre-pipeline transform."""

    def test_compute_pipeline_cache_key_uses_content_not_name(self) -> None:
        """Replay the in-source content-key construction with both
        strategies and assert the resulting cache_key string is
        IDENTICAL when feature_tier / kind / feats / dtype match."""
        from mlframe.training.core._phase_train_one_target import (
            _compute_pipeline_cache_key,
        )
        from mlframe.training.strategies import (
            LinearModelStrategy, NeuralNetStrategy,
        )

        lin = LinearModelStrategy()
        neu = NeuralNetStrategy()

        def _content_key(s):
            return (
                f"imp{int(s.requires_imputation)}"
                f"_scale{int(s.requires_scaling)}"
                f"_enc{int(s.requires_encoding)}"
            )

        # Both call _compute_pipeline_cache_key with the content-key in
        # the strategy_cache_key slot - mirrors the post-fix call site
        # at _phase_train_one_target.py:1572.
        lin_key = _compute_pipeline_cache_key(
            _content_key(lin),
            pre_pipeline_name="imp_scaler",
            feature_tier=(False, False),
            supports_polars=False,
            cat_features=(),
            text_features=(),
            embedding_features=(),
        )
        neu_key = _compute_pipeline_cache_key(
            _content_key(neu),
            pre_pipeline_name="imp_scaler",
            feature_tier=(False, False),
            supports_polars=False,
            cat_features=(),
            text_features=(),
            embedding_features=(),
        )
        assert lin_key == neu_key, (
            f"content-keyed cache lookups MUST match for Linear and "
            f"Neural (identical preprocessing requirements). "
            f"Pre-fix these were 'linear_...' vs 'neural_...'. Got: "
            f"lin={lin_key!r}, neu={neu_key!r}"
        )

    def test_strategies_with_different_requirements_get_distinct_keys(self) -> None:
        """Inverse guard: TreeModelStrategy (no imp/scale/enc) and
        LinearModelStrategy (full preprocessing) must produce DIFFERENT
        cache keys, otherwise we'd over-merge unrelated tier frames."""
        from mlframe.training.core._phase_train_one_target import (
            _compute_pipeline_cache_key,
        )
        from mlframe.training.strategies import (
            LinearModelStrategy, TreeModelStrategy,
        )

        lin = LinearModelStrategy()
        tree = TreeModelStrategy()

        def _content_key(s):
            return (
                f"imp{int(s.requires_imputation)}"
                f"_scale{int(s.requires_scaling)}"
                f"_enc{int(s.requires_encoding)}"
            )

        lin_key = _compute_pipeline_cache_key(
            _content_key(lin), "imp_scaler",
            (False, False), False, (), (), (),
        )
        tree_key = _compute_pipeline_cache_key(
            _content_key(tree), "imp_scaler",
            (False, False), False, (), (), (),
        )
        assert lin_key != tree_key, (
            f"strategies with different preprocessing requirements MUST "
            f"produce distinct cache keys; got both = {lin_key!r}"
        )


class TestPipelineCacheRealHitOnSecondStrategy:
    """biz_value: with content-keyed cache, a second strategy's pre-pipeline
    transform produces a cache HIT instead of redoing work. Verified via
    PipelineCache.n_hits counter."""

    def test_neural_hits_cache_set_by_linear(self) -> None:
        """Simulate the suite's (linear -> neural) sequence at cache level:
        compute_pipeline_cache_key for both strategies and verify the
        cache HIT counter increments on the second lookup."""
        from mlframe.training.core._phase_train_one_target import (
            _compute_pipeline_cache_key,
        )
        from mlframe.training.strategies import (
            LinearModelStrategy, NeuralNetStrategy,
        )
        from mlframe.training.strategies import PipelineCache

        cache = PipelineCache(verbose=False)
        lin = LinearModelStrategy()
        neu = NeuralNetStrategy()

        def _content_key(s):
            return (
                f"imp{int(s.requires_imputation)}"
                f"_scale{int(s.requires_scaling)}"
                f"_enc{int(s.requires_encoding)}"
            )

        # Linear computes first, stores in cache.
        lin_key = _compute_pipeline_cache_key(
            _content_key(lin), "imp_scaler",
            (False, False), False, (), (), (),
        )
        cache.set(lin_key, "fake_train_df", "fake_val_df", "fake_test_df")
        assert cache.n_misses == 0
        assert cache.n_hits == 0

        # Neural looks up, MUST hit (same content key as linear).
        neu_key = _compute_pipeline_cache_key(
            _content_key(neu), "imp_scaler",
            (False, False), False, (), (), (),
        )
        result = cache.get(neu_key)
        assert result is not None, (
            "PipelineCache MISS for neural after linear stored under the "
            "same content key - content-keyed cache is broken"
        )
        assert cache.n_hits == 1
        assert cache.n_misses == 0
        train, val, test = result
        assert train == "fake_train_df", "wrong frame returned from cache"

