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

    @staticmethod
    def _content_key(s, cat_features):
        # Mirror the in-source EFFECTIVE-encoding content key: the encoding bit is ``requires_encoding AND there are cats to encode`` (see
        # _phase_train_one_target_body.py). On an all-numeric frame the target-encoder is a no-op, so the bit folds to 0 regardless of the flag.
        eff_enc = bool(s.requires_encoding) and bool(cat_features)
        return f"imp{int(s.requires_imputation)}_scale{int(s.requires_scaling)}_enc{int(eff_enc)}"

    def test_content_key_matches_for_linear_and_neural_on_numeric_frame(self) -> None:
        # All-numeric frame (cat_features empty): Linear (requires_encoding True) and the MLP with learnable cat embeddings (False) produce the
        # IDENTICAL imp+scale frame, so they MUST share the content key -> one cache slot, no redundant pre-pipeline pass for the second model.
        from mlframe.training.strategies import (
            LinearModelStrategy,
            NeuralNetStrategy,
        )

        lin = LinearModelStrategy()
        neu = NeuralNetStrategy()
        assert lin.requires_imputation == neu.requires_imputation
        assert lin.requires_scaling == neu.requires_scaling
        assert self._content_key(lin, ()) == self._content_key(neu, ()), (
            f"On an all-numeric frame Linear and Neural must share the content key; "
            f"got linear={self._content_key(lin, ())!r}, neural={self._content_key(neu, ())!r}"
        )

    def test_content_key_differs_for_linear_and_neural_with_cats(self) -> None:
        # With categorical columns present, Linear target-encodes them while the MLP (learnable cat embeddings, the default) keeps them raw, so
        # the produced frames DIFFER -> distinct content keys -> the MLP never receives Linear's target-encoded frame.
        from mlframe.training.strategies import (
            LinearModelStrategy,
            NeuralNetStrategy,
        )

        lin = LinearModelStrategy()
        neu = NeuralNetStrategy()
        assert neu.use_learnable_cat_embeddings is True
        assert lin.requires_encoding is True and neu.requires_encoding is False
        assert self._content_key(lin, ("c1",)) != self._content_key(neu, ("c1",))

    def test_tree_and_hgb_have_distinct_content_keys(self) -> None:
        """Sanity: strategies with DIFFERENT preprocessing requirements
        still get DIFFERENT cache keys (otherwise we over-merge)."""
        from mlframe.training.strategies import (
            TreeModelStrategy,
            HGBStrategy,
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
            LinearModelStrategy,
            NeuralNetStrategy,
        )

        lin = LinearModelStrategy()
        neu = NeuralNetStrategy()

        def _content_key(s, cat_features):
            # EFFECTIVE-encoding content key (mirrors production): the encoding bit is ``requires_encoding AND there are cats to encode``.
            eff_enc = bool(s.requires_encoding) and bool(cat_features)
            return f"imp{int(s.requires_imputation)}_scale{int(s.requires_scaling)}_enc{int(eff_enc)}"

        def _key(s, cat_features):
            return _compute_pipeline_cache_key(
                _content_key(s, cat_features),
                pre_pipeline_name="imp_scaler",
                feature_tier=(False, False),
                supports_polars=False,
                cat_features=cat_features,
                text_features=(),
                embedding_features=(),
            )

        # All-numeric frame: identical content-keyed lookups (Linear's target-encoder is a no-op when there are no cats), so the second
        # strategy HITs the first's slot. With cats, Linear encodes / the MLP keeps raw -> distinct keys (covered by the next class).
        assert _key(lin, ()) == _key(neu, ()), (
            f"content-keyed cache lookups MUST match for Linear and Neural on an all-numeric frame. Got: lin={_key(lin, ())!r}, neu={_key(neu, ())!r}"
        )
        assert _key(lin, ("c1",)) != _key(neu, ("c1",))

    def test_strategies_with_different_requirements_get_distinct_keys(self) -> None:
        """Inverse guard: TreeModelStrategy (no imp/scale/enc) and
        LinearModelStrategy (full preprocessing) must produce DIFFERENT
        cache keys, otherwise we'd over-merge unrelated tier frames."""
        from mlframe.training.core._phase_train_one_target import (
            _compute_pipeline_cache_key,
        )
        from mlframe.training.strategies import (
            LinearModelStrategy,
            TreeModelStrategy,
        )

        lin = LinearModelStrategy()
        tree = TreeModelStrategy()

        def _content_key(s):
            return f"imp{int(s.requires_imputation)}_scale{int(s.requires_scaling)}_enc{int(s.requires_encoding)}"

        lin_key = _compute_pipeline_cache_key(
            _content_key(lin),
            "imp_scaler",
            (False, False),
            False,
            (),
            (),
            (),
        )
        tree_key = _compute_pipeline_cache_key(
            _content_key(tree),
            "imp_scaler",
            (False, False),
            False,
            (),
            (),
            (),
        )
        assert lin_key != tree_key, f"strategies with different preprocessing requirements MUST produce distinct cache keys; got both = {lin_key!r}"


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
            LinearModelStrategy,
            NeuralNetStrategy,
        )
        from mlframe.training.strategies import PipelineCache

        cache = PipelineCache(verbose=False)
        lin = LinearModelStrategy()
        neu = NeuralNetStrategy()

        def _content_key(s, cat_features):
            eff_enc = bool(s.requires_encoding) and bool(cat_features)
            return f"imp{int(s.requires_imputation)}_scale{int(s.requires_scaling)}_enc{int(eff_enc)}"

        # All-numeric (cat_features empty): Linear and Neural share the content key (Linear's target-encoder is a no-op with no cats), so the
        # second strategy HITs the first's slot -- the original cache-sharing win, preserved despite the MLP's learnable-cat-embedding default.
        lin_key = _compute_pipeline_cache_key(
            _content_key(lin, ()),
            "imp_scaler",
            (False, False),
            False,
            (),
            (),
            (),
        )
        cache.set(lin_key, "fake_train_df", "fake_val_df", "fake_test_df")
        assert cache.n_misses == 0
        assert cache.n_hits == 0

        neu_key = _compute_pipeline_cache_key(
            _content_key(neu, ()),
            "imp_scaler",
            (False, False),
            False,
            (),
            (),
            (),
        )
        result = cache.get(neu_key)
        assert result is not None, "PipelineCache MISS for neural after linear stored under the same content key - content-keyed cache is broken"
        assert cache.n_hits == 1
        assert cache.n_misses == 0
        train, val, test = result
        assert train == "fake_train_df", "wrong frame returned from cache"
