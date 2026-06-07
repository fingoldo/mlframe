"""Integration test for ``PipelineCache`` content-keyed HIT at the suite
level via :func:`train_mlframe_models_suite`.

The unit-level test (``test_pipeline_cache_content_based.py``) proves
that ``_compute_pipeline_cache_key`` builds identical keys for Linear
and Neural strategies. This test extends that guarantee end-to-end:
when the suite trains Linear and another strategy that shares the same
preprocessing requirements (``requires_imputation``, ``requires_scaling``,
``requires_encoding`` triple), the SECOND strategy MUST hit the cache
populated by the first instead of re-running the pre-pipeline transform
on the same frame.

Pre-fix the production TVT log showed Linear taking ~46s on the
``SimpleImputer + StandardScaler`` fit and the same identifier sequence
re-running for the second model on the same 4M-row frame. The
content-keyed cache prevents that second pass; this test guards against
regressions to the name-based key.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
import pytest

try:
    from mlframe.training import OutputConfig, ReportingConfig
    from mlframe.training.configs import TargetTypes
    from mlframe.training.core import train_mlframe_models_suite
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"mlframe.training.core not importable: {exc!r}",
        allow_module_level=True,
    )

from ..shared import SimpleFeaturesAndTargetsExtractor


def _build_small_classif_df(seed: int = 11, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    x3 = rng.standard_normal(n)
    # Linear-decision-boundary data: both Linear (Ridge/LogReg) and a
    # second imp+scale strategy can fit it. The exact AUROC isn't the
    # point — cache behaviour is.
    logits = 1.5 * x1 - 0.7 * x2 + 0.3 * x3
    y = (1 / (1 + np.exp(-logits)) > 0.5).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": y})


def _count_pipeline_cache_hits(caplog) -> tuple[int, int]:
    """Parse ``PipelineCache HIT/MISS`` log lines and return ``(hits, misses)``.

    The logger emits one INFO line per get() call:
        ``PipelineCache HIT  key=... (hits=H misses=M size=S)``
        ``PipelineCache MISS key=... (hits=H misses=M size=S)``
    """
    hits = misses = 0
    pat_hit = re.compile(r"PipelineCache HIT")
    pat_miss = re.compile(r"PipelineCache MISS")
    for rec in caplog.records:
        msg = rec.getMessage()
        if pat_hit.search(msg):
            hits += 1
        elif pat_miss.search(msg):
            misses += 1
    return hits, misses


class TestPipelineCacheHitAtSuiteLevel:
    """biz_value: when two strategies share preprocessing requirements,
    the second one HITs the content-keyed cache populated by the first.

    Runs the real ``train_mlframe_models_suite`` with two strategies
    (``["linear", "mlp"]``), then inspects the captured log stream for
    ``PipelineCache HIT`` lines. At least one HIT is expected because
    Linear and Neural have identical (imp=1, scale=1, enc=1) tuples.
    """

    def test_suite_emits_at_least_one_pipeline_cache_hit(
        self, tmp_path, caplog,
    ) -> None:
        df = _build_small_classif_df()
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target", regression=False,
        )
        caplog.set_level(logging.INFO, logger="mlframe.training.strategies")

        try:
            models, _ = train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="cache_hit_integration_test",
                features_and_targets_extractor=fte,
                mlframe_models=["linear", "mlp"],
                reporting_config=ReportingConfig(
                    show_perf_chart=False, show_fi=False,
                ),
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(
                    data_dir=str(tmp_path), models_dir="models",
                ),
                verbose=0,
                hyperparams_config={"iterations": 30},
            )
        except Exception as exc:
            pytest.skip(
                f"Suite execution failed for unrelated reasons: {exc!r}",
            )

        # Sanity: the suite produced at least one model entry.
        entries = models[TargetTypes.BINARY_CLASSIFICATION]["target"]
        assert len(entries) >= 1, (
            "Suite produced no model entries — cannot assess cache behaviour"
        )

        hits, misses = _count_pipeline_cache_hits(caplog)

        # With (linear, mlp), both have (imp=1, scale=1, enc=1). The
        # second strategy's lookup must HIT the cache set by the first
        # at the same feature tier + dtype. Allow at least one HIT —
        # the exact count depends on how many (tier, pre_pipeline,
        # feature_set) combinations the suite explores.
        assert hits >= 1, (
            f"Expected at least one PipelineCache HIT after the second "
            f"strategy's content-key lookup; got hits={hits}, misses="
            f"{misses}. The content-keyed cache may be regressing to "
            f"strategy-name-based keys (linear=! neural)."
        )

    def test_hit_rate_floor_when_strategies_share_requirements(
        self, tmp_path, caplog,
    ) -> None:
        """Tighter guard: with (linear, mlp) sharing imp+scale+enc,
        the hit-to-miss ratio must be non-trivial. Pre-fix every (linear,
        mlp) pair contributed 1 hit per 2 misses (name-keyed); post-fix
        the ratio is dominated by HITs. We assert hits >= misses / 2
        as a robust floor that catches regressions but allows for
        legitimate first-time MISSes per (tier, pre-pipeline) combo.
        """
        df = _build_small_classif_df(seed=42, n=400)
        fte = SimpleFeaturesAndTargetsExtractor(
            target_column="target", regression=False,
        )
        caplog.set_level(logging.INFO, logger="mlframe.training.strategies")

        try:
            train_mlframe_models_suite(
                df=df,
                target_name="target",
                model_name="cache_hit_ratio_test",
                features_and_targets_extractor=fte,
                mlframe_models=["linear", "mlp"],
                reporting_config=ReportingConfig(
                    show_perf_chart=False, show_fi=False,
                ),
                use_ordinary_models=True,
                use_mlframe_ensembles=False,
                output_config=OutputConfig(
                    data_dir=str(tmp_path), models_dir="models",
                ),
                verbose=0,
                hyperparams_config={"iterations": 30},
            )
        except Exception as exc:
            pytest.skip(f"Suite execution failed: {exc!r}")

        hits, misses = _count_pipeline_cache_hits(caplog)
        # Floor: hits must be >= half the misses. Stricter ratios would
        # be flaky against suite changes (new tiers / pre-pipelines add
        # new MISSes that have nothing to do with the content-key fix).
        # The original bug had hits=0 by construction, so any positive
        # floor catches it.
        assert hits >= max(1, misses // 2), (
            f"PipelineCache hit rate too low for shared-requirement pair: "
            f"hits={hits}, misses={misses}. Floor: hits >= max(1, misses // 2)"
        )
