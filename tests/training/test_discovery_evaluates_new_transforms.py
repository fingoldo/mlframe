"""End-to-end Pack J/K test: ``CompositeTargetDiscovery.fit`` actually evaluates the new unary + chain transforms.

After registering ``cbrt_y / log_y / yeo_johnson_y / quantile_normal_y`` and ``chain_linres_*`` / ``chain_monres_*`` and extending ``CompositeTargetDiscoveryConfig.transforms`` default to 14, this test runs the real ``CompositeTargetDiscovery.fit`` on a synthetic dataset and verifies:

1. Every one of the 14 transforms appears in ``discovery.report_`` (the per-candidate evaluation log) -- proves the discovery loop actually tried them.
2. The dedup logic skips re-evaluating unary transforms per base (unary names appear exactly once across all bases, NOT per-base).
3. The discovery completes without crashing on a heavy-tail-residual target -- the production failure mode this entire wave addresses.

Whether the chain composites WIN over the bivariate parents is data-dependent (the raw-y baseline gate makes that call), so we only assert the candidates were CONSIDERED, not that they were KEPT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.training.composite.discovery import CompositeTargetDiscovery
from mlframe.training.configs import CompositeTargetDiscoveryConfig


EXPECTED_TRANSFORMS = frozenset(
    {
        "diff",
        "ratio",
        "logratio",
        "linear_residual",
        "quantile_residual",
        "monotonic_residual",
        "cbrt_y",
        "log_y",
        "yeo_johnson_y",
        "quantile_normal_y",
        "chain_linres_cbrt",
        "chain_linres_yj",
        "chain_monres_cbrt",
        "chain_monres_yj",
    }
)


@pytest.fixture
def synthetic_heavy_tail_df() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """y = 0.8 * x_base + 5 + Laplace(scale=3) -- production-shape signal + heavy-tail residual."""
    rng = np.random.default_rng(20260518)
    n = 3000
    df = pd.DataFrame(
        {
            "x_base": rng.normal(100.0, 20.0, n),
            "x_a": rng.normal(0.0, 1.0, n),
            "x_b": rng.normal(0.0, 1.0, n),
            "x_c": rng.normal(0.0, 1.0, n),
            "x_d": rng.normal(0.0, 1.0, n),
        }
    )
    resid = rng.laplace(0.0, 3.0, n)
    df["y"] = 0.8 * df["x_base"] + 5.0 + resid + 0.5 * df["x_a"]
    train_idx = np.arange(int(0.8 * n))
    val_idx = np.arange(int(0.8 * n), n)
    return df, train_idx, val_idx


class TestDiscoveryEvaluatesNewTransforms:
    def test_discovery_runs_without_crash(
        self,
        synthetic_heavy_tail_df: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        df, train_idx, val_idx = synthetic_heavy_tail_df
        cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=1000)
        disc = CompositeTargetDiscovery(config=cfg)
        disc.fit(
            df=df,
            target_col="y",
            feature_cols=["x_base", "x_a", "x_b", "x_c", "x_d"],
            train_idx=train_idx,
            val_idx=val_idx,
        )
        # ``specs_`` may legitimately be empty if the data is too easy / too noisy; we only check the discovery completed without raising.
        assert hasattr(disc, "specs_")
        assert hasattr(disc, "report_")

    def test_all_14_transforms_evaluated(
        self,
        synthetic_heavy_tail_df: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        df, train_idx, val_idx = synthetic_heavy_tail_df
        cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=1000)
        disc = CompositeTargetDiscovery(config=cfg)
        disc.fit(
            df=df,
            target_col="y",
            feature_cols=["x_base", "x_a", "x_b", "x_c", "x_d"],
            train_idx=train_idx,
            val_idx=val_idx,
        )
        evaluated = {r.get("transform_name") for r in disc.report_ if isinstance(r, dict) and r.get("transform_name")}
        missing = EXPECTED_TRANSFORMS - evaluated
        assert not missing, f"Discovery did NOT evaluate {sorted(missing)}; got {sorted(evaluated)}"

    def test_unary_dedup_runs_each_unary_once(
        self,
        synthetic_heavy_tail_df: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Per-base dedup: unary transforms (``requires_base=False``) must appear in the report exactly once each, NOT once per base candidate."""
        df, train_idx, val_idx = synthetic_heavy_tail_df
        cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=1000)
        disc = CompositeTargetDiscovery(config=cfg)
        disc.fit(
            df=df,
            target_col="y",
            feature_cols=["x_base", "x_a", "x_b", "x_c", "x_d"],
            train_idx=train_idx,
            val_idx=val_idx,
        )
        from collections import Counter

        counts = Counter(r.get("transform_name") for r in disc.report_ if isinstance(r, dict))
        for unary in ("cbrt_y", "log_y", "yeo_johnson_y", "quantile_normal_y"):
            assert counts[unary] == 1, f"unary '{unary}' evaluated {counts[unary]} times -- dedup did not apply"

    def test_chain_transforms_evaluated_per_base(
        self,
        synthetic_heavy_tail_df: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    ) -> None:
        """Chain transforms still need a base column, so they SHOULD be evaluated per candidate base (unlike unary). Count >= 1."""
        df, train_idx, val_idx = synthetic_heavy_tail_df
        cfg = CompositeTargetDiscoveryConfig(enabled=True, mi_sample_n=1000)
        disc = CompositeTargetDiscovery(config=cfg)
        disc.fit(
            df=df,
            target_col="y",
            feature_cols=["x_base", "x_a", "x_b", "x_c", "x_d"],
            train_idx=train_idx,
            val_idx=val_idx,
        )
        from collections import Counter

        counts = Counter(r.get("transform_name") for r in disc.report_ if isinstance(r, dict))
        for chain in (
            "chain_linres_cbrt",
            "chain_linres_yj",
            "chain_monres_cbrt",
            "chain_monres_yj",
        ):
            assert counts[chain] >= 1, f"chain '{chain}' not evaluated at all"
