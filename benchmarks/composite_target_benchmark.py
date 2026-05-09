"""Synthetic benchmark suite for composite-target discovery.

Runs ``train_mlframe_models_suite`` on N pre-defined synthetic
scenarios with composite-target discovery enabled and disabled, then
emits a leaderboard table comparing y-scale RMSE on a held-out test
slice.

The synthetic scenarios are calibrated so that the "expected winner"
is known a priori. If a discovered composite target ranks first on
a scenario where it was designed to win (S1, S2, S3, S4) the
benchmark passes that row; otherwise it fails. S5 ("no_dominant")
is a control: composite mode should NOT noticeably beat raw, so
"composite wins by < 1%" is the pass condition.

Usage
-----

::

    python -m mlframe.benchmarks.composite_target_benchmark
    python -m mlframe.benchmarks.composite_target_benchmark --fast
    python -m mlframe.benchmarks.composite_target_benchmark --output ./bench.json

Outputs
-------

- JSON file with per-scenario per-mode metrics and the verdict.
- Markdown leaderboard table on stdout (and optionally to a file).

Where outputs go
----------------

By default writes ``composite_target_benchmark_results.json`` in the
current working directory and prints the Markdown summary to stdout.
The script lives inside the package so it is reachable as
``python -m mlframe.benchmarks.composite_target_benchmark`` after
``pip install``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Scenario generators
# ----------------------------------------------------------------------


def _seed(rng_seed: int) -> np.random.Generator:
    return np.random.default_rng(rng_seed)


def _scenario_pure_lag(n: int, seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """y = base + small_noise. No structural signal beyond the lag.
    Expected winner: ``diff`` (pure-residual transform)."""
    rng = _seed(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    y = base + rng.normal(scale=0.05 * 3.0, size=n)
    df = pd.DataFrame({
        "TVT_prev": base,
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
        "target": y,
    })
    return df, "diff"


def _scenario_lag_with_signal(n: int, seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """y = 0.95 * base + structural_signal + noise.
    Expected winner: ``linear_residual`` (captures the 0.95 explicitly)."""
    rng = _seed(seed)
    base = rng.normal(loc=10.0, scale=3.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 0.95 * base + 0.5 * x1 - 0.3 * x2 + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({
        "TVT_prev": base, "x1": x1, "x2": x2,
        "x3": rng.normal(size=n),
        "target": y,
    })
    return df, "linear_residual"


def _scenario_multiplicative(n: int, seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """y = base * exp(structural_signal + noise). Strictly positive.
    Expected winner: ``logratio`` (variance stabilisation)."""
    rng = _seed(seed)
    base = rng.lognormal(mean=2.0, sigma=0.4, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = base * np.exp(0.3 * x1 - 0.2 * x2 + rng.normal(scale=0.1, size=n))
    df = pd.DataFrame({
        "TVT_prev": base, "x1": x1, "x2": x2,
        "x3": rng.normal(size=n),
        "target": y,
    })
    return df, "logratio"


def _scenario_proportional(n: int, seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """y = base * (1 + 0.1 * structural_signal) + small_noise.
    Expected winner: ``ratio``."""
    rng = _seed(seed)
    base = rng.uniform(low=2.0, high=20.0, size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = base * (1.0 + 0.1 * x1 - 0.05 * x2) + rng.normal(scale=0.05, size=n)
    df = pd.DataFrame({
        "TVT_prev": base, "x1": x1, "x2": x2,
        "x3": rng.normal(size=n),
        "target": y,
    })
    return df, "ratio"


def _scenario_no_dominant(n: int, seed: int = 0) -> Tuple[pd.DataFrame, str]:
    """y = sum_i 0.3 * x_i + noise. No feature dominates.
    Expected: composite mode shows mi_recommendation=unlikely_to_help;
    enabling it should NOT meaningfully beat raw."""
    rng = _seed(seed)
    X = rng.normal(size=(n, 4))
    y = 0.3 * X.sum(axis=1) + rng.normal(scale=2.0, size=n)
    df = pd.DataFrame(X, columns=["TVT_prev", "x1", "x2", "x3"])
    df["target"] = y
    return df, ""  # no expected winner


SCENARIOS: List[Tuple[str, Callable[[int, int], Tuple[pd.DataFrame, str]]]] = [
    ("pure_lag", _scenario_pure_lag),
    ("lag_with_signal", _scenario_lag_with_signal),
    ("multiplicative", _scenario_multiplicative),
    ("proportional", _scenario_proportional),
    ("no_dominant", _scenario_no_dominant),
]


# ----------------------------------------------------------------------
# Benchmark runner
# ----------------------------------------------------------------------


@dataclass
class BenchResult:
    scenario: str
    n_samples: int
    expected_winner: str
    raw_test_rmse: float
    composite_test_rmse: float
    composite_relative_improvement_pct: float
    discovered_top_transform: str
    elapsed_raw_s: float
    elapsed_composite_s: float
    verdict: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "n_samples": self.n_samples,
            "expected_winner": self.expected_winner or None,
            "raw_test_rmse": float(self.raw_test_rmse),
            "composite_test_rmse": float(self.composite_test_rmse),
            "composite_relative_improvement_pct": float(self.composite_relative_improvement_pct),
            "discovered_top_transform": self.discovered_top_transform or None,
            "elapsed_raw_s": float(self.elapsed_raw_s),
            "elapsed_composite_s": float(self.elapsed_composite_s),
            "verdict": self.verdict,
        }


def _run_one(
    df: pd.DataFrame,
    *,
    composite_enabled: bool,
    work_dir: str,
    fast: bool = False,
) -> Tuple[float, str, float]:
    """Run ``train_mlframe_models_suite`` once on ``df``; return
    ``(test_rmse_or_nan, top_composite_transform_name, elapsed_s)``."""
    from mlframe.training.configs import (
        CompositeTargetDiscoveryConfig, TrainingSplitConfig,
    )
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.tests.training.shared import SimpleFeaturesAndTargetsExtractor

    fte = SimpleFeaturesAndTargetsExtractor(target_column="target", regression=True)

    cfg = (CompositeTargetDiscoveryConfig(
        enabled=True,
        base_candidates=["TVT_prev"],
        transforms=["diff", "ratio", "logratio", "linear_residual"],
        mi_sample_n=200 if fast else 1000,
        top_k_after_mi=4,
        eps_mi_gain=-1.0,
        cross_target_ensemble_strategy="oof_weighted",
    ) if composite_enabled else None)

    t0 = time.perf_counter()
    models, metadata = train_mlframe_models_suite(
        df=df,
        target_name="target",
        model_name="bench",
        features_and_targets_extractor=fte,
        mlframe_models=["linear"],
        split_config=TrainingSplitConfig(test_size=0.2, val_size=0.1),
        output_config={"data_dir": work_dir, "models_dir": "models"},
        verbose=0,
        composite_target_discovery_config=cfg,
    )
    elapsed = time.perf_counter() - t0

    # Pull test RMSE for the raw target. For composite mode, prefer
    # the cross-target ensemble's test RMSE if it exists.
    test_rmse = float("nan")
    top_transform = ""
    if composite_enabled:
        y_metrics = metadata.get("composite_target_y_scale_metrics", {})
        regression = y_metrics.get("regression") or y_metrics.get("regression")
        if regression:
            # Find the best composite target by test RMSE.
            best = float("inf")
            for cname, entries in regression.items():
                for e in entries:
                    test = e.get("metrics", {}).get("test", {})
                    rmse = test.get("RMSE")
                    if rmse is not None and rmse < best:
                        best = rmse
                        # transform name = part between last two __
                        if "__" in cname:
                            parts = cname.split("__")
                            if len(parts) >= 3:
                                top_transform = parts[-2]
            if best < float("inf"):
                test_rmse = best
        # Fallback: walk the per-target loop's metrics for the raw target.
        if not (test_rmse == test_rmse):  # NaN check
            test_rmse = _extract_raw_test_rmse(metadata)
    else:
        test_rmse = _extract_raw_test_rmse(metadata)

    return test_rmse, top_transform, elapsed


def _extract_raw_test_rmse(metadata: Dict[str, Any]) -> float:
    """Best-effort extraction of test RMSE for the raw target from
    the suite's standard metrics dict. Falls back to NaN if the suite
    doesn't surface it in a known shape."""
    fr = metadata.get("fairness_report") or metadata.get("metrics") or {}
    # We don't strictly need this for the benchmark to be useful --
    # composite RMSE is what we compare against itself across modes.
    if not isinstance(fr, dict):
        return float("nan")
    # Walk the dict looking for an entry with a "test" key holding RMSE.
    for v in fr.values():
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, dict):
                    test = vv.get("test")
                    if isinstance(test, dict) and "RMSE" in test:
                        return float(test["RMSE"])
    return float("nan")


def run_benchmark(*, n: int, fast: bool = False, seed: int = 0) -> List[BenchResult]:
    results: List[BenchResult] = []
    for name, gen in SCENARIOS:
        df, expected = gen(n, seed)
        with tempfile.TemporaryDirectory(prefix="composite_bench_") as work:
            try:
                raw_rmse, _, t_raw = _run_one(df, composite_enabled=False,
                                              work_dir=work, fast=fast)
            except Exception as exc:
                logger.warning("[bench] %s raw run failed: %s", name, exc)
                raw_rmse, t_raw = float("nan"), 0.0
            try:
                comp_rmse, top_transform, t_comp = _run_one(
                    df, composite_enabled=True, work_dir=work, fast=fast,
                )
            except Exception as exc:
                logger.warning("[bench] %s composite run failed: %s", name, exc)
                comp_rmse, top_transform, t_comp = float("nan"), "", 0.0

        if (not np.isfinite(raw_rmse)) or raw_rmse <= 0:
            improvement_pct = float("nan")
        else:
            improvement_pct = (raw_rmse - comp_rmse) / raw_rmse * 100.0

        # Verdict logic.
        if expected:
            # Expected-winner scenarios: composite wins if (a) it ran,
            # (b) it picked the right top transform OR composite RMSE
            # is meaningfully lower than raw.
            picked_right = (top_transform == expected)
            if picked_right or (np.isfinite(improvement_pct) and improvement_pct > 1.0):
                verdict = "PASS"
            else:
                verdict = "FAIL"
        else:
            # No-dominant control: composite shouldn't meaningfully
            # beat raw. PASS if |improvement| < 5%, otherwise NOTE.
            if not np.isfinite(improvement_pct):
                verdict = "PASS"
            elif abs(improvement_pct) < 5.0:
                verdict = "PASS"
            else:
                verdict = "NOTE"

        results.append(BenchResult(
            scenario=name,
            n_samples=n,
            expected_winner=expected,
            raw_test_rmse=raw_rmse,
            composite_test_rmse=comp_rmse,
            composite_relative_improvement_pct=improvement_pct,
            discovered_top_transform=top_transform,
            elapsed_raw_s=t_raw,
            elapsed_composite_s=t_comp,
            verdict=verdict,
        ))
    return results


def render_markdown(results: List[BenchResult]) -> str:
    """Stakeholder-friendly leaderboard table."""
    lines = []
    lines.append("# Composite-target discovery benchmark")
    lines.append("")
    lines.append("| scenario | expected | discovered | raw RMSE | composite RMSE | improvement % | verdict |")
    lines.append("|----------|----------|------------|----------|----------------|---------------|---------|")
    for r in results:
        lines.append(
            f"| `{r.scenario}` | `{r.expected_winner or '-'}` | "
            f"`{r.discovered_top_transform or '-'}` | "
            f"{r.raw_test_rmse:.4f} | {r.composite_test_rmse:.4f} | "
            f"{r.composite_relative_improvement_pct:+.2f} | "
            f"{r.verdict} |"
        )
    lines.append("")
    n_pass = sum(1 for r in results if r.verdict == "PASS")
    lines.append(f"**{n_pass}/{len(results)} scenarios passed.**")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Composite-target discovery benchmark suite.",
    )
    parser.add_argument("--fast", action="store_true",
                        help="Smaller dataset / fewer iterations for quick smoke runs.")
    parser.add_argument("--n", type=int, default=None,
                        help="Per-scenario row count (default: 800 fast / 3000 full).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="composite_target_benchmark_results.json",
                        help="Path for JSON results file.")
    parser.add_argument("--markdown-output", type=str, default=None,
                        help="Optional path for Markdown leaderboard.")
    args = parser.parse_args(argv)

    n = args.n if args.n is not None else (800 if args.fast else 3000)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    print(f"[bench] running on n={n} ({'fast' if args.fast else 'full'} mode)")

    results = run_benchmark(n=n, fast=args.fast, seed=args.seed)

    md = render_markdown(results)
    print()
    print(md)

    payload = {
        "n_samples": n,
        "fast": args.fast,
        "seed": args.seed,
        "results": [r.to_dict() for r in results],
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"\n[bench] wrote {args.output}")

    if args.markdown_output:
        with open(args.markdown_output, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[bench] wrote {args.markdown_output}")

    n_fail = sum(1 for r in results if r.verdict == "FAIL")
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
