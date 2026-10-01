"""cProfile + per-chart-timer harness for the post-fit report path of ``train_mlframe_models_suite``.

Run:  ``python -m mlframe.reporting._benchmarks.profile_suite_render --task binary --n 60000 --models cb lgb --async-render off --profile``

Prints wall time of the whole suite, the ``Metrics computation`` time, the per-chart-type render table and (with ``--profile``) the
top cumulative functions. ``--async-render on|off|auto`` toggles ``ReportingConfig.async_render``; ``--out DIR`` keeps the artifacts.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import os
import pstats
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def make_frame(task: str, n: int, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame: 8 numerics (a few informative), 2 categoricals, a timestamp column, target by ``task``."""
    rng = np.random.default_rng(seed)
    data: Dict[str, Any] = {f"num_{i}": rng.standard_normal(n).astype("float32") for i in range(8)}
    data["cat_0"] = pd.Categorical(rng.choice(["A", "B", "C", "D", "E"], n))
    data["cat_1"] = pd.Categorical(rng.choice(["x", "y", "z"], n))
    signal = data["num_0"] - 0.5 * data["num_1"] + 0.3 * data["num_2"] * data["num_3"]
    if task == "regression":
        data["target"] = signal + rng.standard_normal(n) * 0.5
    elif task == "multiclass":
        s = signal + rng.standard_normal(n) * 0.7
        data["target"] = np.digitize(s, np.quantile(s, [0.33, 0.66])).astype("int32")
    else:
        data["target"] = (signal + rng.standard_normal(n) * 0.8 > 0).astype("int32")
    return pd.DataFrame(data)


def dir_manifest(root: str) -> Dict[str, str]:
    """Relative path -> sha1 of every file under ``root`` (PNG/HTML content hash)."""
    out: Dict[str, str] = {}
    for base, _dirs, files in os.walk(root):
        for f in files:
            p = os.path.join(base, f)
            if os.sep + "models" + os.sep in p or p.endswith((".dump", ".joblib", ".pkl", ".cbm", ".zst")):
                continue
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha1(fh.read(), usedforsecurity=False).hexdigest()
    return out


def run_suite(task: str, n: int, models: List[str], async_render: Any, out_dir: str, iterations: int = 30, extra_reporting: Any = None, with_ts: bool = True, val_frac: float = 0.3, test_frac: float = 0.34):
    """Run one suite and return ``(wall_seconds, (models, metadata))``."""
    from mlframe.training.configs import BaselineDiagnosticsConfig, OutputConfig, PreprocessingConfig, ReportingConfig, TrainingSplitConfig
    from mlframe.training.core import train_mlframe_models_suite
    from mlframe.training.extractors import SimpleFeaturesAndTargetsExtractor

    df = make_frame(task, n)
    if with_ts:
        df["ts"] = pd.date_range("2022-01-01", periods=n, freq="min")
    if task == "regression":
        fte = SimpleFeaturesAndTargetsExtractor(regression_targets=["target"], ts_field="ts" if with_ts else None)
    else:
        fte = SimpleFeaturesAndTargetsExtractor(classification_targets=["target"], ts_field="ts" if with_ts else None)
    hp: Dict[str, Any] = {"iterations": iterations}
    if "cb" in models:
        hp["cb_kwargs"] = {"task_type": "CPU", "verbose": 0, "thread_count": 2}
    if "lgb" in models:
        hp["lgb_kwargs"] = {"device_type": "cpu", "verbose": -1, "n_jobs": 2}
    if "xgb" in models:
        hp["xgb_kwargs"] = {"device": "cpu", "verbosity": 0, "n_jobs": 2}
    rep_kwargs: Dict[str, Any] = dict(show_perf_chart=False, show_fi=True)
    if async_render is not None and "async_render" in ReportingConfig.model_fields:
        rep_kwargs["async_render"] = async_render
    if extra_reporting:
        rep_kwargs.update(extra_reporting)
    t0 = time.perf_counter()
    res = train_mlframe_models_suite(
        df=df, target_name="tgt", model_name="mdl", features_and_targets_extractor=fte, mlframe_models=list(models), hyperparams_config=hp,
        preprocessing_config=PreprocessingConfig(drop_columns=[]), use_ordinary_models=True, use_mlframe_ensembles=False,
        output_config=OutputConfig(data_dir=out_dir, models_dir="models", save_charts=True),
        reporting_config=ReportingConfig(**rep_kwargs), verbose=1,
        split_config=TrainingSplitConfig(val_size=val_frac, test_size=test_frac), baseline_diagnostics_config=BaselineDiagnosticsConfig(enabled=False),
    )
    return time.perf_counter() - t0, res


def main(argv: List[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="binary", choices=["binary", "multiclass", "regression"])
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--models", nargs="+", default=["cb"])
    ap.add_argument("--async-render", default="off", choices=["on", "off", "auto"])
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--top", type=int, default=35)
    a = ap.parse_args(argv)
    from mlframe.reporting.renderers._render_timings import chart_timings_snapshot

    out = a.out or tempfile.mkdtemp(prefix="mlframe_render_prof_")
    os.makedirs(out, exist_ok=True)
    ar: Any = {"on": True, "off": False, "auto": "auto"}[a.async_render]
    prof = cProfile.Profile() if a.profile else None
    if prof:
        prof.enable()
    wall, _res = run_suite(a.task, a.n, a.models, ar, out)
    if prof:
        prof.disable()
    print(f"\nSUITE WALL: {wall:.2f}s  out={out}")
    rows = chart_timings_snapshot()
    print(f"chart render total (sum over threads): {sum(r['seconds'] for r in rows):.2f}s across {sum(r['count'] for r in rows)} renders")
    by_backend: Dict[str, float] = {}
    for r in rows:
        tag = "plotly" if str(r["chart"]).endswith("[plotly]") else ("matplotlib" if str(r["chart"]).endswith("[matplotlib]") else "other")
        by_backend[tag] = by_backend.get(tag, 0.0) + float(r["seconds"])
    print("render seconds by backend: " + ", ".join(f"{k}={v:.2f}s" for k, v in sorted(by_backend.items())))
    for r in rows[:25]:
        print(f"  {r['seconds']:8.3f}s  x{r['count']:<3d} avg {r['avg_seconds']:.3f}  {r['chart']}")
    if prof:
        pstats.Stats(prof).sort_stats("cumulative").print_stats(a.top)
    if not a.out:
        shutil.rmtree(out, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
