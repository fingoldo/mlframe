"""Bench the PySR FE upgrade vs legacy defaults.

Run from the repo root with src/ on PYTHONPATH (mlframe uses the src-layout so a
plain ``python -m mlframe...`` fails with ModuleNotFoundError unless the package
is pip-installed in editable mode):

.. code-block:: bash

    PYTHONPATH=src python -m mlframe.training._benchmarks.bench_pysr_fe
    # OR (no env var needed):
    python src/mlframe/training/_benchmarks/bench_pysr_fe.py

Synthetic ground-truth: ``y = 3*sin(x1) + log(|x2|+1) - 0.5*x3**2 + noise``
with 5 extra noise features. Measures, per configuration:

- Wall time (PySR fit only, Julia warm-up amortised by a 10s dummy fit before
  the first measured run).
- Holdout RMSE of the best-scoring equation on a 1000-row held-out split.
- Form rediscovery: did the equation string mention ``sin``, ``log``/``safe_log``,
  ``square``/``x*x``? Each found = 1; max 3.

Configurations:
1. legacy_defaults -- procs=1, no batching, niter=200, precision=32 (mimics the
   pipeline.py defaults BEFORE this audit wave).
2. new_minimal -- new env / batching / multithread defaults + minimal preset.
3. new_standard -- new defaults + standard preset (the in-suite default).
4. new_physics -- new defaults + physics preset.

Output:
- Markdown table to stdout.
- Same table appended to ``tests/perf/results/bench_pysr_fe_<UTC-timestamp>.md``.

Skips with a clear WARN if Julia / PySR isn't installed -- bench is a profiler
not a gate.
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _make_synth(n: int = 5000, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """y = 3*sin(x1) + log(|x2|+1) - 0.5*x3**2 + N(0, 0.3) on 8 features."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 8)).astype(np.float32)
    y = (3.0 * np.sin(x[:, 1]) + np.log(np.abs(x[:, 2]) + 1.0) - 0.5 * x[:, 3] ** 2 + 0.3 * rng.standard_normal(n)).astype(np.float32)
    df = pd.DataFrame(x, columns=[f"x{i}" for i in range(8)])
    return df, y


def _equation_form_score(eq_str: str) -> int:
    """Count how many of {sin, log, square} appear in the best equation."""
    s = eq_str.lower()
    hits = 0
    if "sin" in s:
        hits += 1
    if "log" in s or "safe_log" in s:
        hits += 1
    # "square" or x*x or x**2 -- any one counts
    if "square" in s or "x3 * x3" in s or "x3*x3" in s or "x3^2" in s or "x3 ^ 2" in s:
        hits += 1
    return hits


def _run_one(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    preset_name: str | None,
    legacy: bool,
    label: str,
) -> Dict[str, Any]:
    """Fit one PySR config; return wall-time + RMSE + form score."""
    from mlframe.feature_engineering.bruteforce import run_pysr_feature_engineering

    if legacy:
        params = dict(
            niterations=200,
            populations=15,
            population_size=33,
            tournament_selection_n=8,
            maxdepth=5,
            binary_operators=["+", "*"],
            unary_operators=["log", "inv(x) = 1/x"],
            procs=1,
            precision=32,
            verbosity=0,
            update=False,
            progress=False,
        )
    else:
        from mlframe.feature_engineering.pysr_operators import get_preset_kwargs
        preset = get_preset_kwargs(preset_name or "standard")
        params = dict(
            # Mirrors pipeline.py:_apply_pysr_fe defaults exactly so the bench reflects what users see.
            niterations=400,
            populations=max(4, min(15, (os.cpu_count() or 4) // 3)),
            population_size=33,
            tournament_selection_n=15,
            maxsize=20,
            maxdepth=5,
            parsimony=1e-4,
            weight_optimize=0.001,
            heap_size_hint_in_bytes=256 * 1024 * 1024,
            binary_operators=preset["binary_operators"],
            unary_operators=preset["unary_operators"],
            complexity_of_operators=preset["complexity_of_operators"],
            nested_constraints=preset["nested_constraints"],
            extra_sympy_mappings=preset["extra_sympy_mappings"],
            batching=True,
            batch_size=10000,
            precision=32,
            turbo=True,
            bumper=True,
            update=False,
            progress=False,
            verbosity=0,
        )

    # Inject y as a temp column (bruteforce expects target as a column).
    tmp = df_train.copy()
    tmp["__y__"] = y_train
    t0 = time.perf_counter()
    try:
        model = run_pysr_feature_engineering(
            df=tmp, target_col="__y__", sample_size=len(tmp),
            encode_categoricals=False,
            pysr_params_override=params, verbose=0,
        )
    except Exception as e:
        return {"label": label, "wall_s": -1.0, "rmse": float("nan"), "form_score": -1, "error": str(e)[:80]}
    wall = time.perf_counter() - t0

    eqs = getattr(model, "equations_", None)
    if eqs is None or len(eqs) == 0:
        return {"label": label, "wall_s": wall, "rmse": float("nan"), "form_score": 0, "error": "no equations"}

    best = eqs.sort_values("score", ascending=False).iloc[0]
    eq_str = str(best.get("equation", best.get("sympy_format", "")))
    # Predict best equation on holdout
    try:
        y_pred = np.asarray(model.predict(df_holdout, index=best.name), dtype=np.float32).ravel()
        rmse = float(np.sqrt(np.mean((y_pred - y_holdout) ** 2)))
    except Exception:
        rmse = float("nan")
    form = _equation_form_score(eq_str)
    return {"label": label, "wall_s": wall, "rmse": rmse, "form_score": form, "eq": eq_str[:80]}


def main() -> int:
    try:
        import pysr  # noqa: F401
    except ImportError:
        print("WARN: pysr not installed; bench skipped", file=sys.stderr)
        return 0

    df_train, y_train = _make_synth(n=5000, seed=42)
    df_holdout, y_holdout = _make_synth(n=1000, seed=43)
    df_holdout.columns = df_train.columns

    # Julia warm-up so the first measured run doesn't pay the ~30-60s precompile penalty.
    print("Warming up Julia (~30-60s on cold cache)...", file=sys.stderr)
    _run_one(df_train.head(500), y_train[:500], df_holdout.head(100), y_holdout[:100], "minimal", legacy=False, label="warmup")

    print("Running configurations...", file=sys.stderr)
    rows: List[Dict[str, Any]] = []
    rows.append(_run_one(df_train, y_train, df_holdout, y_holdout, None, legacy=True, label="legacy_defaults"))
    rows.append(_run_one(df_train, y_train, df_holdout, y_holdout, "minimal", legacy=False, label="new_minimal"))
    rows.append(_run_one(df_train, y_train, df_holdout, y_holdout, "standard", legacy=False, label="new_standard"))
    rows.append(_run_one(df_train, y_train, df_holdout, y_holdout, "physics", legacy=False, label="new_physics"))

    md_lines = [
        "# bench_pysr_fe results",
        "",
        f"Synthetic ground truth: `y = 3*sin(x1) + log(|x2|+1) - 0.5*x3^2 + N(0,0.3)`; n_train=5000, n_holdout=1000, 8 features (5 noise).",
        f"Host: {os.cpu_count()} cores, JULIA_NUM_THREADS={os.environ.get('JULIA_NUM_THREADS', '?')}, "
        f"PYTHON_JULIACALL_THREADS={os.environ.get('PYTHON_JULIACALL_THREADS', '?')}.",
        f"Run UTC: {datetime.now(timezone.utc).isoformat(timespec='seconds')}.",
        "",
        "| label | wall (s) | holdout RMSE | form score (0-3) | best equation (truncated) |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(f"| {r['label']} | {r['wall_s']:.1f} | {r['rmse']:.4f} | {r.get('form_score', '-')} | " f"{r.get('eq', r.get('error', ''))} |")
    md = "\n".join(md_lines)
    print(md)

    # Persist
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(__file__).resolve().parents[3].parent / "tests" / "perf" / "results" / f"bench_pysr_fe_{stamp}.md"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"\nSaved to {out}", file=sys.stderr)
    except Exception as e:
        print(f"WARN: could not save to {out}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
