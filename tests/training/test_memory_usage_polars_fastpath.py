"""Regression sensor for S49: when a polars-side ``estimated_size()`` is already cached on a
``was_polars_input`` run, ``_phase_helpers`` must NOT recompute the size via the very expensive
``pd.DataFrame.memory_usage(deep=True)`` call (which scans every cell of every object-dtype
column -- multi-minute on 4M-row x 25-col frames per the original observability log).

On a non-polars input the fallback uses ``deep=False`` (buffer-block read, <1ms) instead of
``deep=True`` (per-cell scan, ~17s on a 4M-row object-heavy frame).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from tests.conftest import perf_time_budget

_PHASE_HELPERS_PATH = Path(__file__).resolve().parents[2] / "src" / "mlframe" / "training" / "core" / "_phase_helpers.py"


def _read_phase_helpers() -> str:
    """Read phase helpers."""
    return _PHASE_HELPERS_PATH.read_text(encoding="utf-8")


def _deep_true_call_lines(src: str) -> list[int]:
    """AST-walk for ``.memory_usage(...)`` calls with ``deep=True`` kwarg. Docstring / comment
    occurrences (string literals) are ignored — those describe the banned pattern in prose."""
    tree = ast.parse(src)
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "memory_usage":
            for kw in node.keywords:
                if kw.arg == "deep" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                    hits.append(node.lineno)
    return hits


def test_S49_no_deep_memory_usage_call_in_phase_helpers():
    """``df.memory_usage(deep=True, ...)`` must not appear as an executable call in _phase_helpers.py.

    The deep=True scan on object-dtype columns is the multi-minute hot point that motivated S49.
    The fix uses ``memory_usage(deep=False, ...)`` and skips when a polars ``estimated_size()``
    has already populated the cached size. Docstring/comment references to the retired pattern
    are intentional (they explain why deep=True was retired) and are excluded via AST walking.
    """
    hits = _deep_true_call_lines(_read_phase_helpers())
    assert not hits, (
        f"memory_usage(deep=True) call detected at lines {hits}; this is the multi-minute "
        "hot point S49 retired. Use deep=False or skip when polars estimated_size is cached."
    )


def test_S49_size_compute_skips_when_polars_cache_present(monkeypatch):
    """A polars input keeps its pre-conversion ``estimated_size`` for the size cache and never runs the pandas ``memory_usage`` fallback; a pandas input falls back to the shallow scan."""
    import polars as pl

    from mlframe.training.core._phase_helpers import _phase_pandas_conversion_and_cat_prep

    def _frame(n):
        """A polars frame with a numeric and a long-string column."""
        return pl.DataFrame({"num": np.arange(n, dtype=np.float64), "txt": [f"value_number_{i:08d}_padding_text" for i in range(n)]})

    def _prep(train, val, was_polars_input):
        """Run the pandas-conversion phase with a configuration that forces the conversion."""
        return _phase_pandas_conversion_and_cat_prep(
            train_df=train,
            val_df=val,
            test_df=None,
            train_df_polars_pre=None,
            val_df_polars_pre=None,
            test_df_polars_pre=None,
            cat_features=[],
            was_polars_input=was_polars_input,
            all_models_polars_native=False,
            needs_polars_pre_clone=False,
            mlframe_models=["lgb"],
            recurrent_models=[],
            rfecv_models=[],
            baseline_rss_mb=0.0,
            df_size_mb=0.0,
            verbose=False,
        )

    memory_usage_calls = []
    real_memory_usage = pd.DataFrame.memory_usage

    def _spy(self, *args, **kwargs):
        """Record the shallow fallback scan (deep=False, index=False), then delegate."""
        if kwargs == {"deep": False, "index": False}:
            memory_usage_calls.append(kwargs)
        return real_memory_usage(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "memory_usage", _spy)
    train_pl, val_pl = _frame(500), _frame(200)

    out = _prep(train_pl, val_pl, True)

    assert out[11] is False
    assert out[9] == float(train_pl.estimated_size())
    assert out[10] == float(val_pl.estimated_size())
    assert memory_usage_calls == []

    train_pd = pd.DataFrame({"num": np.arange(500, dtype=np.float64), "txt": [f"value_number_{i:08d}_padding_text" for i in range(500)]})
    out_pd = _prep(train_pd, None, False)

    assert out_pd[9] == float(real_memory_usage(train_pd, deep=False, index=False).sum())
    assert out_pd[10] is None
    assert len(memory_usage_calls) >= 1


def test_S49_shallow_memory_usage_is_fast_and_returns_finite_bytes():
    """Sanity check on the shallow ``memory_usage(deep=False)`` fallback: completes in well
    under a second on a 10k-row x 5-col mixed object/float fixture, returns a positive int.

    Small fixture keeps this safe under concurrent test load (parallel-agent paging pressure).
    The relative speed claim (shallow << deep) is documented; this test just verifies the API
    behaviour we rely on still works.
    """
    import time

    rng = np.random.default_rng(0)
    n = 10_000
    data = {}
    for i in range(3):
        vals = [f"k_{k}" for k in range(10)]
        data[f"oc_{i}"] = pd.Series(rng.choice(vals, size=n), dtype="object")
    for i in range(2):
        data[f"n_{i}"] = rng.standard_normal(n).astype(np.float32)
    df = pd.DataFrame(data)

    t0 = time.perf_counter()
    sz = float(df.memory_usage(deep=False, index=False).sum())
    elapsed = time.perf_counter() - t0
    assert sz > 0
    # Generous ceiling so concurrent-agent paging pressure doesn't trip the test.
    assert elapsed < perf_time_budget(1.0), f"memory_usage(deep=False) took {elapsed:.3f}s; expected <1s on shallow scan"
