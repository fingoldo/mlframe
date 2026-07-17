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


def test_S49_size_compute_skips_when_polars_cache_present():
    """When ``train_df_size_bytes_cached`` was already populated (polars path), the fallback
    pandas ``memory_usage`` must NOT overwrite it.
    """
    src = _read_phase_helpers()
    # Behavioural contract pinned in the new comment + ``is None`` guard.
    assert "train_df_size_bytes_cached is None" in src, (
        "Expected a guard 'train_df_size_bytes_cached is None' so the polars-cached value is preserved when present (skip the pandas fallback)."
    )
    assert "val_df_size_bytes_cached is None" in src, (
        "Expected a guard 'val_df_size_bytes_cached is None' so the polars-cached value is preserved when present (skip the pandas fallback)."
    )


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
    assert elapsed < 1.0, f"memory_usage(deep=False) took {elapsed:.3f}s; expected <1s on shallow scan"
