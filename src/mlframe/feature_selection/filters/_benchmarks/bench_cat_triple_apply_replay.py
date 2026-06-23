"""Microbench: vectorized factorize-gather replay for ``apply_cat_triple_cross``
vs the prior per-row ``mapping.get((a, b, c))`` Python loop.

``apply_cat_triple_cross`` is the transform-time replay of a cat x cat x cat
triple cross (Layer 94). The prior code looped over every test row doing a
Python dict lookup on a freshly-built ``(val_a, val_b, val_c)`` tuple -- O(n)
interpreted-Python work + n tuple allocations + n dict probes. The vectorized
path factorizes the three columns (O(n) hashtable, no sort), packs a joint key,
resolves the lookup ONCE per distinct cell, and gathers -- so the Python work
is O(distinct cells) not O(n). Bit-identical to the per-row loop.

Run:
  CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/src python bench_cat_triple_apply_replay.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._cat_triple_fe import (
    apply_cat_triple_cross,
    _encode_triple,
)
from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str


def _old_apply(X_test, cat_a, cat_b, cat_c, mapping, *, encoding="raw", te_lookup=None, global_mean=0.0):
    """Verbatim prior per-row loop (the OLD baseline for the A/B)."""
    cats_a = _column_to_str(X_test[cat_a])
    cats_b = _column_to_str(X_test[cat_b])
    cats_c = _column_to_str(X_test[cat_c])
    n = len(cats_a)
    sentinel = len(mapping)
    if encoding == "target":
        lookup = te_lookup or {}
        out = np.empty(n, dtype=np.float64)
        for r in range(n):
            code = mapping.get((cats_a[r], cats_b[r], cats_c[r]))
            out[r] = global_mean if code is None else float(lookup.get(code, global_mean))
        return out
    out = np.empty(n, dtype=np.float64)
    for r in range(n):
        code = mapping.get((cats_a[r], cats_b[r], cats_c[r]), sentinel)
        out[r] = float(code)
    return out


def _make_data(n, card, seed=0):
    rng = np.random.default_rng(seed)
    fit = pd.DataFrame({
        "a": rng.integers(0, card, n).astype(str),
        "b": rng.integers(0, card, n).astype(str),
        "c": rng.integers(0, card, n).astype(str),
    })
    _, mapping = _encode_triple(
        np.asarray(_column_to_str(fit["a"])),
        np.asarray(_column_to_str(fit["b"])),
        np.asarray(_column_to_str(fit["c"])),
    )
    # test set: same distribution + a slice of unseen categories to hit the sentinel/global_mean fallback
    test = pd.DataFrame({
        "a": rng.integers(0, card + 2, n).astype(str),
        "b": rng.integers(0, card + 2, n).astype(str),
        "c": rng.integers(0, card + 2, n).astype(str),
    })
    te_lookup = {code: float(rng.normal()) for code in set(mapping.values())}
    return test, mapping, te_lookup


def main():
    print(f"{'n':>8} {'card':>5} {'enc':>7} {'old_ms':>9} {'new_ms':>9} {'speedup':>8}  identity")
    for n, card in ((10_000, 8), (100_000, 12), (1_000_000, 20)):
        test, mapping, te_lookup = _make_data(n, card)
        for enc, kw in (("raw", {}), ("target", {"encoding": "target", "te_lookup": te_lookup, "global_mean": 0.5})):
            old = _old_apply(test, "a", "b", "c", mapping, **kw)
            new = apply_cat_triple_cross(test, "a", "b", "c", mapping, **kw)
            identical = np.array_equal(old, new)

            reps = 3 if n >= 1_000_000 else 7
            best_old = min(_time(lambda: _old_apply(test, "a", "b", "c", mapping, **kw)) for _ in range(reps))
            best_new = min(_time(lambda: apply_cat_triple_cross(test, "a", "b", "c", mapping, **kw)) for _ in range(reps))
            print(f"{n:>8} {card:>5} {enc:>7} {best_old*1e3:>9.2f} {best_new*1e3:>9.2f} {best_old/best_new:>7.2f}x  {'BIT-IDENTICAL' if identical else 'MISMATCH!'}")


def _time(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


if __name__ == "__main__":
    main()
