"""Bench the LTR cat-vocab build at ranker_suite.py:518-545.

Profiled in iter112 follow-up to c0070 (200k rows, 15 cat cols):
the per-row ``_vals.add(_v)`` loop showed up as 13828 lambda calls + ~700ms
across the three splits. ``set.update(.dropna().tolist())`` pushes the
unique-value collection to the C level (~1.45x faster) without losing the
defensive ``try/except TypeError`` for unhashable cells. ``key=str`` replaces
the equivalent ``lambda x: str(x)`` to skip a Python frame per comparison.

Run: ``python profiling/bench_ranker_suite_vocab_build.py``

Variants measured:
  - current        : per-row set.add + key=lambda x: str(x)  (pre-fix)
  - update         : set.update(.tolist()) + key=lambda x: str(x)
  - update+keystr  : set.update(.tolist()) + key=str (shipped)
  - unique+update  : .dropna().unique().tolist() then set.update (slower at this size)
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd


def make_synthetic_splits(
    n_rows: int = 200_000,
    n_cols: int = 15,
    n_unique_per_col: int = 920,
    seed: int = 20260521,
) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cols = {}
    for c in range(n_cols):
        vocab = [f'cat_{c}_v{i}' for i in range(n_unique_per_col)]
        cols[f'col_{c}'] = rng.choice(vocab, size=n_rows)
    train = pd.DataFrame(cols)
    val = train.iloc[:n_rows // 3].copy()
    test = train.iloc[:n_rows // 3].copy()
    return [train, val, test]


def build_vocab_current(splits, to_encode):
    vocabs = {}
    for c in to_encode:
        vals = set()
        for split in splits:
            for v in split[c].dropna().tolist():
                try:
                    vals.add(v)
                except TypeError:
                    pass
        vocabs[c] = {v: i for i, v in enumerate(sorted(vals, key=lambda x: str(x)))}
    return vocabs


def build_vocab_update(splits, to_encode):
    vocabs = {}
    for c in to_encode:
        vals = set()
        for split in splits:
            try:
                vals.update(split[c].dropna().tolist())
            except TypeError:
                continue
        vocabs[c] = {v: i for i, v in enumerate(sorted(vals, key=lambda x: str(x)))}
    return vocabs


def build_vocab_update_keystr(splits, to_encode):
    vocabs = {}
    for c in to_encode:
        vals = set()
        for split in splits:
            try:
                vals.update(split[c].dropna().tolist())
            except TypeError:
                continue
        vocabs[c] = {v: i for i, v in enumerate(sorted(vals, key=str))}
    return vocabs


def build_vocab_unique(splits, to_encode):
    vocabs = {}
    for c in to_encode:
        vals = set()
        for split in splits:
            try:
                vals.update(split[c].dropna().unique().tolist())
            except TypeError:
                continue
        vocabs[c] = {v: i for i, v in enumerate(sorted(vals, key=str))}
    return vocabs


def time_fn(fn, splits, to_encode, n: int = 3):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        out = fn(splits, to_encode)
        times.append(time.perf_counter() - t)
    return min(times), out


def main():
    splits = make_synthetic_splits()
    to_encode = list(splits[0].columns)
    variants = [
        ('current', build_vocab_current),
        ('update', build_vocab_update),
        ('update+keystr', build_vocab_update_keystr),
        ('unique+update', build_vocab_unique),
    ]
    results = []
    for name, fn in variants:
        t, out = time_fn(fn, splits, to_encode)
        n_entries = sum(len(v) for v in out.values())
        results.append((name, t, n_entries))
        print(f'{name:>20}: {t*1000:7.1f}ms (vocab total entries: {n_entries})')
    # Equivalence between current and shipped (update+keystr).
    v_cur = build_vocab_current(splits, to_encode)
    v_new = build_vocab_update_keystr(splits, to_encode)
    for c in to_encode:
        assert v_cur[c] == v_new[c], f'vocab mismatch in column {c}'
    print('equivalence OK (current == update+keystr)')


if __name__ == '__main__':
    main()
