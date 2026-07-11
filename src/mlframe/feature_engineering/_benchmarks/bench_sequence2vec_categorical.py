"""cProfile harness for ``feature_engineering.train_sequence2vec``.

Run: ``python -m mlframe.feature_engineering._benchmarks.bench_sequence2vec_categorical``
"""
from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO

import numpy as np

import pandas as pd

from mlframe.feature_engineering.sequence2vec_categorical import sequence2vec_entity_features, sequence2vec_transform_new_entities, train_sequence2vec


def _make_sequences(n_entities: int, seq_len: int, vocab_size: int, seed: int):
    rng = np.random.default_rng(seed)
    vocab = [f"tok_{i}" for i in range(vocab_size)]
    return [list(rng.choice(vocab, size=seq_len)) for _ in range(n_entities)]


def _make_events_df(n_entities: int, seq_len: int, vocab_size: int, seed: int, entity_offset: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vocab = [f"tok_{i}" for i in range(vocab_size)]
    rows = []
    for e in range(entity_offset, entity_offset + n_entities):
        for t in range(seq_len):
            rows.append({"entity": e, "t": t, "tok": rng.choice(vocab)})
    return pd.DataFrame(rows)


def _run(n_entities: int, seq_len: int, vocab_size: int, n_epochs: int) -> None:
    sequences = _make_sequences(n_entities, seq_len, vocab_size, seed=0)
    train_sequence2vec(sequences, embedding_dim=16, window=3, n_negative=5, n_epochs=n_epochs, random_state=0)


def _run_transform_new_entities(n_entities: int, seq_len: int, vocab_size: int, n_epochs: int) -> None:
    """Fit once on a "training" population, then embed a disjoint held-out population on the frozen basis --
    the new opt-in inference path (:func:`sequence2vec_transform_new_entities`)."""
    train_df = _make_events_df(n_entities, seq_len, vocab_size, seed=0, entity_offset=0)
    holdout_df = _make_events_df(n_entities, seq_len, vocab_size, seed=1, entity_offset=n_entities)
    _, embeddings = sequence2vec_entity_features(
        train_df, "entity", "tok", time_col="t", embedding_dim=16, window=3, n_epochs=n_epochs, random_state=0, return_embeddings=True
    )
    sequence2vec_transform_new_entities(holdout_df, "entity", "tok", embeddings, time_col="t")


if __name__ == "__main__":
    for n_entities, seq_len, vocab_size, n_epochs in [(200, 10, 50, 3), (2_000, 20, 200, 1)]:
        t0 = time.perf_counter()
        _run(n_entities, seq_len, vocab_size, n_epochs)
        wall = time.perf_counter() - t0
        print(f"train_sequence2vec         n_entities={n_entities:>6,} seq_len={seq_len:>3} vocab={vocab_size:>4} n_epochs={n_epochs} -> {wall * 1000:9.2f} ms")

    for n_entities, seq_len, vocab_size, n_epochs in [(200, 10, 50, 3), (2_000, 20, 200, 1)]:
        t0 = time.perf_counter()
        _run_transform_new_entities(n_entities, seq_len, vocab_size, n_epochs)
        wall = time.perf_counter() - t0
        print(
            f"fit+transform_new_entities n_entities={n_entities:>6,} seq_len={seq_len:>3} vocab={vocab_size:>4} n_epochs={n_epochs} -> {wall * 1000:9.2f} ms"
        )

    profiler = cProfile.Profile()
    profiler.enable()
    _run(200, 10, 50, 3)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())

    profiler = cProfile.Profile()
    profiler.enable()
    _run_transform_new_entities(200, 10, 50, 3)
    profiler.disable()
    buf = StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(15)
    print(buf.getvalue())
