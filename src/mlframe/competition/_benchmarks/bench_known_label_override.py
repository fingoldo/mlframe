"""cProfile benchmark for the COMPETITION-ONLY monotonic_entity_override / known_label_override.

Not for production use — profiles Kaggle-only prediction-override utilities (mlframe.competition).
Run directly: `python bench_known_label_override.py`.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np

from mlframe.competition.known_label_override import known_label_override, monotonic_entity_override


def _make_entity_data(n: int, n_entities: int, seed: int) -> tuple[np.ndarray, np.ndarray, set]:
    rng = np.random.default_rng(seed)
    entity_ids = rng.integers(0, n_entities, size=n)
    preds = rng.beta(a=0.5, b=8.0, size=n)
    known_positive = set(rng.choice(n_entities, size=max(1, n_entities // 20), replace=False).tolist())
    return preds, entity_ids, known_positive


def _make_known_label_map(n: int, n_known: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    idxs = rng.choice(n, size=n_known, replace=False)
    labels = rng.integers(0, 2, size=n_known).astype(float)
    return dict(zip(idxs.tolist(), labels.tolist()))


def run_monotonic_once(n: int) -> None:
    preds, entity_ids, known_positive = _make_entity_data(n=n, n_entities=max(10, n // 20), seed=0)
    monotonic_entity_override(preds, entity_ids, known_positive)


def run_known_label_once(n: int) -> None:
    preds = np.random.default_rng(1).beta(a=0.5, b=8.0, size=n)
    known_map = _make_known_label_map(n=n, n_known=max(1, n // 50), seed=1)
    known_label_override(preds, known_map, asymmetric_safe_direction="positive")


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for n in (1_000, 10_000, 100_000):
        for _ in range(3):
            run_monotonic_once(n)
    for n in (1_000, 10_000, 100_000):
        for _ in range(3):
            run_known_label_once(n)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
