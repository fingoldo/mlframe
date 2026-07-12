"""cProfile benchmark for the COMPETITION-ONLY power_rescale_to_target_sum / asymmetric_scale_by_sign.

Not for production use — profiles Kaggle-only post-hoc rescaling utilities (mlframe.competition).
Run directly: `python bench_power_rescale.py`.
"""

from __future__ import annotations

import cProfile
import io
import pstats

import numpy as np

from mlframe.competition.power_rescale import asymmetric_scale_by_sign, power_rescale_to_target_sum


def _make_probs(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.beta(a=0.5, b=8.0, size=n)


def _make_signed_preds(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.normal(size=n)
    y_pred = np.where(y_true < 0, y_true * 1.6, y_true * 1.0) + rng.normal(scale=0.05, size=n)
    return y_true, y_pred


def run_power_rescale_once(n: int) -> None:
    probs = _make_probs(n=n, seed=0)
    target_sum = float(np.sum(probs)) * 1.8
    power_rescale_to_target_sum(probs, target_sum)


def run_asymmetric_scale_once(n: int) -> None:
    y_true, y_pred = _make_signed_preds(n=n, seed=42)

    def neg_mse(candidate: np.ndarray) -> float:
        return -float(np.mean((y_true - candidate) ** 2))

    asymmetric_scale_by_sign(y_pred, neg_mse, scale_range=(1.0, 2.0), n_steps=101)


def main() -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    for n in (1_000, 10_000, 100_000):
        for _ in range(3):
            run_power_rescale_once(n)
    for n in (1_000, 10_000, 50_000):
        for _ in range(3):
            run_asymmetric_scale_once(n)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats(30)
    print(stream.getvalue())


if __name__ == "__main__":
    main()
