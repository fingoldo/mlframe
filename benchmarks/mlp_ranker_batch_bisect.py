"""Find the FIRST training batch at which MLPRanker's cold fit parts from its warm fits.

Everything reachable from outside the training loop has been eliminated (see known_complications.md):
CPU is bit-identical, the CUDA generator state is byte-identical before every fit, no global precision
switch is mutated, and neither the cuBLAS determinism pair, a pre-warmed device generator, nor
empty_cache() between fits changes the 0.00631118 divergence.

So bisect INSIDE training. A Lightning callback records the loss of every training batch. Fit twice in one
process and report the first batch index whose loss differs, which separates:

  * diverges at batch 0  -> the very first backward pass already differs, i.e. cold-call kernel selection
  * identical for N > 0  -> state accumulates somewhere and only then drifts

Run in a cold process (not under pytest).
"""

from __future__ import annotations

import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


class LossTape(Callback):
    """Record the scalar loss of every training batch, in order."""

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.batch_fps: list[str] = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx) -> None:
        # Fingerprint the batch CONTENT. A different loss on an identical batch means the computation
        # differs; a different fingerprint means the sampler/dataloader handed over different rows.
        import hashlib

        parts = []
        for t in (batch if isinstance(batch, (list, tuple)) else [batch]):
            if torch.is_tensor(t):
                parts.append(t.detach().cpu().numpy().tobytes())
        self.batch_fps.append(hashlib.blake2b(b"|".join(parts), digest_size=8).hexdigest())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        value = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.losses.append(float(value.detach().cpu()) if torch.is_tensor(value) else float(value))


def run_once() -> tuple[list[float], np.ndarray]:
    """Fit the standard fixture once, returning the per-batch loss tape and the predictions."""
    from mlframe.training.neural.ranker import MLPRanker

    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 5)).astype(np.float32)
    group_ids = np.repeat(np.arange(n // 6), 6)
    y = rng.integers(0, 4, size=n).astype(np.float32)

    # MLPRanker builds its own callback list internally and takes no `callbacks` kwarg, so the tape is
    # injected by wrapping Trainer.__init__ for the duration of the fit.
    import lightning.pytorch as pl

    tape = LossTape()
    original_init = pl.Trainer.__init__

    def _patched_init(self, *args, **kwargs):
        """Append the loss tape to whatever callbacks the estimator asked for."""
        kwargs["callbacks"] = [*list(kwargs.get("callbacks") or []), tape]
        original_init(self, *args, **kwargs)

    pl.Trainer.__init__ = _patched_init
    try:
        torch.manual_seed(999)
        model = MLPRanker(seed=7, n_estimators=3, hidden_layers=(8,), early_stopping_patience=None)
        model.fit(X, y, group_ids)
    finally:
        pl.Trainer.__init__ = original_init
    return (tape.losses, tape.batch_fps), np.asarray(model.predict(X), dtype=np.float64)


def main() -> None:
    """Fit twice and report the first batch whose loss differs."""
    (losses_a, fps_a), pred_a = run_once()
    (losses_b, fps_b), pred_b = run_once()
    print(f"batch CONTENT identical across fits: {fps_a == fps_b}")
    if fps_a != fps_b:
        first = next(i for i, (a, b) in enumerate(zip(fps_a, fps_b)) if a != b)
        print(f"  first differing batch content at index {first}: {fps_a[first]} vs {fps_b[first]}")

    print(f"batches recorded: fit1={len(losses_a)} fit2={len(losses_b)}")
    print(f"prediction maxdiff: {np.max(np.abs(pred_a - pred_b)):.8f}")
    if not losses_a:
        print("NO BATCHES RECORDED -- the callback did not reach the trainer; wire it differently")
        return

    for i, (a, b) in enumerate(zip(losses_a, losses_b)):
        if a != b:
            print(f"FIRST DIVERGING BATCH: index {i} of {len(losses_a)}  fit1={a!r}  fit2={b!r}  delta={a - b:+.3e}")
            break
    else:
        print("every recorded batch loss is bit-identical -- the divergence is NOT in the training loss path")


if __name__ == "__main__":
    main()
