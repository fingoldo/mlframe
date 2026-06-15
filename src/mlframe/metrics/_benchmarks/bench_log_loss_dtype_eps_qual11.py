"""qual-11 bench (REJECTED): is fast_log_loss's dtype-dependent clip eps a bug?

Run: PYTHONPATH=src python src/mlframe/metrics/_benchmarks/bench_log_loss_dtype_eps_qual11.py

Verdict: REJECT. The legacy ``eps = np.finfo(y_pred.dtype).eps`` default is the correct precision-matched floor. The "always float64 eps" alternative
(2.22e-16) does NOT improve float32 honesty -- it OVERSHOOTS because float32 cannot represent (1 - 1.19e-7, 1): a confident ``1 - 1e-8`` collapses to
exactly ``1.0`` on cast, and clipping that at 2.22e-16 penalises a confident-wrong row with ``-log(2.22e-16) ~= 36`` vs the honest ``-log(1e-8) ~= 18``.

Measured at conf=1e-8, n=100k, 5 seeds:
  honest (float64/sklearn) ~= 0.90
  ml64 (eps=2.22e-16)      ~= 0.90  (matches honest)
  ml32 legacy (eps=1.19e-7)~= 0.78  (caps confident-wrong -- optimistic, the apparent "bug")
  ml32 forced float64 eps  ~= 1.34  (OVERSHOOTS -- the alternative is worse, not better)
So neither eps gives float32 the honest value: float32 simply cannot carry the near-1 information. The dtype-dependent default is the least-wrong floor.
"""
import os
import sys

os.environ.setdefault("MLFRAME_SKIP_NUMBA_PREWARM", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("NUMBA_DISABLE_CUDA", "1")
import scipy.stats  # noqa: F401
import numba  # noqa: F401

sys.modules["cupy"] = None
import numpy as np
from sklearn.metrics import log_loss as sk_log_loss

from mlframe.metrics.core import fast_log_loss


def main() -> None:
    n = 100_000
    print("=== overconfident-wrong (conf=1e-8, 5% wrong) ===")
    for seed in range(5):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.5).astype(np.int64)
        wrong = rng.random(n) < 0.05
        conf = 1e-8
        p1 = np.where(y == 1, 1 - conf, conf)
        p1 = np.where(wrong, 1 - p1, p1)
        p = np.clip(p1, 1e-12, 1 - 1e-12)
        honest = sk_log_loss(y, p)
        ml32 = fast_log_loss(y, p.astype(np.float32))
        ml64 = fast_log_loss(y, p.astype(np.float64))
        forced64 = fast_log_loss(y, p.astype(np.float32), eps=float(np.finfo(np.float64).eps))
        print(f" s{seed} honest={honest:.4f} ml64={ml64:.4f} ml32_legacy={ml32:.4f} ml32_forced_f64eps={forced64:.4f}")


if __name__ == "__main__":
    main()
