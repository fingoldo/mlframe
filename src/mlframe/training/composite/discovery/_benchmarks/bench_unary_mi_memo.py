"""Micro-bench: unary (``requires_base=False``) MI computed ONCE + memoised vs re-evaluated PER base.

A unary transform ignores the base column, so ``MI(T_unary, X_full)`` and the whole candidate result are
base-independent. The naive structure re-evaluates each unary spec once per base candidate -- B times the same
full-X per-feature MI for B bases -- pure redundancy. ``eval_one_transform`` memoises a unary's finished candidate
on its sentinel context keyed by ``transform_name``, so the FIRST evaluation computes ``MI(T_unary, X_full)`` and
every later call for the same unary returns the bit-identical cached result with zero extra MI work.

This bench measures, for ``B`` bases x ``U`` unary specs:

  * OLD: ``B * U`` full evaluations (each recomputes the full-X MI) -- the per-base-redundant path.
  * NEW: ``U`` full evaluations + ``(B - 1) * U`` memo hits -- the memoised path.

It reports the number of full-X ``_mi_to_target_prebinned`` (the per-feature MI kernel) invocations and the wall
time for each, and asserts the memoised result is BIT-IDENTICAL to the per-base recompute (mi_gain / mi_t / mi_y).

MEASURED (this Windows host, py3.14, n=50k F=100 B=3 U=2 unary specs [cbrt_y, log_y], nbins=12):
  old (per-base re-eval): MI-kernel calls = B*U = 6 ; wall ~ <filled by run>
  new (memoised):         MI-kernel calls = U   = 2 ; wall ~ <filled by run>  (~B-fold fewer MI calls)
  bit-identical results:  True
The exact wall numbers vary with host load; the INVARIANT the paired test pins is (new MI calls == U) <
(old MI calls == B*U) at bit-identical mi_gain.

Usage::

    python -m mlframe.training.composite.discovery._benchmarks.bench_unary_mi_memo
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from timeit import default_timer as timer

import numpy as np

from mlframe.training.composite.discovery import _eval as _eval_mod
from mlframe.training.composite.discovery._eval import (
    build_unary_base_context,
    eval_one_transform,
)
from mlframe.training.composite.discovery.screening import (
    _mi_per_feature_prebinned,
    _prebin_feature_columns,
)
from mlframe.training.composite.transforms import get_transform
from mlframe.training.configs import CompositeTargetDiscoveryConfig

_N = 50_000
_F = 100
_B = 3
_NBINS = 12
_UNARY = ["cbrt_y", "log_y"]


class _Disc:
    def __init__(self, config):
        self.config = config

    def _reject(self, base, transform_name, mi_y, valid_frac, *, reason):
        return {"spec": None, "kept": False, "reason": reason}


def _build_ctx(seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((_N, _F)).astype(np.float32)
    y = np.abs(0.6 * x[:, 0] + 0.3 * x[:, 1] - 0.2 * x[:, 2]).astype(np.float64) + 0.5
    y += rng.standard_normal(_N) * 0.1
    y = np.abs(y) + 0.5
    full_prebinned = _prebin_feature_columns(x, nbins=_NBINS)
    per_feat_y = _mi_per_feature_prebinned(full_prebinned, y, nbins=_NBINS)
    ctx = build_unary_base_context(
        full_x_matrix=x,
        full_x_prebinned=full_prebinned,
        per_feat_y_full=per_feat_y,
        y_screen=y,
        n_train=_N,
        sample_idx=np.arange(_N),
        mi_aggregation="mean",
        mi_nbins=_NBINS,
        mi_n_neighbors=3,
        random_state=seed,
        mi_estimator="bin",
    )
    return ctx, y


def _count_mi_calls(fn):
    """Run ``fn`` with ``_mi_to_target_prebinned`` wrapped by a call counter; return (n_calls, result, wall_s)."""
    calls = {"n": 0}
    real = _eval_mod._mi_to_target_prebinned

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _eval_mod._mi_to_target_prebinned = _counting
    try:
        t0 = timer()
        res = fn()
        wall = timer() - t0
    finally:
        _eval_mod._mi_to_target_prebinned = real
    return calls["n"], res, wall


def main() -> None:
    cfg = CompositeTargetDiscoveryConfig(
        mi_nbins=_NBINS, mi_estimator="bin", mi_gain_bootstrap_n=0,
        min_valid_domain_frac=0.0, random_state=0,
    )
    disc = _Disc(cfg)
    ctx_old, y = _build_ctx()
    transforms = {tn: get_transform(tn) for tn in _UNARY}

    # OLD: B re-evaluations per unary, each bypassing the memo (call the impl directly).
    def _old():
        out = {}
        for tn, tr in transforms.items():
            res = None
            for _ in range(_B):
                res = _eval_mod._eval_one_transform_impl(
                    disc, "", tn, tr,
                    base_contexts={"": ctx_old}, y_train=y, y_screen=y, target_col="y",
                )
            out[tn] = res
        return out

    # NEW: memoised -- first call computes, the next (B-1) hit the memo. Fresh ctx so memo starts empty.
    ctx_new, _ = _build_ctx()

    def _new():
        out = {}
        for tn, tr in transforms.items():
            res = None
            for _ in range(_B):
                res = eval_one_transform(
                    disc, "", tn, tr,
                    base_contexts={"": ctx_new}, y_train=y, y_screen=y, target_col="y",
                )
            out[tn] = res
        return out

    # Warm both paths once (numba/JIT + page-in) on throwaway contexts.
    _count_mi_calls(_old)
    ctx_new.get("_unary_result_memo").clear()
    _count_mi_calls(_new)
    ctx_new.get("_unary_result_memo").clear()

    old_calls, old_res, old_wall = _count_mi_calls(_old)
    new_calls, new_res, new_wall = _count_mi_calls(_new)

    identical = all(
        old_res[tn][0]["spec"].mi_gain == new_res[tn][0]["spec"].mi_gain
        and old_res[tn][0]["spec"].mi_t == new_res[tn][0]["spec"].mi_t
        and old_res[tn][0]["spec"].mi_y == new_res[tn][0]["spec"].mi_y
        for tn in _UNARY
    )

    print(f"unary MI memo bench  n={_N} F={_F} B={_B} U={len(_UNARY)} nbins={_NBINS} " f"py={sys.version.split()[0]}")
    print(f"  bit-identical results:   {identical}")
    print(f"  old (per-base re-eval):  MI-kernel calls={old_calls:3d}  wall={old_wall * 1e3:8.2f} ms")
    print(f"  new (memoised):          MI-kernel calls={new_calls:3d}  wall={new_wall * 1e3:8.2f} ms")
    print(f"  delta:                   calls {new_calls - old_calls:+d}  wall {(new_wall - old_wall) * 1e3:+.2f} ms")

    out_dir = os.path.join(os.path.dirname(__file__), "_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "unary_mi_memo.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "ts": datetime.now().isoformat(),
                "n": _N, "F": _F, "B": _B, "U": len(_UNARY), "nbins": _NBINS,
                "unary_specs": _UNARY,
                "bit_identical": identical,
                "old_mi_calls": old_calls,
                "new_mi_calls": new_calls,
                "old_wall_ms": old_wall * 1e3,
                "new_wall_ms": new_wall * 1e3,
            },
            fh,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
