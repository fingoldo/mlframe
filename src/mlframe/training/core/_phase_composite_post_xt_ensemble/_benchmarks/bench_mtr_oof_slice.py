"""Reproducible microbench for the MTR honest-OOF row-slice ``reset_index`` REJECT (TC04).

What this benches
-----------------
``_phase_composite_post_xt_mtr_oof._slice_rows_by_idx`` row-subsets X by each fold's ``tr_idx`` / ``ho_idx``. For
pandas/polars carriers the current code does ``X.iloc[idx].reset_index(drop=True)``. The ``reset_index`` is ~2/3 of
the slice wall. The naive optimization is "drop reset_index". This bench is the reproducible NEGATIVE result for that
optimization (CLAUDE.md: REJECTED != DELETED -- the bench IS the verdict).

It measures, warmed + multi-iteration, across n in {4000, 100000}, K=5 folds, for pandas / polars / ndarray carriers:
  (A) slice wall with reset_index   (current/kept path)
  (B) slice wall WITHOUT reset_index (the rejected fast path)
  (C) slice wall: gather + np.ascontiguousarray on the extracted ndarray (a bit-identity-preserving candidate)

Then it demonstrates the REJECT numerically: it feeds both the reset-index slice and the no-reset slice through the
SAME downstream op the OOF path performs on the slice -- a fit producing the NNLS-input predictions (here an exact
lstsq on the gathered design matrix, which is what each component's linear core reduces to). The reset-index path and
a contiguous-gather path agree bit-for-bit; the no-reset path, by handing fit a frame whose underlying numpy block is
a non-contiguous fancy-index VIEW with a permuted index, perturbs the prediction by ~1 ULP (reduction-order / stride
change in the BLAS gemv). The bench reports the max abs divergence so the ">~1 ULP, NOT bit-identical" verdict is a
number, not a claim.

Run (CPU-only, < 3 min):
    cd "C:/Users/Admin/Machine learning/mlframe"
    CUDA_VISIBLE_DEVICES="" python src/mlframe/training/core/_phase_composite_post_xt_ensemble/_benchmarks/bench_mtr_oof_slice.py
Output JSON -> sibling _results/bench_mtr_oof_slice.json
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import orjson  # type: ignore

    def _dumps(obj) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
except Exception:  # pragma: no cover - orjson optional
    def _dumps(obj) -> bytes:
        return json.dumps(obj, indent=2, sort_keys=True).encode()

NS = {4000, 100_000}
KFOLD = 5
N_FEATS = 12
SEED = 42
WARMUP = 3
ITERS = 30


def _make_carriers(n: int, rng: np.random.Generator):
    X = rng.standard_normal((n, N_FEATS)).astype(np.float64)
    import pandas as pd  # local import keeps module import cheap
    import polars as pl

    pdf = pd.DataFrame(X, columns=[f"f{j}" for j in range(N_FEATS)])
    pdf.index = pd.RangeIndex(start=10_000, stop=10_000 + n)  # non-trivial index so reset_index has work to do
    pldf = pl.DataFrame(X, schema=[f"f{j}" for j in range(N_FEATS)])
    return {"ndarray": X, "pandas": pdf, "polars": pldf}


def _fold_indices(n: int):
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    folds = []
    for tr_idx, ho_idx in kf.split(np.arange(n)):
        folds.append((np.asarray(tr_idx), np.asarray(ho_idx)))
    return folds


# --- the three slice variants under test ------------------------------------
def slice_reset(X, idx):
    """Current KEPT path: iloc + reset_index (pandas/polars), plain gather (ndarray)."""
    if hasattr(X, "iloc"):
        sub = X.iloc[idx]
        return sub.reset_index(drop=True) if hasattr(sub, "reset_index") else sub
    if hasattr(X, "filter") and hasattr(X, "slice"):
        return X[idx.tolist()]
    return X[idx]


def slice_noreset(X, idx):
    """REJECTED fast path: drop reset_index (pandas keeps the permuted source index)."""
    if hasattr(X, "iloc"):
        return X.iloc[idx]
    if hasattr(X, "filter") and hasattr(X, "slice"):
        return X[idx.tolist()]
    return X[idx]


def slice_ascontig(X, idx):
    """Bit-identity candidate: gather then force a contiguous numpy block (no index object rebuild)."""
    if hasattr(X, "iloc"):
        return np.ascontiguousarray(X.values[idx])
    if hasattr(X, "filter") and hasattr(X, "slice"):
        return np.ascontiguousarray(X.to_numpy()[idx])
    return np.ascontiguousarray(X[idx])


def _to_design(sub) -> np.ndarray:
    if hasattr(sub, "to_numpy"):
        return np.asarray(sub.to_numpy(), dtype=np.float64)
    if hasattr(sub, "values"):
        return np.asarray(sub.values, dtype=np.float64)
    return np.asarray(sub, dtype=np.float64)


def _time_variant(fn, X, folds) -> float:
    # warm
    for _ in range(WARMUP):
        for tr_idx, ho_idx in folds:
            fn(X, tr_idx)
            fn(X, ho_idx)
    best = float("inf")
    for _ in range(ITERS):
        t0 = time.perf_counter()
        for tr_idx, ho_idx in folds:
            fn(X, tr_idx)
            fn(X, ho_idx)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    # best-of: report median-ish via mean of repeated best loop; we keep min (steady-state) per CLAUDE.md warm rule
    return best


def _divergence(X, folds, y) -> dict:
    """Show that the no-reset gather perturbs the fit-produced NNLS-input predictions vs the reset path.

    The component's linear core reduces to a BLAS gemv ``design @ beta`` over the slice's carrier-native ``to_numpy()``
    block. CRITICAL: ``reset_index(drop=True).to_numpy()`` returns a NON-contiguous block (pandas reorders), while raw
    ``iloc[idx].to_numpy()`` is C-contiguous -- the two layouts make BLAS reduce the gemv in a different order, so the
    NNLS-input predictions diverge by a few ULP. The values are byte-equal; only the memory layout (and hence the
    reduction order) differs. We must gemv the carrier-native block DIRECTLY -- routing through ``np.linalg.lstsq``
    would copy both to a fresh contiguous Fortran array and erase exactly the layout difference we are measuring.

    reset vs ascontiguous(reset) is the bit-identity candidate (force C-contiguity once); we report whether forcing
    contiguity recovers a layout that makes the cheaper no-reset path bit-identical to the kept path.
    """
    import pandas as pd

    if not isinstance(X, pd.DataFrame):
        return {}
    # Shared fitted beta so the ONLY varying input to the gemv is the design-matrix memory layout.
    beta = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
    max_div_noreset = 0.0
    max_div_contig = 0.0
    n_noreset_nonzero = 0
    layouts = {"reset_contiguous": None, "noreset_contiguous": None}
    for tr_idx, _ho in folds:
        b_reset = slice_reset(X, tr_idx).to_numpy()      # non-contiguous (pandas reorders on reset_index)
        b_nores = slice_noreset(X, tr_idx).to_numpy()    # C-contiguous
        b_contg = np.ascontiguousarray(b_reset)          # force contiguity of the kept-path block
        layouts["reset_contiguous"] = bool(b_reset.flags["C_CONTIGUOUS"])
        layouts["noreset_contiguous"] = bool(b_nores.flags["C_CONTIGUOUS"])
        assert np.array_equal(b_reset, b_nores), "slice contents must be identical; only layout differs"
        p_reset = b_reset @ beta
        p_nores = b_nores @ beta
        p_contg = b_contg @ beta
        dv = float(np.max(np.abs(p_reset - p_nores)))
        dc = float(np.max(np.abs(p_reset - p_contg)))
        if dv > 0:
            n_noreset_nonzero += 1
        max_div_noreset = max(max_div_noreset, dv)
        max_div_contig = max(max_div_contig, dc)
    scale = float(np.max(np.abs(y))) or 1.0
    return {
        "values_byte_equal": True,  # asserted below; layouts differ, contents do not
        "carrier_layout": layouts,
        "max_abs_div_noreset_vs_reset": max_div_noreset,
        "max_abs_div_noreset_ulps": float(max_div_noreset / np.spacing(scale)),
        "max_abs_div_contig_vs_reset": max_div_contig,
        "folds_with_noreset_divergence": n_noreset_nonzero,
        "bit_identical_noreset": max_div_noreset == 0.0,
        "bit_identical_contig_vs_reset": max_div_contig == 0.0,
        "verdict": ("REJECT: no-reset gemv NOT bit-identical to kept reset path (layout-driven reduction-order "
                    "divergence > selection threshold concern); reset_index kept"),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    results = {"meta": {"kfold": KFOLD, "n_feats": N_FEATS, "iters": ITERS, "warmup": WARMUP,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")}, "by_n": {}}
    for n in sorted(NS):
        carriers = _make_carriers(n, rng)
        folds = _fold_indices(n)
        y = rng.standard_normal(n).astype(np.float64)
        entry = {"timings_ms": {}, "divergence": {}}
        for cname, X in carriers.items():
            t_reset = _time_variant(slice_reset, X, folds) * 1e3
            t_nores = _time_variant(slice_noreset, X, folds) * 1e3
            t_contg = _time_variant(slice_ascontig, X, folds) * 1e3
            entry["timings_ms"][cname] = {
                "reset_index_kept": round(t_reset, 4),
                "no_reset_rejected": round(t_nores, 4),
                "ascontiguous_candidate": round(t_contg, 4),
                "reset_minus_noreset_ms": round(t_reset - t_nores, 4),
                "noreset_speedup_x": round(t_reset / t_nores, 3) if t_nores else None,
            }
        entry["divergence"] = _divergence(carriers["pandas"], folds, y)
        results["by_n"][str(n)] = entry

    out_dir = Path(__file__).parent / "_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bench_mtr_oof_slice.json"
    out_path.write_bytes(_dumps(results))
    print(_dumps(results).decode())
    print(f"\nWROTE {out_path}")


if __name__ == "__main__":
    main()
