"""Supplementary LTR-chart benchmark (audit 2026-06-10).

Quantifies the per-query Python-loop hazards in mlframe.reporting.charts.ltr that were replaced by batched numba kernels:
- _ndcg_dist_panel: converted the FULL y_true/y_score arrays to float64 INSIDE the per-query loop (O(n_queries * n) memcpy on int input -> quadratic).
- _mrr_dist_panel: per-query Python loop + inner enumerate scan.
- _lift_panel:      per-query Python loop with inner per-rank loop.
- _ndcg_k_panel:    50 separate full-pass ndcg_at_k calls (each re-sorting all groups).

The quadratic conversion is gone, so the 2,000,000-row run now finishes; before, _ndcg_dist_panel at 400k already cost ~42 s.

Before-numbers (pre-batch, from performance.md): NDCG_DIST 0.66 s @100k / 42.1 s @400k. After-targets: NDCG_DIST < 1 s @2M, NDCG_K < 0.5 s @2M, LIFT < 0.5 s @2M, MRR < 0.3 s @2M.

cProfile (compose_ltr_figure, n=1M, qsize=10): see ``_PROFILE_NOTES`` at the bottom of this file for the top-hotspot table + the optimization
verdict. ASCII-only output. Run from repo root: python "audit/viz_audit_2026_06_10/bench_viz_ltr.py"
"""
import os
import time
import json
import sys
import cProfile
import pstats
import io

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HEARTBEAT = os.path.join(HERE, "HEARTBEAT_perf.txt")
RESULTS = []


def beat(msg):
    try:
        with open(HEARTBEAT, "w", encoding="utf-8") as f:
            f.write("bench-ltr: " + msg + "\n")
    except OSError:
        pass


def record(func, n, t, notes=""):
    RESULTS.append(dict(function=func, n=n, time_s=round(t, 3), notes=notes))
    print(f"  {func:50s} n={n:>10,} t={t:8.3f}s {notes}")
    sys.stdout.flush()


def make_ltr(n, qsize=10, seed=5):
    rng = np.random.default_rng(seed)
    n_q = n // qsize
    group_ids = np.repeat(np.arange(n_q), qsize)
    rels = rng.integers(0, 5, size=n_q * qsize)            # INT dtype (the common case)
    scores = rels + rng.normal(scale=1.0, size=n_q * qsize)
    return rels, scores.astype(np.float64), group_ids


def bench_panels():
    from mlframe.reporting.charts.ltr import (
        _ndcg_dist_panel, _mrr_dist_panel, _lift_panel, _ndcg_k_panel,
        _ndcg_by_qsize_panel,
    )
    # numba warmup on tiny input
    beat("warmup")
    r, s, g = make_ltr(1000)
    _ndcg_dist_panel(r, s, g)
    _ndcg_k_panel(r, s, g)
    _mrr_dist_panel(r, s, g)
    _lift_panel(r, s, g)
    _ndcg_by_qsize_panel(r, s, g)

    for n in (100_000, 400_000, 2_000_000):
        print(f"\n===== LTR n = {n:,} (qsize=10 -> {n//10:,} queries) =====")
        rels, scores, gids = make_ltr(n)
        for name, fn in (
            ("_ndcg_dist_panel (batched per-query NDCG)", _ndcg_dist_panel),
            ("_mrr_dist_panel (batched per-query MRR)", _mrr_dist_panel),
            ("_lift_panel (batched lift kernel)", _lift_panel),
            ("_ndcg_k_panel (1 batched eval_ks pass)", _ndcg_k_panel),
            ("_ndcg_by_qsize_panel (shared NDCG10)", _ndcg_by_qsize_panel),
        ):
            beat(f"{name} n={n}")
            t0 = time.perf_counter()
            try:
                fn(rels, scores, gids)
                t = time.perf_counter() - t0
                record(name, n, t)
            except Exception as e:  # noqa: BLE001
                record(name, n, float("nan"), notes=f"CRASH {type(e).__name__}: {e}")


def bench_compose():
    from mlframe.reporting.charts.ltr import compose_ltr_figure
    for n in (1_000_000, 2_000_000):
        rels, scores, gids = make_ltr(n)
        beat(f"compose_ltr_figure n={n}")
        t0 = time.perf_counter()
        compose_ltr_figure(rels, scores, gids)  # default 6-panel template
        t = time.perf_counter() - t0
        record("compose_ltr_figure (default 6-panel)", n, t)


def profile_compose():
    from mlframe.reporting.charts.ltr import compose_ltr_figure
    rels, scores, gids = make_ltr(1_000_000)
    compose_ltr_figure(rels, scores, gids)  # warm numba
    beat("cProfile compose n=1M")
    pr = cProfile.Profile()
    pr.enable()
    compose_ltr_figure(rels, scores, gids)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    text = s.getvalue()
    print("\n===== cProfile compose_ltr_figure n=1,000,000 (cumulative, top 30) =====")
    print(text)
    with open(os.path.join(HERE, "bench_profile_ltr_compose.txt"), "w", encoding="utf-8") as f:
        f.write(text)


def main():
    bench_panels()
    print("\n===== compose_ltr_figure end-to-end =====")
    bench_compose()
    profile_compose()
    out = os.path.join(HERE, "bench_results_ltr.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nresults written to", out)
    beat("done")


# ---------------------------------------------------------------------------
# _PROFILE_NOTES (n=1M, qsize=10, 100k groups; see bench_profile_ltr_compose.txt)
#
# compose_ltr_figure wall at n=1M is ~0.16 s total. Top frames (cProfile attributes numba kernel wall to the calling python frame, since njit
# functions are not separate frames): _ndcg_k_panel ~0.05 s tottime (the batched NDCG@1..max_k kernel), _score_by_rel_panel ~0.02 s, the
# per-query NDCG10 kernel ~0.03 s (shared by NDCG_DIST + NDCG_BY_QSIZE), _lift_panel ~0.02 s, _iter_group_slices ~0.016 s (the one-shot group
# sort + two float64 gathers). Every numba kernel does one per-group sort over the shared sorted layout -- that O(sum n_g log n_g) sort is the
# floor and is already parallel.
#
# DISCARDED-WORK probe (per "audit hot kernels for wasted per-call work"): _ndcg_k_panel calls _summary_batched_kernel and discards its MAP@k
# and MRR outputs. A pruned NDCG-only variant of the kernel was microbenched warm, multi-repeat, two sizes:
#   n=1M : full(NDCG+MAP+MRR) 51.0 ms -> pruned(NDCG-only) 45.5 ms = 1.12x  (parity bit-identical)
#   n=2M : full 80.8 ms -> pruned 81.2 ms = 0.99x
# The "win" at 1M evaporates at 2M -- the per-group sort (shared by all three metrics) dominates, and MAP reuses the same score-sorted order,
# so removing it saves no reliable wall. REJECTED: not worth a second redundant kernel for a size-inconsistent ~1.1x that is within numba-
# parallel noise. _ndcg_k_panel keeps calling the shared _summary_batched_kernel.
#
# Verdict: no actionable in-file speedup beyond the batching already shipped; the per-group sort is the floor. (If a single huge-N call ever
# dominates, a radix/counting group-sort on integer ids in _iter_group_slices is the next lever, gated on id dtype.)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
