"""MAKE-OR-BREAK probe for cross-fit recipe warm-start prior (backlog idea #20).

Question: on bootstrap-overlapping subsamples (the CV/partial_fit case), does
seeding the per-pair polynomial CMA-ES with the PRIOR FOLD'S surviving
coefficients reduce iters-to-converge + wall-time vs the existing
ALS+canonical warm-start, WITHOUT changing which coefficients win?

This probes the inner optimiser directly (the load-bearing cost) before any
production plumbing, so we know if there is a win to harvest at all.
"""
import sys, time
import numpy as np

sys.path.insert(0, "src")

from mlframe.feature_selection.filters._hermite_fe_optimise_pair import optimise_hermite_pair
from mlframe.feature_selection.filters import _hermite_fe_optimise as OPT


def make_pair_target(n, seed):
    """Non-monotone-inner pair target: y depends on a NON-MONOTONE warp of a
    times b. This is the regime where the CMA actually has to search (a plain
    mul of identity operands cannot represent it), so warm-start has something
    to save. Returns x_a, x_b, y (binary)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    inner = (a ** 3 - 2.0 * a) * (b ** 2 - 1.0)  # non-monotone in both operands
    noise = rng.normal(size=n) * 0.3
    z = inner + noise
    y = (z > np.median(z)).astype(np.int64)
    return a, b, y


# Capture n_evals from the batch CMA by wrapping it.
_LAST = {"n_evals": None}
_orig_batch = OPT._run_cma_search_batch
def _wrapped_batch(*args, **kw):
    res = _orig_batch(*args, **kw)
    if res is not None:
        _LAST["n_evals"] = res[4]
    return res
OPT._run_cma_search_batch = _wrapped_batch


def run_one(a, b, y, prior_seeds=None, n_trials=300):
    """Run optimise_hermite_pair; optionally inject prior coefficient seeds via
    the module-level hook used by the production warm-start prior. Returns
    (HermiteResult, n_evals, wall_s)."""
    _LAST["n_evals"] = None
    t0 = time.perf_counter()
    res = optimise_hermite_pair(
        x_a=a, x_b=b, y=y, discrete_target=True,
        max_degree=4, min_degree=2, n_trials=n_trials,
        seed=42, sweep_degrees=True, basis="chebyshev",
        optimizer="cma_batch", warm_start=True, warm_start_als=True,
        cross_fit_prior_seeds=prior_seeds,
        multi_fidelity=False,
    )
    wall = time.perf_counter() - t0
    return res, _LAST["n_evals"], wall


def main_extended(n_boot=12):
    """Stricter: more bootstraps, report selection-flip count + iter dist."""
    N = 4000
    a0, b0, y0 = make_pair_target(N, seed=0)
    rng = np.random.default_rng(7)
    res0, ev0, w0 = run_one(a0, b0, y0)
    if res0 is None:
        print("extended: master None"); return
    prior_seeds = [np.concatenate([res0.coef_a, res0.coef_b])]
    flips = 0; warm_better = 0; cold_better = 0
    ce, we = [], []
    for k in range(n_boot):
        idx = rng.choice(N, size=int(0.85 * N), replace=True)
        a, b, y = a0[idx], b0[idx], y0[idx]
        rc, evc, wc = run_one(a, b, y, prior_seeds=None)
        rw, evw, ww = run_one(a, b, y, prior_seeds=prior_seeds)
        ce.append(evc); we.append(evw)
        mic = getattr(rc, "mi", float("nan")); miw = getattr(rw, "mi", float("nan"))
        if not (rc and rw and rc.bin_func_name == rw.bin_func_name and abs(mic - miw) < 1e-9):
            flips += 1
            if miw > mic + 1e-9: warm_better += 1
            elif mic > miw + 1e-9: cold_better += 1
    print(f"\n=== EXTENDED ({n_boot} boots) ===")
    print(f"selection NON-identical on {flips}/{n_boot} boots "
          f"(warm_higher_mi={warm_better}, cold_higher_mi={cold_better})")
    print(f"iters median COLD={np.median(ce):.0f} WARM={np.median(we):.0f}")


def main():
    N = 4000
    # Master fixture; 5 bootstrap subsamples drawn from it (80%+ overlap).
    a0, b0, y0 = make_pair_target(N, seed=0)
    rng = np.random.default_rng(123)

    # COLD reference fit on the master set: gives us the prior coefficients.
    res0, ev0, w0 = run_one(a0, b0, y0)
    print(f"master COLD: n_evals={ev0} wall={w0:.3f}s mi={getattr(res0,'mi',None)}")
    if res0 is None:
        print("master fit returned None -- target too weak; aborting probe")
        return
    prior_seeds = [np.concatenate([res0.coef_a, res0.coef_b])]
    prior_degree = (res0.degree_a, res0.degree_b)
    print(f"prior degree={prior_degree} coef_a={np.round(res0.coef_a,3)} coef_b={np.round(res0.coef_b,3)}")

    cold_evals, warm_evals, cold_walls, warm_walls = [], [], [], []
    cold_mis, warm_mis, identical = [], [], []
    for k in range(5):
        idx = rng.choice(N, size=int(0.85 * N), replace=True)  # ~85% bootstrap overlap
        a, b, y = a0[idx], b0[idx], y0[idx]
        rc, evc, wc = run_one(a, b, y, prior_seeds=None)
        rw, evw, ww = run_one(a, b, y, prior_seeds=prior_seeds)
        mic = getattr(rc, "mi", float("nan"))
        miw = getattr(rw, "mi", float("nan"))
        # bin_func + degrees + sign-of-best identical => same engineered column shape
        same = (rc is not None and rw is not None
                and rc.bin_func_name == rw.bin_func_name
                and abs(mic - miw) < 1e-9)
        cold_evals.append(evc); warm_evals.append(evw)
        cold_walls.append(wc); warm_walls.append(ww)
        cold_mis.append(mic); warm_mis.append(miw); identical.append(same)
        print(f"  boot{k}: COLD evals={evc} wall={wc:.3f} mi={mic:.4f} | "
              f"WARM evals={evw} wall={ww:.3f} mi={miw:.4f} | bf_same+mi_eq={same}")

    def med(x): return float(np.median(x))
    iters_pct = 100.0 * (med(cold_evals) - med(warm_evals)) / med(cold_evals)
    wall_pct = 100.0 * (med(cold_walls) - med(warm_walls)) / med(cold_walls)
    print("\n=== SUMMARY (median across 5 bootstraps) ===")
    print(f"iters  COLD={med(cold_evals):.0f}  WARM={med(warm_evals):.0f}  reduction={iters_pct:+.1f}%")
    print(f"wall   COLD={med(cold_walls):.4f} WARM={med(warm_walls):.4f} reduction={wall_pct:+.1f}%")
    print(f"mi equal on all bootstraps: {all(identical)} (per-boot {identical})")


def main_cprofile():
    """cProfile the WARM path so we can confirm the prior-seed lookup + clip +
    inject is cheap relative to the optimiser it feeds (and the DEFAULT-OFF
    no-fire path costs ~0)."""
    import cProfile, pstats, io
    N = 4000
    a0, b0, y0 = make_pair_target(N, seed=0)
    res0, _, _ = run_one(a0, b0, y0)
    prior = [np.concatenate([res0.coef_a, res0.coef_b])]
    # Warm up JIT once.
    run_one(a0, b0, y0, prior_seeds=prior)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(5):
        run_one(a0, b0, y0, prior_seeds=prior)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print("\n=== cPROFILE (WARM path, 5 fits) top-15 cumulative ===")
    print(s.getvalue())
    # Isolate the injection-block cost: it is a handful of numpy ops per degree.
    # Default-OFF (cross_fit_prior_seeds=None) skips it entirely -> ~0.


if __name__ == "__main__":
    main()
    main_extended()
    main_cprofile()
