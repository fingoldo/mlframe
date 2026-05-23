"""Bench std-based degenerate-init probe vs torch.linalg.matrix_rank (SVD).

Context
-------
``generate_mlp`` in ``src/mlframe/training/neural/flat.py`` runs a per-layer
sanity probe after weights initialisation to catch dead inits
(``zeros_`` / ``constant_`` / scalar) that would silently train a useless
model. The original probe used ``torch.linalg.matrix_rank``, which under the
hood runs full SVD on each Linear's weight matrix -- O(n*m^2). On c0039
(iter256, 10-Linear MLP suite, 2 MLPs trained) cProfile attributed
**7.85 s cumulative** to ``linalg_matrix_rank`` = 5.2pct of the 149.94 s
total wall.

The pathologies the probe was added to catch (zero-init, constant-init,
scalar-broadcast init) all collapse to a zero-std weight matrix. So a
``torch.std(W) < eps`` check covers every common-case pathology this probe
was added to detect in O(n*m) time -- microseconds per layer.

Bench
-----
Local measurement (Windows, Python 3.11, torch 2.x, CPU; 10-layer MLP with
realistic shapes 256-1024-1024-512-128-64-1024-512-256-64-1, kaiming init)::

    Old (matrix_rank): 204.57ms -> ratio=1.000
    New (std):         2.6158ms -> std=4.4147e-02
    Speedup:           78x

Sanity (degenerate detection still works)::

    init=zeros_:   std=0.00e+00 -> DEGENERATE (caught)
    init=kaiming_: std=4.43e-02 -> OK

Coverage trade-off
------------------
What the std-based check still catches:
  * ``torch.nn.init.zeros_`` -> std == 0
  * ``torch.nn.init.constant_(W, c)`` -> std == 0 (for any constant c,
    including c != 0)
  * scalar-broadcast inits (``W[:] = scalar``) -> std == 0

What the std-based check no longer catches:
  * rank-deficient init with non-zero std (e.g. a custom callable that
    sets every row of W to the same random vector). This produces
    ``rank == 1`` with ``std > 0``. The original ``matrix_rank`` probe
    would catch it; the std-based one will not.

Rationale: this "non-zero-std but rank-deficient by construction" case
requires a user-supplied custom init callable that deliberately produces
rank-deficient weights. That is a model-design bug to catch at design
time, not at every fit. Default kaiming / xavier / normal / uniform
inits all produce full-rank matrices with overwhelming probability.

Run: ``python profiling/bench_mlp_rank_probe_std_vs_svd.py``
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


def _build_mlp_modules() -> list[nn.Linear]:
    """Realistic MLP shapes from c0039 fuzz combo."""
    torch.manual_seed(0)
    sizes = [
        (256, 1024),
        (1024, 1024),
        (1024, 512),
        (512, 128),
        (128, 64),
        (64, 1024),
        (1024, 512),
        (512, 256),
        (256, 64),
        (64, 1),
    ]
    modules = []
    for in_d, out_d in sizes:
        m = nn.Linear(in_d, out_d)
        nn.init.kaiming_normal_(m.weight)
        modules.append(m)
    return modules


def probe_svd(modules: list[nn.Linear]) -> float:
    """Old path: full SVD-based rank check."""
    worst = 1.0
    for m in modules:
        with torch.no_grad():
            W = m.weight.detach()
            r = int(torch.linalg.matrix_rank(W).item())
            ratio = r / max(min(W.shape), 1)
            worst = min(worst, ratio)
    return worst


def probe_std(modules: list[nn.Linear]) -> float:
    """New path: std-based degenerate check."""
    worst = float("inf")
    for m in modules:
        with torch.no_grad():
            W = m.weight.detach()
            std = float(torch.std(W, unbiased=False).item())
            worst = min(worst, std)
    return worst


def main() -> None:
    modules = _build_mlp_modules()

    # Warmup
    probe_svd(modules)
    probe_std(modules)

    n_iter = 5
    t0 = time.perf_counter()
    for _ in range(n_iter):
        r_svd = probe_svd(modules)
    t1 = time.perf_counter()
    for _ in range(n_iter):
        r_std = probe_std(modules)
    t2 = time.perf_counter()

    svd_ms = (t1 - t0) * 1000 / n_iter
    std_ms = (t2 - t1) * 1000 / n_iter
    print(f"SVD probe:  {svd_ms:.2f}ms/run (ratio={r_svd:.3f})")
    print(f"std probe:  {std_ms:.4f}ms/run (std={r_std:.4e})")
    print(f"Speedup:    {svd_ms / std_ms:.0f}x")

    print()
    print("Sanity -- degenerate detection still works:")
    for init_name, init_fn in [
        ("zeros_", nn.init.zeros_),
        ("constant_(0.5)", lambda w: nn.init.constant_(w, 0.5)),
        ("kaiming_normal_", nn.init.kaiming_normal_),
        ("xavier_uniform_", nn.init.xavier_uniform_),
    ]:
        mods = []
        for in_d, out_d in [(256, 1024), (1024, 512), (512, 1)]:
            m = nn.Linear(in_d, out_d)
            init_fn(m.weight)
            mods.append(m)
        s = probe_std(mods)
        verdict = "DEGENERATE (caught)" if s < 1e-8 else "OK"
        print(f"  init={init_name:20s}: std={s:.2e} -> {verdict}")


if __name__ == "__main__":
    main()
