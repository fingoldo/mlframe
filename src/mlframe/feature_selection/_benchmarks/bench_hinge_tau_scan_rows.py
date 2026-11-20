"""Old (all-row) vs corrected (train-row) hinge tau scan: chosen-tau shift and held-out-gate pass rate."""
from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters._hinge_basis_fe import _detect_hinge_breakpoints


def _cases():
    """Yield (name, generator(rng, n) -> (x, y)) fixtures: planted kinks of varying strength, plus pure noise."""
    yield "kink_strong_n4000", lambda r, n: _kink(r, n, 5.0, 0.1)
    yield "kink_mid_n4000", lambda r, n: _kink(r, n, 1.0, 0.3)
    yield "kink_weak_n2000", lambda r, n: _kink(r, n, 0.5, 0.5)
    yield "noise_n2000", lambda r, n: (r.standard_normal(n), r.standard_normal(n))


def _kink(rng, n, slope, noise):
    """Uniform x with a relu leg at 0.7 of the given slope."""
    x = rng.uniform(0, 1, n)
    return x, 2 * x + slope * np.maximum(x - 0.7, 0.0) + noise * rng.standard_normal(n)


def main(seeds: int = 40) -> None:
    """Print per fixture: share of seeds with a hinge, mean |tau-0.7| old vs new, share of seeds where tau changed."""
    for name, gen in _cases():
        n = int(name.split("_n")[-1])
        rows = []
        for s in range(seeds):
            x, y = gen(np.random.default_rng(s), n)
            old = _detect_hinge_breakpoints(x, y, max_breakpoints=1, scan_on_train_rows=False)
            new = _detect_hinge_breakpoints(x, y, max_breakpoints=1)
            rows.append((old, new))
        po = np.mean([bool(o) for o, _ in rows]); pn = np.mean([bool(w) for _, w in rows])
        both = [(o[0], w[0]) for o, w in rows if o and w]
        chg = np.mean([abs(a - b) > 1e-12 for a, b in both]) if both else float("nan")
        eo = np.mean([abs(a - 0.7) for a, _ in both]) if both else float("nan")
        en = np.mean([abs(b - 0.7) for _, b in both]) if both else float("nan")
        print(f"{name}: hinge_rate old={po:.3f} new={pn:.3f} tau_changed_share={chg:.3f} mean|tau-0.7| old={eo:.4f} new={en:.4f} (n_both={len(both)})")


if __name__ == "__main__":
    main()
