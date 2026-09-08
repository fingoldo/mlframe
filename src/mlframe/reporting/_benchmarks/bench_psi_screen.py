"""Selection-equivalence and speed for the two-pass PSI screen in ``charts/drift.py``.

The screen only chooses WHICH features are drawn -- every drawn cell is still computed on all rows -- so
the bar it has to clear is that the drawn feature SET does not change. Run across a spread of drift shapes
because a screen that is right on a broad mean shift can still miss a drift that lives in one narrow bucket.
"""

from __future__ import annotations

import time

import numpy as np


from mlframe.reporting.charts.drift import compute_psi_matrix


def _shapes(rng, n, ncols):
    """Frames whose drift lives in different places: broad, late-only, one-bucket spike, variance-only, none."""
    ts = np.sort(rng.random(n))
    out = {}
    frac = np.linspace(0.0, 1.0, n)
    base = rng.standard_normal((n, ncols))
    for name in ("broad", "late", "spike", "variance", "none"):
        x = base.copy()
        drifted = rng.choice(ncols, size=max(1, ncols // 4), replace=False)
        if name == "broad":
            x[:, drifted] += frac[:, None] * 2.0
        elif name == "late":
            x[:, drifted] += (frac > 0.8)[:, None] * 3.0
        elif name == "spike":
            win = (frac > 0.45) & (frac < 0.55)
            x[:, drifted] += win[:, None] * 4.0
        elif name == "variance":
            x[:, drifted] *= 1.0 + frac[:, None] * 4.0
        out[name] = (x, ts)
    return out


def main():
    """Drawn-set agreement and paired wall time, screened against exact, per drift shape.

    Trials are interleaved rather than run in blocks: this box is often busy with other work, and a block
    layout charges whichever half happened to overlap the load.
    """
    rng = np.random.default_rng(20260907)
    for n, ncols, maxf in ((200_000, 200, 40), (1_000_000, 200, 40), (200_000, 60, 40)):
        for name, (x, ts) in _shapes(rng, n, ncols).items():
            t_exact, t_screen = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                m_e, names_e, _ = compute_psi_matrix(x, ts, max_features=maxf, screen_rows=0)
                t_exact.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                m_s, names_s, _ = compute_psi_matrix(x, ts, max_features=maxf)
                t_screen.append(time.perf_counter() - t0)
            e, sc = float(np.median(t_exact)), float(np.median(t_screen))
            same = set(names_e) == set(names_s)
            diff = np.nanmax(np.abs(m_e - m_s)) if names_e == names_s else float("nan")
            print(
                f"n={n:>9,} cols={ncols:>3} {name:<9} same_set={same!s:<5} "
                f"missed={len(set(names_e) - set(names_s)):>2} maxdiff={diff:.2e} "
                f"exact={e:6.2f}s screen={sc:6.2f}s speedup={e / max(sc, 1e-9):5.2f}x"
            )


if __name__ == "__main__":
    main()
