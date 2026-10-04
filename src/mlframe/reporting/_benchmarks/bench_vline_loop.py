"""add_vline cost for the 1 (one-sided) and 2 (symmetric) reference lines the plotly bar renderer can draw; the loop is hard-bounded at 2 iterations."""

from __future__ import annotations

import time

from plotly.subplots import make_subplots


def main() -> None:
    """Best-of-7 wall time of k add_vline calls on a 2x2 subplot grid."""
    for k in (1, 2, 50):
        best = 1e9
        for _ in range(7):
            fig = make_subplots(rows=2, cols=2)
            t = time.perf_counter()
            for i in range(k):
                fig.add_vline(x=i, row=1, col=1, line=dict(color="red", dash="dash", width=1.3))
            best = min(best, time.perf_counter() - t)
        print(f"k={k} {best * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
