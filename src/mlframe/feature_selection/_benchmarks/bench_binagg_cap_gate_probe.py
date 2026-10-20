"""Probe: does the ``max_pairs`` cap or a gate decide whether a binned-agg column reaches MRMR?

``binned_numeric_agg_with_recipes`` caps the (group, agg) pairs to ``max_pairs`` by (group MI, agg variance) BEFORE any
out-of-fold work, then runs a local-MI gate and a conditional-MI redundancy gate. This script builds the project's own
``sigma(cell)`` scenario (``aux ~ N(0, 0.5 + 2|g - 0.5|)``, target a function of that spread) and adds ``n_decoys`` columns
that carry the target linearly, so decoy groups outrank ``g`` and the pair ``(aux | g)`` falls into the capped-off tail.

Measured on 20000 rows (2026-10-20), ``(aux|qbin(g))`` binned-agg columns that came out:

* 11 decoys, cap 64: 0 before any gate (the cap removed the pair); cap 400: 4 before the gates.
* every decoy count, either cap, default gates: 0. The MI gate keeps 2 of the 24 columns in the no-decoy case and the
  redundancy gate then removes them, because the target is a function of ``g`` and the raw ``g`` already explains it.

So the cap does discard the pair in an adversarial layout, but no layout tried here produced a column that survives the
gates, so the cap cost nothing measurable. A layout where a column survives the gates is still to be found.

Usage: ``python bench_binagg_cap_gate_probe.py [n_rows] [seed]``.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._binned_numeric_agg_fe import binned_numeric_agg_with_recipes


def _frame(n_decoys: int, n: int, seed: int) -> tuple:
    """The sigma(cell) frame plus ``n_decoys`` linear-signal columns; returns ``(X, y)`` with a binary target."""
    rng = np.random.default_rng(seed)
    g = rng.uniform(0, 1, n)
    sigma = 0.5 + 2.0 * np.abs(g - 0.5)
    cols = {f"d{i}": rng.standard_normal(n) for i in range(n_decoys)}
    cols["g"] = g
    cols["aux"] = rng.normal(0, sigma, n)
    X = pd.DataFrame(cols)
    decoy_sum = sum((X[f"d{i}"].to_numpy() for i in range(n_decoys)), 0.0)
    z = (sigma - sigma.mean()) / sigma.std() + 3.0 * decoy_sum + 0.5 * rng.standard_normal(n)
    return X, (z > 0).astype(np.int64)


def _aux_by_g_columns(result) -> list:
    """Names of the ``aux`` aggregated over cells of ``g`` in a ``binned_numeric_agg_with_recipes`` result."""
    frame = result[0] if isinstance(result, tuple) else result
    names = list(frame.columns) if hasattr(frame, "columns") else list(frame)
    return [name for name in names if "binagg" in name and "(aux|qbin(g))" in name]


def main(n: int = 20000, seed: int = 0) -> None:
    """Print how many ``(aux|g)`` columns survive for each decoy count, cap and gate setting."""
    settings = (("no gates", {"mi_gate": False, "redundancy_gate": False}), ("default gates", {}))
    for n_decoys in (0, 3, 11):
        X, y = _frame(n_decoys, n, seed)
        for cap in (64, 400):
            for label, kwargs in settings:
                found = _aux_by_g_columns(binned_numeric_agg_with_recipes(X, y, max_pairs=cap, **kwargs))
                print(f"decoys={n_decoys:2d} cap={cap:3d} {label:13s} (aux|g) columns={len(found)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20000, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
