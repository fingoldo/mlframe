"""Bootstrap stability-selection wrapper around :func:`cascade_select`.

A single ``cascade_select`` run's ``final_selected`` set is sensitive to which rows happened to land in the
CV folds -- a borderline feature (weak true signal, or a noise column that got lucky) can flip in/out of the
final set purely because of the row sample, not because of its real usefulness. Stability selection (Meinshausen
& Buhlmann, 2010) addresses this generically: rerun the whole selection procedure over many bootstrap
row-resamples and only keep features that were selected in at least a chosen fraction of the runs. This module
reuses :func:`cascade_select` unmodified as the inner selector, run B times over independent bootstrap resamples.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .cascade_select import cascade_select


def cascade_select_stable(
    X: Any,
    y: np.ndarray,
    estimator_factory: Callable[[], Any],
    n_bootstrap: int = 20,
    stability_threshold: float = 0.6,
    bootstrap_random_state: int = 0,
    **cascade_kwargs: Any,
) -> Dict[str, Any]:
    """Run :func:`cascade_select` across ``n_bootstrap`` row-resamples and keep only stable features.

    Opt-in wrapper: with ``n_bootstrap`` left at its default of 1 run's worth of behavior disabled (i.e. when
    this function is simply not called), ``cascade_select`` itself is untouched -- this module only orchestrates
    repeated calls to it, it does not alter ``cascade_select``'s own logic or defaults in any way.

    Parameters
    ----------
    X, y, estimator_factory
        Passed through to each :func:`cascade_select` call.
    n_bootstrap
        Number of bootstrap row-resamples (sampling with replacement, same size as ``X``) to run the full
        cascade over.
    stability_threshold
        A feature is kept in ``stable_selected`` only if it appears in ``final_selected`` in at least this
        fraction of the ``n_bootstrap`` runs (e.g. ``0.6`` -> selected in >=60% of runs).
    bootstrap_random_state
        Seed for drawing the bootstrap row indices (independent of ``cascade_kwargs["random_state"]``, which
        still seeds each inner cascade run's own Boruta/RFECV randomness).
    cascade_kwargs
        Extra keyword arguments forwarded verbatim to every :func:`cascade_select` call (``n_boruta_iterations``,
        ``forward_max_features``, ``cv``, ``scoring``, ``random_state``, ``rfecv_kwargs``, ...).

    Returns
    -------
    dict
        ``run_results`` (list of the raw per-run :func:`cascade_select` outputs), ``selection_frequency`` (dict
        mapping every feature name that appeared in any run's ``final_selected`` to its selection fraction),
        ``stable_selected`` (list, features at/above ``stability_threshold``, in decreasing-frequency order).
    """
    if not hasattr(X, "columns") or not hasattr(X, "iloc"):
        raise TypeError("cascade_select_stable: X must be a pandas DataFrame with named columns.")
    if n_bootstrap < 1:
        raise ValueError(f"cascade_select_stable: n_bootstrap must be >=1, got {n_bootstrap}.")
    if not 0.0 < stability_threshold <= 1.0:
        raise ValueError(f"cascade_select_stable: stability_threshold must be in (0, 1], got {stability_threshold}.")

    rng = np.random.default_rng(bootstrap_random_state)
    n_rows = len(X)

    run_results: List[Dict[str, Any]] = []
    selection_counts: Dict[str, int] = {}

    for _ in range(n_bootstrap):
        row_idx = rng.integers(0, n_rows, size=n_rows)
        X_boot = X.iloc[row_idx].reset_index(drop=True)
        y_boot = np.asarray(y)[row_idx]

        result = cascade_select(X_boot, y_boot, estimator_factory, **cascade_kwargs)
        run_results.append(result)

        for feature in result["final_selected"]:
            selection_counts[feature] = selection_counts.get(feature, 0) + 1

    selection_frequency = {feature: count / n_bootstrap for feature, count in selection_counts.items()}
    stable_selected = sorted(
        (feature for feature, freq in selection_frequency.items() if freq >= stability_threshold),
        key=lambda feature: selection_frequency[feature],
        reverse=True,
    )

    return {
        "run_results": run_results,
        "selection_frequency": selection_frequency,
        "stable_selected": stable_selected,
    }


__all__ = ["cascade_select_stable"]
