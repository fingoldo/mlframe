"""The held-out incremental-R^2 probe the support-protection passes admit a re-added column on.

A protection pass re-adds a column the MRMR screen dropped. Whether that is a rescue or a regression depends on the SELECTED design, not on
the column's own history: a leg validated against raw ``x`` in the FE stage can still be fully subsumed by a pair composite that survived the
screen. The probe answers the only question that matters at this point, "does adding this to the design we actually kept buy anything on data
the fit has not seen", and every protection pass in this package decides on it.

It lived as a closure inside the hinge block of ``_group2``, which meant the orth-basis protection reached it through ``locals()`` and the
``_group1`` passes could not reach it at all, so those re-added unconditionally. Building it here lets each pass construct its own against its
own view of the selected set.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence
import logging

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._raw_protect_r2 import heldout_r2_scorer

logger = logging.getLogger(__name__)


def selected_design_columns(*, X: Any, cols: Sequence[str], selected_vars: Sequence[int], eng_continuous_snapshot: dict, y_ref: Optional[np.ndarray]) -> list:
    """Continuous values of the currently-selected columns (engineered from the snapshot, raw from ``X``): the baseline the candidate must beat."""
    out: list = []
    if y_ref is None or not isinstance(X, pd.DataFrame):
        return out
    for name in dict.fromkeys(cols[i] for i in selected_vars if 0 <= i < len(cols)):
        values = eng_continuous_snapshot.get(name)
        if values is None and name in X.columns:
            values = X[name].to_numpy()
        if values is None:
            continue
        try:
            values = np.asarray(values, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            # A raw categorical/string selected column (e.g. under skip_categorical_encoding) is not a numeric R^2-baseline regressor.
            continue
        if values.shape[0] == y_ref.shape[0] and np.all(np.isfinite(values)):
            out.append(values)
    return out


def coerce_gate_target(y_np: Any, n_rows: int) -> Optional[np.ndarray]:
    """``y`` as a finite float64 vector of ``n_rows``, or None when it cannot serve as a regression target for the probe."""
    try:
        y_vec = np.asarray(y_np, dtype=np.float64).reshape(-1)
    except Exception as exc:
        logger.debug("mrmr: y coercion for the held-out protection gate failed: %r", exc, exc_info=True)
        return None
    if y_vec.shape[0] == int(n_rows) and np.all(np.isfinite(y_vec)):
        return y_vec
    return None


def build_heldout_incr_probe(*, y_gate: Optional[np.ndarray], sel_value_cols: list, random_seed: Optional[int]) -> Callable[..., float]:
    """Return ``probe(candidate_vals, src_vals=None) -> float``, the held-out R^2 gain of adding the candidate to the selected design.

    Everything that does not depend on the candidate is done once here: the split, the validation target and its centred sum of squares, and
    the QR factorisation of the selected design's train block. Each candidate then costs one column insert into that factor rather than a
    fresh materialisation of an ``(n, k)`` design and an SVD, which at production width was the dominant cost of the pass.

    Including ``[src, src^2]`` in the baseline when ``src_vals`` is given is the SMOOTH-CURVE guard: a parabola (y=x^2) is captured by
    ``src^2`` so a kink adds ~0 over it and is rejected (no spurious hinge on a smooth target), while a genuine slope change still beats
    ``[src, src^2]`` out of sample because a quadratic cannot fit a sharp two-slope kink, so the hidden-champion leg is kept.

    ``candidate_vals`` may be a single column or a 2-D block of columns that only work together, such as the sin/cos legs of one adaptive
    frequency, whose individual marginals are low by construction because the phase is split across them.
    """
    if y_gate is None:
        # No usable target: the gate cannot decide, and must not silently drop columns.
        return lambda candidate_vals, src_vals=None: 1.0

    n = int(y_gate.shape[0])
    # Seeded shuffle-then-stride, not a raw positional (idx % 3) == 0 split - the latter is not an honest i.i.d. holdout on
    # time/group/label-sorted input (this module explicitly supports sorted input elsewhere via ``groups`` / the ``temporal_agg`` FE
    # family), which can bias the held-out R^2 this gate decides on. The draw depends only on the seed and n, so it is made once.
    perm = np.random.default_rng(int(random_seed or 0)).permutation(n)
    va = np.zeros(n, dtype=bool)
    va[perm[: n // 3]] = True
    tr = ~va
    if int(tr.sum()) < 32 or int(va.sum()) < 16:
        return lambda candidate_vals, src_vals=None: 1.0

    # The scorer slices the base per column, so the full-height selected design is never materialised, and factorises the train block once.
    scorer = heldout_r2_scorer([np.ones(n), *sel_value_cols], y_gate, tr, va)

    def _heldout_incr_over_selected(candidate_vals, src_vals=None) -> float:
        """Held-out R^2 gain of adding ``candidate_vals`` to the selected design plus the source and its degree-2 poly."""
        cand = np.asarray(candidate_vals, dtype=np.float64)
        cand = cand.reshape(-1, 1) if cand.ndim == 1 else cand.reshape(cand.shape[0], -1)
        if cand.shape[0] != n or not np.all(np.isfinite(cand)):
            return 0.0
        smooth: list = []
        if src_vals is not None:
            sv = np.asarray(src_vals, dtype=np.float64).reshape(-1)
            if sv.shape[0] == n and np.all(np.isfinite(sv)):
                # The source is usually a selected column already; repeating it would make the design rank deficient, which the scorer
                # handles by falling back to lstsq, but not adding the duplicate keeps the baseline meaningful in the first place.
                if not any(np.array_equal(sv, col) for col in sel_value_cols):
                    smooth.append(sv)
                smooth.append(sv * sv)
        base_extra = np.column_stack(smooth) if smooth else None
        r2_base = scorer(base_extra)
        r2_full = scorer(np.column_stack([*smooth, cand]) if smooth else cand)
        if not (np.isfinite(r2_base) and np.isfinite(r2_full)):
            # The solve failed, so the uplift is unmeasured, not zero. -inf keeps the reject verdict at every ``< floor`` call site
            # (a NaN would compare False there and admit the candidate) while no longer reading as a measured 0.0.
            logger.debug("mrmr: held-out R^2 probe could not be solved (base=%r, full=%r); rejecting the candidate", r2_base, r2_full)
            return float("-inf")
        return float(r2_full - r2_base)

    return _heldout_incr_over_selected


def candidate_values(name: str, *, X: Any, eng_continuous_snapshot: dict, y_ref: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Continuous values of one candidate column by name, or None when it cannot be scored against ``y_ref``."""
    values = eng_continuous_snapshot.get(name)
    if values is None and isinstance(X, pd.DataFrame) and name in X.columns:
        values = X[name].to_numpy()
    if values is None or y_ref is None:
        return None
    try:
        values = np.asarray(values, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.shape[0] != y_ref.shape[0] or not np.all(np.isfinite(values)):
        return None
    return values
