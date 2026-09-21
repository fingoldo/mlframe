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

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Above this condition number the normal-equations solve is not trustworthy and the probe falls back to lstsq, whose min-norm solution
# handles a rank-deficient design correctly. A well-conditioned design here sits around 1e1; an exactly duplicated column pushes it past 1e17.
_MAX_NORMAL_EQUATION_COND = 1e12


def selected_design_columns(*, X, cols, selected_vars, eng_continuous_snapshot, y_ref) -> list:
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


def coerce_gate_target(y_np, n_rows: int):
    """``y`` as a finite float64 vector of ``n_rows``, or None when it cannot serve as a regression target for the probe."""
    try:
        y_vec = np.asarray(y_np, dtype=np.float64).reshape(-1)
    except Exception as exc:
        logger.debug("mrmr: y coercion for the held-out protection gate failed: %r", exc, exc_info=True)
        return None
    if y_vec.shape[0] == int(n_rows) and np.all(np.isfinite(y_vec)):
        return y_vec
    return None


def build_heldout_incr_probe(*, y_gate, sel_value_cols, random_seed):
    """Return ``probe(candidate_vals, src_vals=None) -> float``, the held-out R^2 gain of adding the candidate to the selected design.

    Including ``[src, src^2]`` in the baseline when ``src_vals`` is given is the SMOOTH-CURVE guard: a parabola (y=x^2) is captured by
    ``src^2`` so a kink adds ~0 over it and is rejected (no spurious hinge on a smooth target), while a genuine slope change still beats
    ``[src, src^2]`` out of sample because a quadratic cannot fit a sharp two-slope kink, so the hidden-champion leg is kept.

    ``candidate_vals`` may be a single column or a 2-D block of columns that only work together, such as the sin/cos legs of one adaptive
    frequency, whose individual marginals are low by construction because the phase is split across them.
    """

    def _heldout_incr_over_selected(candidate_vals, src_vals=None) -> float:
        """Held-out R^2 gain of adding ``candidate_vals`` to the selected design plus the source and its degree-2 poly."""
        if y_gate is None:
            return 1.0  # gate disabled -> fall back to the source-survived rule
        cand = np.asarray(candidate_vals, dtype=np.float64)
        cand = cand.reshape(-1, 1) if cand.ndim == 1 else cand.reshape(cand.shape[0], -1)
        n = cand.shape[0]
        if n != y_gate.shape[0] or not np.all(np.isfinite(cand)):
            return 0.0
        # Seeded shuffle-then-stride, not a raw positional (idx % 3) == 0 split - the latter is not an honest i.i.d. holdout on
        # time/group/label-sorted input (this module explicitly supports sorted input elsewhere via ``groups`` / the ``temporal_agg`` FE
        # family), which can bias the held-out R^2 this gate decides on.
        perm = np.random.default_rng(int(random_seed or 0)).permutation(n)
        va = np.zeros(n, dtype=bool)
        va[perm[: n // 3]] = True
        tr = ~va
        if int(tr.sum()) < 32 or int(va.sum()) < 16:
            return 1.0
        yv = y_gate[va]
        ss = float(np.sum((yv - yv.mean()) ** 2))
        if ss < 1e-24:
            return 0.0
        base = [np.ones(n), *sel_value_cols]
        if src_vals is not None:
            sv = np.asarray(src_vals, dtype=np.float64).reshape(-1)
            if sv.shape[0] == n and np.all(np.isfinite(sv)):
                # The source is usually a selected column itself, and repeating it would make the design exactly rank-deficient. The
                # conditioning check in _r2 would catch that, but not adding the duplicate keeps the baseline meaningful in the first place.
                if not any(np.array_equal(sv, col) for col in sel_value_cols):
                    base = [*base, sv]
                base = [*base, sv * sv]

        def _r2(design_cols) -> float:
            """Fit an OLS design on the train stride and return held-out R^2 on the validation stride (``-inf`` on a singular/failed solve).

            Normal-equations solve on the well-conditioned small-k design (intercept + a handful of base/candidate columns) instead of a full
            SVD lstsq, the same win already proven for this module's sibling OLS fit (``_deflate_sincos`` in ``_orth_extra_basis_fe.py``).
            Falls back to lstsq if A.T@A is singular.
            """
            A = np.column_stack(design_cols)
            A_tr = A[tr]
            y_tr = y_gate[tr]
            try:
                AtA = A_tr.T @ A_tr
                # np.linalg.solve only raises on an exactly zero pivot, so on a merely collinear design it returns a garbage fit instead of
                # failing, and the gate then compares two meaningless R^2 values (a nested full fit can come out WORSE than its own base).
                # Collinear designs are ordinary here: the selected set can hold two copies of the same signal. Decide on the conditioning.
                if np.linalg.cond(AtA) > _MAX_NORMAL_EQUATION_COND:
                    raise np.linalg.LinAlgError("design too ill-conditioned for normal equations")
                coef = np.linalg.solve(AtA, A_tr.T @ y_tr)
            except np.linalg.LinAlgError:
                try:
                    coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
                except Exception as e:
                    logger.debug("Held-out gate OLS lstsq fallback failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                    return -np.inf
            except Exception as e:
                logger.debug("Held-out gate OLS lstsq failed (%s: %s) -- treating as a failed candidate", type(e).__name__, e)
                return -np.inf
            pred = A[va] @ coef
            return 1.0 - float(np.sum((yv - pred) ** 2)) / ss

        r2_base = _r2(base)
        r2_full = _r2([*base, *cand.T])
        if not (np.isfinite(r2_base) and np.isfinite(r2_full)):
            # The solve failed, so the uplift is unmeasured, not zero. -inf keeps the reject verdict at every ``< floor`` call site
            # (a NaN would compare False there and admit the candidate) while no longer reading as a measured 0.0.
            logger.debug("mrmr: held-out R^2 probe could not be solved (base=%r, full=%r); rejecting the candidate", r2_base, r2_full)
            return float("-inf")
        return float(r2_full - r2_base)

    return _heldout_incr_over_selected


def candidate_values(name, *, X, eng_continuous_snapshot, y_ref):
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
