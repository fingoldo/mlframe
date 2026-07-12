"""MRMR usability-aware multi-list POST-PASS (wiring layer, 2026-06-13).

``MRMR.fit`` produces ``support_`` -- a pure-MI selection that is the right
objective for nonlinear / tree downstreams (they build interactions internally;
MI relevance + redundancy is model-agnostic). It is NOT the right list for a
LINEAR / additive downstream: MI is rank-based and blind to linear usability, so
it ranks a high-MI monotone warp above the lower-MI but linearly-aligned
interaction a linear model needs (measured on F2: linear test MAE 0.096 with the
pure-MI list ``[d, c, a**2/b]`` -- which has raw c and d but no c*d interaction --
vs ~0.05 once a ``mul(log(c),sin(d))``-shaped form is selected).

This module runs the standalone ``select_usability_aware_features`` (see
``_usability_aware_selection.py``) as a SECOND pass over a freshly-built
candidate pool and stores TWO additional selections on the fitted estimator:

* ``support_linear_``  -- ``w -> 1`` usability-only relevance (for linear / additive models).
* ``support_universal_`` -- a blended ``MI + lambda*usability`` list (a universal / linear-leaning list).

plus ``support_nonlinear_`` (an alias for the existing ``support_``). Each entry
is a ``UsableCandidate`` carrying a replayable ``EngineeredRecipe`` (or ``None``
for a raw column), so ``MRMR.transform_usability(X, which=...)`` can reproduce the
exact feature space on test data.

The pass is OFF by default (``usability_aware_lists=False``): the CV-MAE forward
selection it runs costs seconds-to-minutes, which must not be charged to every
fit. The suite turns it on and routes linear/additive models to the linear list.
``support_`` is never touched -- the existing tree pipelines stay byte-identical.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _raw_numeric_frame(X: Any, feature_names: list[str]):
    """Return a pandas DataFrame of the CONTINUOUS-numeric raw columns of ``X`` (named by
    ``feature_names``), dropping non-numeric / all-constant columns. ``X`` may be a DataFrame or a
    2-D ndarray aligned to ``feature_names``."""
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        df = X
    else:
        arr = np.asarray(X)
        if arr.ndim != 2 or arr.shape[1] != len(feature_names):
            raise ValueError("ndarray X shape does not match feature_names; cannot build raw frame")
        df = pd.DataFrame(arr, columns=list(feature_names))

    keep: dict[str, np.ndarray] = {}
    for name in feature_names:
        if name not in df.columns:
            continue
        col = pd.to_numeric(df[name], errors="coerce")
        v = np.asarray(col, dtype=np.float64)
        if not np.isfinite(v).any():
            continue
        _n_coerced = int((~np.isfinite(v)).sum())
        if _n_coerced:
            logger.warning(
                "usability: raw column %r had %d/%d non-finite/unparseable value(s) coerced to 0.0.",
                name, _n_coerced, v.size,
            )
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if float(np.nanstd(v)) < 1e-12:
            continue  # constant column carries no usability signal
        keep[name] = v
    if not keep:
        raise ValueError("no continuous-numeric raw columns available for the usability pass")
    return pd.DataFrame(keep)


def _scope_base_names(df, y_cont: np.ndarray, max_base_features: int) -> list[str]:
    """Cap the base-name set to the ``max_base_features`` columns with the highest marginal MI with
    the (binned) target, so the O(p^2) pair enumeration stays tractable on wide inputs. For small p
    (<= cap) every column is kept."""
    names = list(df.columns)
    if len(names) <= max_base_features:
        return names
    from ._usability_aware_selection import _binned_mi
    from ._mi_greedy_cmi_fe import _quantile_bin

    y_codes = _quantile_bin(np.asarray(y_cont, dtype=np.float64), 10)
    scored = [(name, _binned_mi(df[name].to_numpy(), y_codes, 10)) for name in names]
    scored.sort(key=lambda t: t[1], reverse=True)
    return [name for name, _ in scored[:max_base_features]]


def build_usability_lists(mrmr: Any, X: Any, y_cont: "np.ndarray | None") -> None:
    """Compute ``support_linear_`` / ``support_universal_`` (and the ``support_nonlinear_`` alias)
    on the fitted ``mrmr`` from a fresh usability-aware candidate pool. No-op (lists set to ``None``)
    when ``y_cont`` is unavailable (non-numeric target) or no usable raw columns exist. Never raises
    into the fit -- the caller guards it, but degenerate inputs short-circuit cleanly here too."""
    # support_nonlinear_ is always the existing pure-MI selection (alias, not a copy of the array's
    # identity-sensitive semantics -- callers read it as "the tree list").
    mrmr.support_nonlinear_ = getattr(mrmr, "support_", None)
    mrmr.support_linear_ = None
    mrmr.support_universal_ = None

    if y_cont is None:
        return
    y_cont = np.asarray(y_cont, dtype=np.float64).ravel()
    if y_cont.size == 0 or not np.isfinite(y_cont).any() or float(np.nanstd(y_cont)) < 1e-12:
        return

    # feature_names_in_ is an ndarray; "or []" would test truthiness and raise on a multi-element array.
    feature_names = list(getattr(mrmr, "feature_names_in_", []))
    if not feature_names:
        return
    df = _raw_numeric_frame(X, feature_names)
    if df.shape[0] != y_cont.size:
        return  # row mismatch (e.g. target-row dropping); skip rather than misalign

    from ._usability_aware_selection import build_usability_candidate_pool, usability_greedy

    max_base = int(getattr(mrmr, "usability_max_base_features", 16) or 16)
    base_names = _scope_base_names(df, y_cont, max_base)

    pool_kwargs = dict(getattr(mrmr, "usability_pool_kwargs", None) or {})
    pool_kwargs.setdefault("feature_dtype", getattr(mrmr, "usability_feature_dtype", np.float32))
    pool_kwargs.setdefault("quantization_nbins", int(getattr(mrmr, "quantization_nbins", 10) or 10))
    greedy_kwargs = dict(getattr(mrmr, "usability_greedy_kwargs", None) or {})
    seed = int(getattr(mrmr, "random_seed", None) or 0)
    w_lin = float(getattr(mrmr, "usability_w_linear", 0.85))
    w_uni = float(getattr(mrmr, "usability_w_universal", 0.5))

    # ``w`` only affects usability_greedy's pre-rank, never build_usability_candidate_pool (no RNG in
    # there either), so the linear/universal passes can share ONE pool build instead of two -- the
    # dominant cost of this whole pass. _drop_stored_values must run on BOTH result lists only AFTER
    # both greedy calls finish: it mutates candidates' .values in place, and a candidate object can be
    # shared between the two result lists since both greedy calls read the same pool.
    pool = build_usability_candidate_pool(df, y_cont, base_names, **pool_kwargs)
    linear = usability_greedy(pool, y_cont, w=w_lin, seed=seed, **greedy_kwargs)
    universal = usability_greedy(pool, y_cont, w=w_uni, seed=seed, **greedy_kwargs)
    mrmr.support_linear_ = _drop_stored_values(linear)
    mrmr.support_universal_ = _drop_stored_values(universal)


def _drop_stored_values(candidates: list):
    """Clear each selected candidate's full-n ``values`` array before it is attached to the fitted
    estimator. The greedy needed those arrays DURING selection (CV folds), but ``transform_usability``
    replays each feature from its ``recipe`` (or passes a raw column through by NAME) and never reads
    ``values`` -- so keeping them would EMBED THE TRAINING DATA in the pickled model (a privacy leak +
    bloat: ~n*8 bytes per selected candidate, tens of MB at large n). Replace with a 0-length array of
    the same dtype so the field stays a type-stable ndarray. The recipe / name / mi / src / ops survive."""
    if not candidates:
        return candidates
    for c in candidates:
        v = getattr(c, "values", None)
        if isinstance(v, np.ndarray) and v.size:
            c.values = np.empty(0, dtype=v.dtype)
    return candidates


def materialize_usability_features(candidates: list, X: Any):
    """Build the feature DataFrame for a usability list (``support_linear_`` / ``support_universal_``)
    on new data: a raw candidate (``recipe is None``) passes its named column through; an engineered
    candidate replays its ``EngineeredRecipe``. Output columns are in selection order, scrubbed."""
    import pandas as pd
    from .engineered_recipes import apply_recipe

    if not candidates:
        return pd.DataFrame(index=getattr(X, "index", None))

    def _col(cand):
        """Materialize one usability-list candidate column: raw source column or its recorded engineered recipe."""
        if cand.recipe is None:
            src = X[cand.name] if isinstance(X, pd.DataFrame) else None
            if src is None:
                raise ValueError(f"raw usability feature {cand.name!r} needs a DataFrame X to replay")
            _replayed = pd.to_numeric(src, errors="coerce").to_numpy(dtype=np.float64)
            _n_coerced = int((~np.isfinite(_replayed)).sum())
            if _n_coerced:
                logger.warning(
                    "usability: replay of raw column %r produced %d/%d non-finite/unparseable value(s) coerced to 0.0.",
                    cand.name, _n_coerced, _replayed.size,
                )
            return np.nan_to_num(_replayed, nan=0.0, posinf=0.0, neginf=0.0)
        return np.nan_to_num(np.asarray(apply_recipe(cand.recipe, X), dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    data = {cand.name: _col(cand) for cand in candidates}
    return pd.DataFrame(data, index=getattr(X, "index", None))
