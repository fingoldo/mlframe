"""Recipe construction and the with-recipes entry point for the orthogonal triplet FE.

Carved out of ``_orthogonal_triplet_fe`` for file size; the parent re-exports every name defined here.
"""
from __future__ import annotations

import logging
from typing import Optional
from collections.abc import Sequence

import numpy as np


import pandas as pd

from mlframe.utils.log_throttle import log_throttle
from .hermite_fe import basis_route_by_moments
from ._orthogonal_shared import parse_code_deg_with_basis
from ._orthogonal_univariate_fe import (
    _evaluate_basis_column,
)

logger = logging.getLogger(__name__)
# the triplet module's own logger (the records of the recipe code are emitted under its name)
_otf_logger = logging.getLogger("mlframe.feature_selection.filters._orthogonal_triplet_fe")


def _hybrid_orth_mi_triplet_fe(*args, **kwargs):
    """``_orthogonal_triplet_fe.hybrid_orth_mi_triplet_fe``, looked up on each call: that module re-exports this one at its bottom, so a module-level import
    here would close an import cycle, and a lookup per call keeps a test's patch of the triplet scorer visible."""
    from ._orthogonal_triplet_fe import hybrid_orth_mi_triplet_fe

    return hybrid_orth_mi_triplet_fe(*args, **kwargs)


from ._fe_family_timing import fe_timed
from .engineered_recipes import EngineeredRecipe


def _triplet_recipe_for_name(name: str, X: pd.DataFrame, basis: str) -> Optional[EngineeredRecipe]:
    """Build the replay recipe for one appended engineered column (triplet cross or univariate), or None when its name cannot be parsed."""
    from .engineered_recipes import (
        build_orth_univariate_recipe,
        build_orth_triplet_cross_recipe,
    )

    head = name.rsplit("__", 1)[0]
    legs = head.split("*")
    if len(legs) == 3:
        # triplet cross: "{col_i}*{col_j}*{col_k}__{code_a}{deg_a}_{code_b}{deg_b}_{code_c}{deg_c}"
        try:
            suffix = name.rsplit("__", 1)[1]
            parts = suffix.split("_")
        except (ValueError, IndexError):
            log_throttle(
                _otf_logger,
                "triplet_recipe_cannot_parse_suffix",
                logging.WARNING,
                "hybrid_orth_mi_triplet_fe_with_recipes: cannot parse suffix in %r; skipping recipe.",
                name,
            )
            return None
        if len(parts) != 3:
            log_throttle(
                _otf_logger,
                "triplet_recipe_bad_parts_count",
                logging.WARNING,
                "hybrid_orth_mi_triplet_fe_with_recipes: expected 3 deg parts in %r; skipping recipe.",
                name,
            )
            return None
        basis_a, deg_a = parse_code_deg_with_basis(parts[0])
        basis_b, deg_b = parse_code_deg_with_basis(parts[1])
        basis_c, deg_c = parse_code_deg_with_basis(parts[2])
        if basis_a is None or basis_b is None or basis_c is None or deg_a is None or deg_b is None or deg_c is None:
            log_throttle(
                _otf_logger,
                "triplet_recipe_cannot_parse_code_deg",
                logging.WARNING,
                "hybrid_orth_mi_triplet_fe_with_recipes: cannot parse code/deg from %r; skipping recipe.",
                name,
            )
            return None
        # Same auto-routing fixup as Layer 22 pair recipe builder.
        col_i, col_j, col_k = legs
        # Value-construction (recovers the basis preprocess params for recipe replay): always
        # float64, matching the fresh-path's own dtype (see generate_triplet_cross_basis_features).
        x_i = X[col_i].to_numpy(dtype=np.float64)  # Intentional float64: value construction of the triplet product (see above)
        x_j = X[col_j].to_numpy(dtype=np.float64)  # Intentional float64: value construction of the triplet product (see above)
        x_k = X[col_k].to_numpy(dtype=np.float64)  # Intentional float64: value construction of the triplet product (see above)
        if basis == "auto":
            try:
                basis_a = basis_route_by_moments(x_i)
                basis_b = basis_route_by_moments(x_j)
                basis_c = basis_route_by_moments(x_k)
            except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
                _otf_logger.debug("suppressed: %s", e)
                pass
        # REPLAY-FIDELITY FIX: freeze each leg's fit-time basis-preprocess params
        # (z-score mean/std, min-max lo/hi, ...) so ``transform()`` replays the basis axis
        # byte-exactly instead of refitting it from the apply-time rows (slice-replay corruption).
        # Guarded -> on any failure params stay None (legacy refit path), never crashing the emit.
        _pp_a = _pp_b = _pp_c = None
        try:
            _, _pp_a = _evaluate_basis_column(x_i, basis_a, deg_a, return_params=True)
            _, _pp_b = _evaluate_basis_column(x_j, basis_b, deg_b, return_params=True)
            _, _pp_c = _evaluate_basis_column(x_k, basis_c, deg_c, return_params=True)
        except Exception as e:  # nosec B110 - swallow converted to debug-log, non-fatal by design
            _otf_logger.debug("suppressed: %s", e)
            pass
        return build_orth_triplet_cross_recipe(
            name=name,
            src_a_name=col_i,
            src_b_name=col_j,
            src_c_name=col_k,
            basis_i=basis_a,
            basis_j=basis_b,
            basis_k=basis_c,
            deg_a=deg_a,
            deg_b=deg_b,
            deg_c=deg_c,
            preprocess_params_i=_pp_a,
            preprocess_params_j=_pp_b,
            preprocess_params_k=_pp_c,
        )
    elif len(legs) == 1:
        # univariate: "{col}__{code}{degree}"
        src = legs[0]
        suffix = name.rsplit("__", 1)[1] if "__" in name else ""
        chosen_basis, chosen_degree = parse_code_deg_with_basis(suffix)
        if chosen_basis is None or chosen_degree is None:
            log_throttle(
                _otf_logger,
                "triplet_recipe_cannot_parse_basis_degree",
                logging.WARNING,
                "hybrid_orth_mi_triplet_fe_with_recipes: cannot parse basis/degree from %r; skipping recipe.",
                name,
            )
            return None
        # REPLAY-FIDELITY FIX: freeze the fit-time basis-preprocess params (mirrors
        # the orth_univariate BUG2 fix); without them replay refits the axis from apply-time rows.
        _pp_u = None
        try:
            # Intentional float64: value construction of the triplet product (see above)
            _x_u = X[src].to_numpy(dtype=np.float64)  # value-construction: always float64, see above
            _, _pp_u = _evaluate_basis_column(_x_u, chosen_basis, chosen_degree, return_params=True)
        except Exception as e:  # nosec B110 - optional dependency import guard
            _otf_logger.debug("Could not freeze fit-time basis-preprocess params for %r (%s: %s); recipe replays without them", src, type(e).__name__, e)
        return build_orth_univariate_recipe(
            name=name,
            src_name=src,
            basis=chosen_basis,
            degree=chosen_degree,
            preprocess_params=_pp_u,
        )
    else:
        log_throttle(
            _otf_logger,
            "triplet_recipe_unexpected_leg_count",
            logging.WARNING,
            "hybrid_orth_mi_triplet_fe_with_recipes: unexpected leg count in %r (legs=%r); skipping recipe.",
            name,
            legs,
        )
        return None


@fe_timed("triplet")
def hybrid_orth_mi_triplet_fe_with_recipes(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cols: Optional[Sequence[str]] = None,
    degrees: Sequence[int] = (2, 3),
    triplet_max_degree: int = 1,
    basis: str = "auto",
    top_k: int = 5,
    top_triplet_count: int = 2,
    top_triplet_seed_k: int = 4,
    min_uplift: float = 1.05,
    min_abs_mi_frac: float = 0.1,
    triplet_min_uplift: float = 1.05,
    triplet_min_abs_mi_frac: float = 0.1,
    nbins: int = 10,
    explicit_triplets: Optional[Sequence[tuple]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[EngineeredRecipe]]:
    """Same as :func:`_otf.hybrid_orth_mi_triplet_fe` but additionally returns
    a flat list of recipes (univariate + triplet, in append order) so
    that ``MRMR.transform`` can replay each engineered column from X
    alone (no y). ``explicit_triplets`` (GBM seeder) forces the
    triplet stage to enumerate EXACTLY the given column-name triples (the
    order-3-floored proposer survivors) instead of the C(seed_k, 3) over the
    univariate-MI seed pool.
    """
    X_aug, uni_scores, triplet_scores = _hybrid_orth_mi_triplet_fe(
        X,
        y,
        cols=cols,
        degrees=degrees,
        basis=basis,
        triplet_max_degree=triplet_max_degree,
        top_k=top_k,
        top_triplet_count=top_triplet_count,
        top_triplet_seed_k=top_triplet_seed_k,
        min_uplift=min_uplift,
        min_abs_mi_frac=min_abs_mi_frac,
        triplet_min_uplift=triplet_min_uplift,
        triplet_min_abs_mi_frac=triplet_min_abs_mi_frac,
        nbins=nbins,
        explicit_triplets=explicit_triplets,
    )
    appended = [c for c in X_aug.columns if c not in X.columns]

    recipes = []
    for name in appended:
        recipe = _triplet_recipe_for_name(name, X, basis)
        if recipe is not None:
            recipes.append(recipe)
    return X_aug, uni_scores, triplet_scores, recipes
