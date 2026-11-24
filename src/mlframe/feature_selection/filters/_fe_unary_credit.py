"""Linear-usability credit of unary transforms of the SELECTED raw columns (a default post-FE pass, the sibling of the pure-form retention of ``_fe_pure_form_retention``).

MRMR selects by mutual information, which cannot credit ``log(x)`` of a lognormal column or ``sin(x1)`` of ``y = sin(x1) + x2``: a function of a selected column has zero conditional MI, and rank-binned MI is the
same for every monotone transform. A linear or additive head still gains a great deal from the right form (research tt_a2 / tt_b2 / tt_b4, 2026-11-24). This pass asks that question with the statistics of
``_unary_credit_stats`` (partialled Pearson gain against the residual of the other selected columns, a max-over-candidates permutation null, family-wise correction over the selected columns, replication on a
held-out half) and appends every credited form as a replayable ``unary_credit`` recipe to ``_engineered_recipes_``, exactly like a retained pure form: ``transform()`` replays it, ``get_feature_names_out`` lists it,
``support_`` and the screening matrix are not touched. Purely additive; a column the null does not credit adds nothing, so a selection with no transformable column is byte-identical.

Scope: a regression target (continuous ``y``) and linear / additive heads. A tree head gains nothing from a transform of a feature (measured: at most 0.008 R^2 / 0.0014 nats), so it is neither helped nor
hurt. Rollback of the pass: ``MLFRAME_FE_UNARY_CREDIT=0``."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mlframe.utils.env_flags import FALSE_VALUES, env_reader

logger = logging.getLogger(__name__)

#: The credit runs on at most this many selected numeric columns (the strongest by selection order); the family-wise correction counts exactly these.
MAX_COLUMNS = 30

_enabled_reader = env_reader("MLFRAME_FE_UNARY_CREDIT")


def unary_credit_enabled() -> bool:
    """On unless ``MLFRAME_FE_UNARY_CREDIT`` is set to a false spelling."""
    value = _enabled_reader()
    return value is None or value.strip().lower() not in FALSE_VALUES - {""}


def credit_unary_forms(
    mrmr: Any, X: Any, y_continuous: "np.ndarray | None", selected_names: "list[str]", *, seed: int = 0, verbose: int = 0
) -> "list[tuple[Any, str]]":
    """``[(recipe, name)]`` of the unary forms credited for the numeric columns among ``selected_names`` (empty when nothing is credited).

    ``X`` is the original frame (a DataFrame), ``y_continuous`` the target aligned with its rows: a continuous one takes the Pearson-information criterion (``_unary_credit_stats``), a two-valued one the
    logit-deviance criterion (``_unary_credit_logit``), anything else (multiclass, constant) is left alone. Names already used by an engineered column of the fit are skipped.
    """
    import pandas as pd

    from ._unary_credit_logit import credit_columns_binary
    from ._unary_credit_stats import credit_columns
    from .engineered_recipes import build_unary_credit_recipe

    if not unary_credit_enabled() or y_continuous is None or not isinstance(X, pd.DataFrame):
        return []
    y = np.asarray(y_continuous)
    if y.ndim != 1 or y.shape[0] != len(X) or not np.issubdtype(y.dtype, np.number):
        return []
    levels = np.unique(y[~np.isnan(y.astype(np.float64))]) if y.size else y
    binary = levels.size == 2
    if not binary and np.unique(y[:5000]).size < 20:
        return []
    names = [nm for nm in dict.fromkeys(selected_names) if nm in X.columns and pd.api.types.is_numeric_dtype(X[nm].dtype)][:MAX_COLUMNS]
    if not names:
        return []
    taken = {str(n) for n in (getattr(mrmr, "_engineered_features_", None) or [])}
    block = np.column_stack([np.asarray(X[nm].to_numpy(), dtype=np.float64) for nm in names])
    if binary:
        verdicts = credit_columns_binary(block, (y == levels[1]).astype(np.float64), seed=seed)
    else:
        verdicts = credit_columns(block, np.asarray(y, dtype=np.float64), seed=seed)
    out = []
    for nm, verdict in zip(names, verdicts):
        if verdict.form is None:
            continue
        new_name = f"{verdict.form.name}({nm})"
        if new_name in taken:
            continue
        out.append((build_unary_credit_recipe(name=new_name, src_name=nm, form=verdict.form), new_name))
        if verbose:
            logger.info("MRMR unary credit: %s (linear gain %.3f nats, held-out effect %.4f, p %.3f)", new_name, verdict.gain, verdict.effect, verdict.p_value)
    return out
