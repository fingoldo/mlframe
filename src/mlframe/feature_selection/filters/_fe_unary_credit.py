"""Linear-usability credit of unary transforms of the SELECTED raw columns (a default post-FE pass, the sibling of the pure-form retention of ``_fe_pure_form_retention``).

MRMR selects by mutual information, which cannot credit ``log(x)`` of a lognormal column or ``sin(x1)`` of ``y = sin(x1) + x2``: a function of a selected column has zero conditional MI, and rank-binned MI is the
same for every monotone transform. A linear or additive head still gains a great deal from the right form (research tt_a2 / tt_b2 / tt_b4, 2026-11-24). This pass asks that question with the statistics of
``_unary_credit_stats`` (partialled Pearson gain against the residual of the other selected columns, a max-over-candidates permutation null, family-wise correction over the selected columns, replication on a
held-out half) and appends every credited form as a replayable ``unary_credit`` recipe to ``_engineered_recipes_``, exactly like a retained pure form: ``transform()`` replays it, ``get_feature_names_out`` lists it,
``support_`` and the screening matrix are not touched. Purely additive; a column the null does not credit adds nothing, so a selection with no transformable column is byte-identical.

Interaction credit (regression target only): a column whose effect on the target runs through a PRODUCT with another column (``sin(x1) * log(x2)``, a sum of such products) has no linear effect for the test above to
see. The raw columns that are the two sources of an engineered pair column are tested with ``_unary_credit_interaction`` (a column's form against flexible functions of its partners); the credited pairs become
replayable ``unary_credit_product`` recipes, ``mul(sin(x1),log(x2))``, which the credit parsimony then weighs against the approximations the pair engine built. The detected form of every column, credited alone or
through a product, is recorded in ``unary_forms_`` (``{'x1': 'sin', 'x2': 'log'}``); where the data cannot tell the credited form from others (``asinh`` of a normal column and ``tanh`` at a different scale
predict equally well), ``unary_forms_tied_`` lists the alternatives (``{'x2': ('asinh',)}``): the generating form is among the two.

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
#: The interaction credit tests at most this many columns (the most frequent sources of the engineered pair columns) against each other.
MAX_INTERACTION_COLUMNS = 8
#: An engineered column with at most this many raw sources counts as evidence of an interaction among them.
MAX_PAIR_SOURCES = 4

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
    taken = {str(n) for n in (getattr(mrmr, "_engineered_features_", None) or [])}
    forms_seen = getattr(mrmr, "unary_forms_", None)
    forms_seen = dict(forms_seen) if isinstance(forms_seen, dict) else {}
    single_verdicts: "dict[str, Any]" = {}
    if names:
        block = np.column_stack([np.asarray(X[nm].to_numpy(), dtype=np.float64) for nm in names])
        if binary:
            verdicts = credit_columns_binary(block, (y == levels[1]).astype(np.float64), seed=seed)
        else:
            verdicts = credit_columns(block, np.asarray(y, dtype=np.float64), seed=seed)
        single_verdicts = {nm: v for nm, v in zip(names, verdicts) if v.form is not None}
    single_form = {nm: v.form for nm, v in single_verdicts.items()}
    # the interaction credit judges the sources of the pair columns, selected or not, with the main effect inside its block: where it credits a form it replaces the single-column form of the same column
    interaction: "dict[str, Any]" = {}
    interaction_verdicts: "dict[str, Any]" = {}
    products: "list[tuple[Any, str]]" = []
    if not binary:
        interaction_verdicts, products = _credit_products(mrmr, X, np.asarray(y, dtype=np.float64), names, single_form, taken, seed=seed, verbose=verbose)
        interaction = {nm: v.form for nm, v in interaction_verdicts.items()}
    final = {**single_form, **{nm: f for nm, f in interaction.items() if f is not None}}
    engineered = {str(n) for n in (getattr(mrmr, "_engineered_features_", None) or [])}
    forms_seen.update({nm: f.name for nm, f in final.items() if nm not in engineered})
    tied_seen = getattr(mrmr, "unary_forms_tied_", None)
    tied_seen = dict(tied_seen) if isinstance(tied_seen, dict) else {}
    verdict_of = {**single_verdicts, **interaction_verdicts}
    tied_seen.update({nm: verdict_of[nm].tied for nm in final if nm not in engineered and nm in verdict_of and verdict_of[nm].tied})
    soft = set(getattr(mrmr, "_unary_credit_soft_", None) or ())
    out = []
    for nm in final:
        form = final[nm]
        new_name = f"{form.name}({nm})"
        if new_name in taken:
            continue
        out.append((build_unary_credit_recipe(name=new_name, src_name=nm, form=form), new_name))
        if nm in interaction:
            soft.add(new_name)  # credited only through a product: the parsimony may drop it when the product carries the signal
        if verbose and nm in single_verdicts:
            v = single_verdicts[nm]
            logger.info("MRMR unary credit: %s (linear gain %.3f nats, held-out effect %.4f, p %.3f)", new_name, v.gain, v.effect, v.p_value)
    out.extend(products)
    mrmr.unary_forms_ = forms_seen
    mrmr.unary_forms_tied_ = tied_seen
    mrmr._unary_credit_soft_ = soft
    return out


def _raw_ancestors(source: str, raw: "list[str]") -> "list[str]":
    """The raw columns a source of an engineered recipe stands for: itself when it is raw, else the raw names that occur in it as whole words (``(x2__ratio_log__x3)`` -> ``x2``, ``x3``)."""
    import re

    if source in raw:
        return [source]
    return [nm for nm in raw if re.search(rf"(?<![A-Za-z0-9]){re.escape(nm)}(?![A-Za-z0-9])", source)]


def pair_operand_names(mrmr: Any, X: Any) -> "list[str]":
    """Numeric raw columns of ``X`` that are among the (two to four) raw ancestors of an engineered column of the fit, the most frequent first: the likeliest partners of a product effect."""
    import pandas as pd

    engineered = {str(n) for n in (getattr(mrmr, "_engineered_features_", None) or [])}
    given = getattr(mrmr, "feature_names_in_", None)
    pool = [str(c) for c in (given if given is not None else X.columns) if str(c) in X.columns and str(c) not in engineered]
    raw = sorted((c for c in pool if pd.api.types.is_numeric_dtype(X[c].dtype)), key=lambda c: -len(c))
    counts: "dict[str, int]" = {}
    for recipe in getattr(mrmr, "_engineered_recipes_", None) or []:
        ancestors = dict.fromkeys(a for src in (getattr(recipe, "src_names", ()) or ()) for a in _raw_ancestors(str(src), raw))
        if 2 <= len(ancestors) <= MAX_PAIR_SOURCES:
            for a in ancestors:
                counts[a] = counts.get(a, 0) + 1
    return sorted(counts, key=lambda c: -counts[c])


def _credit_products(
    mrmr: Any, X: Any, y: np.ndarray, selected: "list[str]", single_form: "dict[str, Any]", taken: "set[str]", *, seed: int, verbose: int
) -> "tuple[dict[str, Any], list[tuple[Any, str]]]":
    """``(verdicts of the interaction credit by column, [(recipe, name)] of the products)`` among the sources of the engineered pair columns (see the module docstring)."""
    from ._unary_credit_interaction import credit_interactions, select_products
    from .engineered_recipes import build_unary_credit_product_recipe

    candidates = pair_operand_names(mrmr, X)[:MAX_INTERACTION_COLUMNS]
    if len(candidates) < 2:
        return {}, []
    columns = (candidates + [nm for nm in selected if nm not in candidates])[: MAX_INTERACTION_COLUMNS + 4]
    block = np.column_stack([np.asarray(X[nm].to_numpy(), dtype=np.float64) for nm in columns])
    skip = tuple(i for i, nm in enumerate(columns) if nm not in candidates)
    verdicts, _partner_of = credit_interactions(block, y, start=[single_form.get(nm) for nm in columns], skip=skip, seed=seed)
    found_verdicts = {nm: v for nm, v in zip(columns, verdicts) if v.form is not None}
    found = {nm: v.form for nm, v in found_verdicts.items()}
    form_of = {nm: found.get(nm, single_form.get(nm)) for nm in candidates}
    pairs = [(i, j) for i in range(len(candidates)) for j in range(i + 1, len(candidates)) if form_of[candidates[i]] is not None or form_of[candidates[j]] is not None]
    block_c = np.column_stack([np.asarray(X[nm].to_numpy(), dtype=np.float64) for nm in candidates])
    accepted = select_products(block_c, y, [form_of[nm] for nm in candidates], pairs, seed=seed)
    out = []
    for i, j in accepted:
        a, b = candidates[i], candidates[j]
        form_a, form_b = form_of[a], form_of[b]
        name = f"mul({form_a.name + '(' + a + ')' if form_a else a},{form_b.name + '(' + b + ')' if form_b else b})"
        if name in taken:
            continue
        out.append((build_unary_credit_product_recipe(name=name, src_a=a, form_a=form_a, src_b=b, form_b=form_b), name))
        if verbose:
            logger.info("MRMR interaction credit: %s", name)
    return found_verdicts, out
