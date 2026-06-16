"""Single source of truth for mlframe feature-selector contract testing.

Every shared cross-selector contract test imports ``SELECTOR_SPECS`` from here so
that "which selectors are contract-covered" is decided in ONE place. Adding a
selector to the production registry without a spec here trips
``test_every_registered_selector_has_contract_factory`` (see
test_selector_registry.py) -- registration then implies contract coverage instead
of relying on an author remembering to append to a hand-rolled factory list in two
separate suites.

Each spec carries CAPABILITY FLAGS instead of the old per-test ``try/except ->
pytest.skip`` hatches: a contract a selector genuinely supports runs as a hard
assertion (a regression FAILS, never silently degrades to a skip); a contract it
legitimately does not support is an explicit ``xfail``/branch keyed off the flag,
so the asymmetry stays visible in the report rather than hiding.

``make(task)`` returns a FRESH unfitted selector. The heavy real-selector members
(BorutaShap shadow trials, ShapProxiedFS / HybridSelector internal model fits) are
~10-12 s each, so their specs set ``slow=True`` and ``needs_shap=True``; the
collection hook in conftest.py skips slow specs under ``MLFRAME_FAST=1``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge


# ---------------------------------------------------------------------------
# Robust selected-mask extraction. Selectors expose their selection in three
# different shapes; normalise all of them to a bool mask of length
# n_features_in_ so a single assertion layer works across the family:
#   - RFECV / ShapProxiedFS / BorutaShap: ``support_`` (bool mask OR int indices)
#   - MRMR:                                ``support_`` int indices
#   - HybridSelector:                      no ``support_``; ``get_support()`` only
# Falls back to deriving the mask from get_feature_names_out vs feature_names_in_
# for selectors that expose neither (none today, but keeps the helper total).


def selected_mask(selector) -> np.ndarray:
    """Bool mask aligned with ``feature_names_in_`` (length ``n_features_in_``)."""
    n = int(getattr(selector, "n_features_in_", -1))
    if n <= 0:
        raise ValueError(f"{type(selector).__name__}: n_features_in_ unavailable")

    get_support = getattr(selector, "get_support", None)
    if callable(get_support):
        try:
            s = np.asarray(get_support())
        except Exception:
            s = None
        if s is not None:
            return _to_bool_mask(s, n)

    if hasattr(selector, "support_"):
        return _to_bool_mask(np.asarray(selector.support_), n)

    gfno = getattr(selector, "get_feature_names_out", None)
    names_in = list(getattr(selector, "feature_names_in_", []))
    if callable(gfno) and names_in:
        out = list(gfno())
        keep = {nm for nm in out if nm in names_in}  # drop engineered tail
        return np.array([nm in keep for nm in names_in], dtype=bool)

    raise AttributeError(f"{type(selector).__name__}: no support_/get_support/gfno to derive a mask")


def _to_bool_mask(s: np.ndarray, n: int) -> np.ndarray:
    if s.dtype == bool or s.dtype == np.bool_:
        return s.astype(bool)
    mask = np.zeros(n, dtype=bool)
    if s.size > 0:
        mask[s.astype(int)] = True
    return mask


def selected_names(selector) -> list[str]:
    """The selected RAW feature names (engineered tail excluded)."""
    names_in = list(getattr(selector, "feature_names_in_", []))
    if not names_in:
        return [f"x{i}" for i, keep in enumerate(selected_mask(selector)) if keep]
    return [nm for nm, keep in zip(names_in, selected_mask(selector)) if keep]


# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectorSpec:
    """Declarative capability profile + factory for one selector under contract test.

    Capability flags (default to the common case; flip per selector):
      accepts_ndarray     -- fit/transform accept a bare np.ndarray
      pipeline_compatible -- usable as a step in sklearn.pipeline.Pipeline
      has_gfno            -- exposes get_feature_names_out()
      has_get_support     -- exposes get_support()
      pickle_safe         -- a fitted instance round-trips through pickle with equal transform
      supports_sample_weight -- fit() accepts sample_weight=
      rejects_nan_in_y    -- fit raises on NaN in y
      rejects_single_class_y -- fit raises on constant y (classification-only selectors)
      rejects_duplicate_names -- fit raises on duplicate column NAMES (else silent positional pick)
      nan_in_X_policy     -- "tolerates" | "raises" | "unknown" (skip the hard assert)
      determinism         -- 1.0 => same-seed refit is exactly equal; 0<f<1 => Jaccard floor
      column_order_invariant -- fitting on column-reversed X selects the same name set
      validates_transform_width -- transform raises on a wrong-width ndarray
    """
    name: str
    make: Callable[[str], Any]
    tasks: tuple[str, ...] = ("binary",)
    accepts_ndarray: bool = True
    pipeline_compatible: bool = True
    has_gfno: bool = True
    has_get_support: bool = True
    supports_sample_weight: bool = True
    pickle_safe: bool = True
    rejects_nan_in_y: bool = True
    rejects_single_class_y: bool = True
    rejects_duplicate_names: bool = False
    nan_in_X_policy: str = "unknown"
    determinism: float = 1.0
    column_order_invariant: bool = True
    validates_transform_width: bool = True
    slow: bool = False
    needs_shap: bool = False
    marks: tuple = field(default_factory=tuple)


# --- factories -------------------------------------------------------------
# Tiny, fast configs; identical-shaped so the contract layer stays uniform.


def _make_mrmr(task: str = "binary"):
    from mlframe.feature_selection.filters.mrmr import MRMR
    return MRMR(
        min_relevance_gain=0.0, cv=3, run_additional_rfecv_minutes=False,
        full_npermutations=3, random_seed=0, min_features_fallback=1, verbose=False,
    )


def _make_rfecv(task: str = "binary"):
    from mlframe.feature_selection.wrappers import RFECV
    est = Ridge() if task == "regression" else LogisticRegression(max_iter=200, random_state=0)
    return RFECV(estimator=est, cv=3, max_refits=3, random_state=0,
                 leakage_corr_threshold=None, n_features_selection_rule="argmax")


def _make_shap_proxied(task: str = "binary"):
    from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cls = task != "regression"
    model = (RandomForestClassifier(n_estimators=10, random_state=0) if cls
             else RandomForestRegressor(n_estimators=10, random_state=0))
    return ShapProxiedFS(
        model=model, classification=cls, n_splits=3, n_models=1, max_features=None,
        top_n=10, holdout_size=0.25, revalidate=False, trust_guard=False,
        prefilter_top=None, cluster_features=False, random_state=0, n_jobs=1,
    )


def _make_boruta_shap(task: str = "binary"):
    from mlframe.feature_selection.boruta_shap import BorutaShap
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    cls = task != "regression"
    model = (RandomForestClassifier(n_estimators=20, random_state=0) if cls
             else RandomForestRegressor(n_estimators=20, random_state=0))
    return BorutaShap(model=model, classification=cls, n_trials=10,
                      random_state=0, train_or_test="train", verbose=False)


def _make_hybrid(task: str = "binary"):
    # use_fe / use_tree_member off keeps it deterministic + as fast as the heavy
    # members allow; the MRMR + ShapProxied members still run (that is the point
    # of exercising the real composition through the contract).
    from mlframe.feature_selection.hybrid_selector import HybridSelector
    return HybridSelector(use_fe=False, use_tree_member=False, random_state=0)


def _make_group_aware_rfecv(task: str = "binary"):
    # The PRODUCTION-DEFAULT wrap the training suite instantiates via the registry
    # (cluster_reduce=True). Built through the registry so the contract exercises
    # the exact default-ON code path users get, not a hand-rolled GroupAwareMRMR.
    from mlframe.feature_selection import registry
    return registry.get("RFECV").instantiate(
        estimator=LogisticRegression(max_iter=200, random_state=0),
        cv=3, max_refits=3, random_state=0, leakage_corr_threshold=None,
        n_features_selection_rule="argmax",
    )


# --- the registry ----------------------------------------------------------

SELECTOR_SPECS: dict[str, SelectorSpec] = {
    "MRMR": SelectorSpec(
        name="MRMR", make=_make_mrmr, tasks=("binary", "regression"),
        nan_in_X_policy="tolerates", determinism=1.0, rejects_duplicate_names=True,
    ),
    "RFECV": SelectorSpec(
        name="RFECV", make=_make_rfecv, tasks=("binary", "regression"),
        nan_in_X_policy="unknown", determinism=1.0, rejects_duplicate_names=True,
    ),
    "ShapProxiedFS": SelectorSpec(
        name="ShapProxiedFS", make=_make_shap_proxied, tasks=("binary", "regression"),
        supports_sample_weight=False, determinism=0.6,
        column_order_invariant=False,       # bootstrapped CV + SHAP-coalition ordering
        validates_transform_width=False,    # no transform-time width guard (backlog)
        nan_in_X_policy="unknown", slow=True, needs_shap=True,
    ),
    "BorutaShap": SelectorSpec(
        name="BorutaShap", make=_make_boruta_shap, tasks=("binary", "regression"),
        has_gfno=False, has_get_support=False, supports_sample_weight=False,
        rejects_single_class_y=False, nan_in_X_policy="unknown",
        determinism=1.0, column_order_invariant=False,  # shadow-feature ordering
        slow=True, needs_shap=True,
    ),
    "HybridSelector": SelectorSpec(
        name="HybridSelector", make=_make_hybrid, tasks=("binary",),
        supports_sample_weight=False, rejects_single_class_y=False,
        nan_in_X_policy="unknown", determinism=1.0,
        column_order_invariant=False,       # composed members' ordering
        validates_transform_width=False,    # no transform-time width guard (backlog)
        slow=True, needs_shap=True,
    ),
    "GroupAware(RFECV)": SelectorSpec(
        name="GroupAware(RFECV)", make=_make_group_aware_rfecv, tasks=("binary",),
        supports_sample_weight=True,        # forwards sample_weight to the inner RFECV
        column_order_invariant=False,       # corr-cluster medoid pick is order-sensitive
        validates_transform_width=False,    # wrapper does not re-validate ndarray width
        nan_in_X_policy="unknown", determinism=1.0, slow=True, rejects_duplicate_names=True,
    ),
}


def spec_params(include_slow: bool = True):
    """pytest.param list over SELECTOR_SPECS, attaching markers (slow/needs_shap)."""
    out = []
    for key, spec in SELECTOR_SPECS.items():
        if spec.slow and not include_slow:
            continue
        marks = list(spec.marks)
        if spec.slow:
            marks.append(pytest.mark.slow)
        out.append(pytest.param(spec, id=key, marks=marks))
    return out
