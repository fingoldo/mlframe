"""Layer 54 - FE provenance tracking + human report.

User-facing question this layer answers: "Which engineered columns are in
my ``support_``, why were they selected, and what's each one's MRMR gain
contribution?". Pre-Layer-54 the answer required querying ~13 scattered
``*_features_`` fitted attributes (``hybrid_orth_features_``,
``mi_greedy_features_``, ``kfold_te_features_``, ``count_encoding_features_``,
``frequency_encoding_features_``, ``cat_num_interaction_features_``,
``missingness_indicator_features_``, ``missingness_count_features_``,
``missingness_pattern_features_``, ``pairwise_ratio_features_``,
``pairwise_log_ratio_features_``, ``grouped_delta_features_``,
``lagged_diff_features_``) AND the recipes list AND the predictors log.

DESIGN
------
``fe_provenance_`` is a pandas DataFrame populated at the very end of
``fit()``. It is an AUDIT TRAIL: one row per raw column that survived into
support_ PLUS one row per engineered column the FE pipeline PRODUCED this
fit -- including columns the greedy CMI screen / accuracy gate / cross-stage
dedup dropped before support finalisation. Survivors carry their real greedy
gain + support_rank; screened-out engineered columns carry NaN gain /
support_rank -1 but DO carry their mechanism origin, so a downstream audit /
pickle-replay can recover which mechanism produced every engineered column
even when it lost the greedy competition. (The user-facing survivor rosters
``hybrid_orth_features_`` etc. stay intersected with support_ -- pinned by
layer28 -- so this produced-set completeness lives in fe_provenance_ alone.)
Columns:

- ``feature_name`` : the column name in ``support_`` / engineered output.
- ``origin`` : a categorical label drawn from the
  ``FE_ORIGIN_LABELS`` set below. ``"raw"`` for input columns;
  mechanism-specific labels for engineered ones.
- ``mechanism_details`` : a dict-as-string with the source columns and
  any mech-specific knobs (basis name, degree, etc.). Stringified so the
  whole frame survives pickle / clone trivially.
- ``mrmr_gain`` : the greedy gain at the moment this feature was added to
  ``predictors`` (matches ``mrmr_gains_`` at the same position when the
  selection order is preserved).
- ``support_rank`` : 0-based position in the greedy selection order.

PURE ADDITIVE
-------------
This layer touches NO decision logic; ``fe_provenance_`` is metadata only.
The DataFrame is built from already-stored fitted attrs (``support_``,
``feature_names_in_``, ``_engineered_features_``, ``_engineered_recipes_``,
``mrmr_gains_``, the various ``*_features_`` rosters). When the
ingredients are missing (e.g. someone unpickles a pre-Layer-54 model and
never calls fit again), the helper returns an empty DataFrame instead of
raising so downstream report calls degrade gracefully.

DEFAULT-ON
----------
There is no opt-in flag: every fitted MRMR carries ``fe_provenance_``.
Cost is O(n_features_) string building at end of fit().
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


# Canonical origin labels. Public surface; tests pin the membership.
FE_ORIGIN_LABELS = (
    "raw",
    "hybrid_orth",
    "mi_greedy",
    "kfold_te",
    "count_enc",
    "freq_enc",
    "cat_num_resid",
    "missing_indicator",
    "missing_count",
    "missing_pattern",
    "pairwise_ratio",
    "pairwise_log_ratio",
    "grouped_delta",
    "lagged_diff",
    "dcd_aggregate",
    "cluster_aggregate",
    "unary_binary_pair",
    "hinge_basis",
    "numeric_decompose",
    "periodic",
    "group_distance",
    "temporal_agg",
    "wavelet_basis",
    "cat_cross",
    "grouped_agg",
    "extra_fe",
    "conditional_gate",
    "row_argmax",
    "integer_lattice",
    "engineered_unknown",
)


# Mapping recipe.kind -> origin label. Covers every kind currently emitted
# by engineered_recipes.py (Layer 23+). Anything missing falls back to
# ``"engineered_unknown"`` so the report never silently mislabels.
_RECIPE_KIND_TO_ORIGIN: dict[str, str] = {
    "orth_univariate": "hybrid_orth",
    "orth_pair_cross": "hybrid_orth",
    "orth_spline": "hybrid_orth",
    "orth_fourier": "hybrid_orth",
    "orth_diff_basis": "hybrid_orth",
    "orth_cluster_basis": "hybrid_orth",
    "orth_triplet_cross": "hybrid_orth",
    "orth_quadruplet_cross": "hybrid_orth",
    "hermite_pair": "hybrid_orth",
    "mi_greedy_transform": "mi_greedy",
    "kfold_target_encoded": "kfold_te",
    "count_encoded": "count_enc",
    "frequency_encoded": "freq_enc",
    "cat_num_residual": "cat_num_resid",
    "missing_indicator": "missing_indicator",
    "missingness_count": "missing_count",
    "missingness_pattern": "missing_pattern",
    "pairwise_ratio": "pairwise_ratio",
    "div": "pairwise_ratio",
    "log_div": "pairwise_log_ratio",
    "grouped_delta": "grouped_delta",
    "lagged_diff": "lagged_diff",
    "grouped_agg": "grouped_agg",
    "composite_group_agg": "grouped_agg",
    "grouped_quantile": "grouped_agg",
    "target_aware_group_bin": "grouped_agg",
    "cluster_aggregate": "cluster_aggregate",
    "target_encoding": "kfold_te",
    "cat_pair_cross": "cat_cross",
    "cat_triple_cross": "cat_cross",
    "unary_binary": "unary_binary_pair",
    # Piecewise-linear / change-point hinge legs (a__relu_gt.../relu_lt...).
    "hinge_basis": "hinge_basis",
    # Numeric decomposition family (rounding buckets, digit extraction).
    "numeric_rounding": "numeric_decompose",
    "digit_extract": "numeric_decompose",
    # Periodic / modular-arithmetic family.
    "modular": "periodic",
    # Cross-row distance-to-group family.
    "group_distance": "group_distance",
    # Temporal expanding/rolling/lag aggregates.
    "temporal_expanding": "temporal_agg",
    "temporal_rolling": "temporal_agg",
    "temporal_lag": "temporal_agg",
    # Wavelet basis family.
    "orth_wavelet": "wavelet_basis",
    # Extra FE families (rare-category, conditional residual/dispersion, rankgauss).
    "rare_category": "extra_fe",
    "conditional_residual": "extra_fe",
    "conditional_dispersion": "extra_fe",
    "rankgauss": "extra_fe",
    # Threshold-gate / number-theoretic / binned-aggregate families (default-ON 2026; previously
    # unmapped -> their surviving columns showed as engineered_unknown in fe_provenance_).
    "conditional_gate": "conditional_gate",   # gate_mask__ / gate_select__ regime-switch
    "row_argmax": "row_argmax",
    "pairwise_integer_lattice": "integer_lattice",  # il_gcd / il_lcm / il_and
    "pairwise_modular": "periodic",            # pmod_ residue -- same family as the single-col "modular"
    "binned_numeric_agg": "grouped_agg",       # binagg_ qbin-strata aggregate of a numeric col
    # ``factorize`` stays engineered_unknown by design: the cat k-way
    # materializer emits it as a generic ordinal lookup with no single
    # mechanism semantics worth a dedicated origin bucket.
    "factorize": "engineered_unknown",
}


# Mapping mlframe roster attribute name -> origin label, used as a
# fallback when the recipe is missing (higher-order engineered feature
# with no replayable recipe) or the kind is ``unary_binary`` / generic.
# Roster lookup runs AFTER recipe lookup so per-recipe specificity wins.
#
# Ordering invariant (Layer 64): specific-mechanism rosters (L33 kfold_te,
# L34 count/freq/cat-num, L37 missingness, L38 ratio/grouped/lagged)
# precede the catch-all ``hybrid_orth_features_`` and ``mi_greedy_features_``
# rosters. ``_mrmr_fit_impl`` deliberately maintains
# ``hybrid_orth_features_`` as a cumulative "all engineered cols
# appended" tracker (so the downstream cross-stage dedup pass can walk
# one list), which means it carries names from the specific buckets
# too. If hybrid_orth were checked first the lookup would label every
# count-encoded / missingness / pairwise column as ``hybrid_orth`` and
# fe_provenance_ would lose its per-mechanism breakdown.
_ROSTER_ATTR_TO_ORIGIN: tuple[tuple[str, str], ...] = (
    ("kfold_te_features_", "kfold_te"),
    ("count_encoding_features_", "count_enc"),
    ("frequency_encoding_features_", "freq_enc"),
    ("cat_num_interaction_features_", "cat_num_resid"),
    ("missingness_indicator_features_", "missing_indicator"),
    ("missingness_count_features_", "missing_count"),
    ("missingness_pattern_features_", "missing_pattern"),
    ("pairwise_ratio_features_", "pairwise_ratio"),
    ("pairwise_log_ratio_features_", "pairwise_log_ratio"),
    ("grouped_delta_features_", "grouped_delta"),
    ("lagged_diff_features_", "lagged_diff"),
    # Catch-all rosters last so specific buckets win the lookup.
    ("mi_greedy_features_", "mi_greedy"),
    ("hybrid_orth_features_", "hybrid_orth"),
)


# Provenance DataFrame schema. The ``compute_fe_provenance`` helper
# returns an empty frame with this column order when there is no fitted
# state to inspect; downstream callers can safely ``.iterrows()`` /
# ``.itertuples()`` without a column-existence check.
_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "feature_name",
    "origin",
    "mechanism_details",
    "mrmr_gain",
    "support_rank",
)


def _safe_str(value: Any) -> str:
    """Stable stringification for ``mechanism_details``. Sorts dict keys
    so the report column is deterministic across Python hash seeds (the
    rest of MRMR pins selection-order determinism; provenance must
    follow). Falls back to ``repr`` on unexpected types."""
    try:
        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda kv: str(kv[0]))
            inner = ", ".join(f"{k!r}: {_safe_str(v)}" for k, v in items)
            return "{" + inner + "}"
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(_safe_str(v) for v in value) + "]"
        if isinstance(value, np.ndarray):
            # Don't dump the full array (could be a huge factorize lookup
            # table). Surface shape + dtype for diagnosis without bloating
            # the report column.
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        return repr(value)
    except Exception:
        return "<unstringifiable>"


def _origin_from_recipe(recipe: Any) -> tuple[str, dict]:
    """Pull (origin, details) from an EngineeredRecipe-like object.

    Returns ("engineered_unknown", {}) if ``recipe`` is None or its
    ``kind`` is unrecognised. ``details`` always includes ``src_names``
    when available so the report shows where the engineered feature came
    from regardless of the kind-specific bucket.
    """
    if recipe is None:
        return "engineered_unknown", {}
    kind = str(getattr(recipe, "kind", "") or "")
    origin = _RECIPE_KIND_TO_ORIGIN.get(kind, "engineered_unknown")
    details: dict[str, Any] = {"kind": kind}
    src = getattr(recipe, "src_names", None)
    if src:
        details["src_names"] = tuple(src)
    # Pull a few well-known knobs from ``extra`` without dragging the
    # whole dict (which may include large ndarrays).
    extra = getattr(recipe, "extra", None) or {}
    for key in ("basis", "basis_i", "basis_j", "degree", "deg_a", "deg_b",
                "degree_a", "degree_b", "freq", "knots", "fn_name",
                "preset", "smoothing", "folds", "eps", "agg"):
        if key in extra:
            details[key] = extra[key]
    # Recipe knobs that live on the dataclass itself. Each is a small
    # scalar / tuple by construction; ndarray-valued knobs are never on
    # the dataclass surface (they live in ``extra``), so a plain truthy
    # check is safe here.
    for attr in ("unary_names", "binary_name", "unary_preset",
                 "binary_preset", "factorize_nbins", "unknown_strategy"):
        val = getattr(recipe, attr, None)
        if val is None:
            continue
        # Use len() for collections instead of ``not in (..., (), [])``
        # because the latter compares with ``==`` which triggers
        # element-wise broadcasting on ndarray-shaped values.
        try:
            if isinstance(val, (tuple, list, dict, str)) and len(val) == 0:
                continue
        except TypeError:
            pass
        details[attr] = val
    return origin, details


def _origin_from_rosters(name: str, mrmr_self: Any) -> str:
    """Fallback origin lookup against the mechanism roster attributes."""
    for attr, label in _ROSTER_ATTR_TO_ORIGIN:
        # ``or ()`` is unsafe when the roster is an ndarray (truth test
        # raises on size>1). Use an explicit ``None`` guard.
        roster = getattr(mrmr_self, attr, None)
        if roster is None:
            continue
        try:
            # Force list membership to dodge ndarray ``__contains__``
            # returning a broadcast result on some dtypes.
            if name in list(roster):
                return label
        except Exception:
            continue
    # DCD aggregate fallback: cluster_members_ is keyed by the engineered
    # aggregate name when DCD compose ran.
    cluster_members = getattr(mrmr_self, "cluster_members_", None)
    if isinstance(cluster_members, dict) and name in cluster_members:
        return "dcd_aggregate"
    return "engineered_unknown"


def _greedy_rank_for_name(name: str, predictors: Iterable[Any]) -> int:
    """Find the support_rank of ``name`` in the greedy predictor log.
    Returns -1 when the name is not in the log (e.g. raw column carried
    via the empty-support fallback). Tests treat -1 as "no greedy rank".
    """
    # Compare on the SIMPLIFIED name on BOTH sides: ``name`` arrives already simplified
    # (it is a ``_final_feature_order`` entry, which mirrors get_feature_names_out's
    # value-preserving DISPLAY canonicalisation), but the predictor log stores the RAW
    # op-name, so a raw-vs-simplified compare would miss the rank of any column whose
    # name canonicalises (e.g. a dead ``neg(b)`` under an ``abs``). ``simplify_fe_name``
    # is idempotent, so simplifying an already-simplified ``name`` is a no-op.
    from .engineered_recipes._recipe_name_simplify import simplify_fe_name
    _target = simplify_fe_name(str(name)) if name is not None else name
    for idx, entry in enumerate(predictors or ()):
        try:
            _en = entry.get("name")
            if _en is not None and simplify_fe_name(str(_en)) == _target:
                return idx
        except Exception:
            continue
    return -1


def _final_feature_order(mrmr_self: Any) -> list[str]:
    """Return ``support_`` raw names followed by every engineered name the FE pipeline PRODUCED this fit (in production order). Survivor names come first (mirroring ``transform()`` column order so the
    provenance DataFrame zips positionally against transform output), then any engineered column the greedy CMI screen / accuracy gate / cross-stage dedup dropped before support finalisation.

    The dropped-but-produced names are drained from ``_produced_recipes_`` -- the read-only audit ledger snapshotted in ``_mrmr_fit_impl`` before the screen ran. They are REQUIRED for the audit / pickle-
    replay paths to recover which mechanism produced each engineered column even when it lost the greedy competition; without them a healthy fit on a kitchen-sink frame where the screen keeps only the
    strongest ~5 of ~18 produced columns would mislabel the ledger as "only 3-5 mechanisms fired" when in fact ~12 did. The survivor-only rosters (``hybrid_orth_features_`` etc.) stay intersected with
    support_ (pinned by layer28), so this completeness is carried by fe_provenance_ alone, not by the user-facing rosters."""
    names: list[str] = []
    seen: set[str] = set()

    def _append(nm: Any) -> None:
        if nm is None:
            return
        s = str(nm)
        if s in seen:
            return
        seen.add(s)
        names.append(s)

    support = getattr(mrmr_self, "support_", None)
    feature_names_in = getattr(mrmr_self, "feature_names_in_", None) or []
    if support is not None and len(feature_names_in):
        try:
            for idx in np.asarray(support).tolist():
                if 0 <= int(idx) < len(feature_names_in):
                    _append(feature_names_in[int(idx)])
        except Exception:
            pass
    # NAME-SYNC with get_feature_names_out (2026-06-20). ``get_feature_names_out`` advertises
    # engineered columns through ``simplified_recipe_names`` (value-preserving DISPLAY canonicalisation,
    # e.g. ``abs(div(sqr(a),neg(b)))`` -> ``abs(div(sqr(a),b))``), so the support_/output label of an
    # engineered feature is its SIMPLIFIED name. The provenance table documents ``feature_name`` as
    # "the column name in support_/engineered output", so it MUST use the same simplified form -- else a
    # lookup ``prov[prov.feature_name == out_name]`` misses (observed for the clean additive composite
    # whose ``neg(b)`` inside an ``abs`` canonicalises away in the output but not here). Apply the SAME
    # per-name simplifier; it is idempotent + safe on raw column names (returns them unchanged).
    from .engineered_recipes._recipe_name_simplify import simplify_fe_name
    engineered_recipes = getattr(mrmr_self, "_engineered_recipes_", None) or ()
    for recipe in engineered_recipes:
        _nm = getattr(recipe, "name", None)
        _append(simplify_fe_name(_nm) if _nm is not None else None)
    # Drain every survivor roster the transform path replays so each surviving mechanism shows up even when no recipe object was emitted by the FE step (the L34/L37/L38 stages store their outputs as
    # roster names + replay logic, not as EngineeredRecipe instances). Rosters are post-reconciliation (survivor-only), so this never adds a screened-out name.
    for attr, _label in _ROSTER_ATTR_TO_ORIGIN:
        roster = getattr(mrmr_self, attr, None)
        if roster is None:
            continue
        try:
            for nm in list(roster):
                _append(nm)
        except Exception:
            continue
    # Audit-trail completion: append every engineered column the FE stages PRODUCED but the screen / gate / dedup dropped. These carry no greedy rank (rank -1, NaN gain) but DO carry their origin so the
    # ledger reflects the full set of mechanisms that fired, not just the survivors.
    produced = getattr(mrmr_self, "_produced_recipes_", None) or ()
    for recipe in produced:
        _nm = getattr(recipe, "name", None)
        _append(simplify_fe_name(_nm) if _nm is not None else None)
    return names


def compute_fe_provenance(mrmr_self: Any) -> pd.DataFrame:
    """Build the Layer 54 ``fe_provenance_`` DataFrame from the fitted
    estimator. Pure-read; no estimator state mutation.

    Returns an empty (but correctly-shaped) DataFrame when the estimator
    is unfitted or the fitted attributes have been wiped.
    """
    columns = list(_PROVENANCE_COLUMNS)
    empty = pd.DataFrame({col: pd.Series([], dtype=object) for col in columns})
    if getattr(mrmr_self, "feature_names_in_", None) is None:
        return empty
    if getattr(mrmr_self, "support_", None) is None:
        return empty

    feature_names_in = list(mrmr_self.feature_names_in_)
    engineered_recipes = list(getattr(mrmr_self, "_engineered_recipes_", []) or [])
    # Index BOTH the survivor recipes and the full produced-recipes ledger by name so every engineered column -- survivor or screened-out -- resolves to its true origin via its recipe.kind rather than
    # falling through to "engineered_unknown". Survivor recipes take precedence (same object when both contain a name); the produced ledger only ADDS the dropped names.
    produced_recipes = list(getattr(mrmr_self, "_produced_recipes_", []) or [])
    # Key by the SIMPLIFIED recipe name (the same value-preserving DISPLAY canonicalisation
    # ``get_feature_names_out`` and ``_collect_*_names`` advertise) so the origin/details lookup
    # below resolves on the simplified ``final_names`` -- keying by the RAW recipe name would miss
    # every simplified column (e.g. ``abs(div(sqr(a),neg(b)))`` -> ``abs(div(sqr(a),b))``) and
    # mis-tag it ``engineered_unknown``. ``simplify_fe_name`` is idempotent + safe on raw names.
    from .engineered_recipes._recipe_name_simplify import simplify_fe_name
    recipe_by_name = {
        simplify_fe_name(str(getattr(r, "name", ""))): r
        for r in produced_recipes
        if getattr(r, "name", None) is not None
    }
    recipe_by_name.update({
        simplify_fe_name(str(getattr(r, "name", ""))): r
        for r in engineered_recipes
        if getattr(r, "name", None) is not None
    })
    predictors = getattr(mrmr_self, "_predictors_log_", None) or ()
    # mrmr_gains_ is in greedy selection order (set in _mrmr_fit_impl ~line
    # 2169). Index by position when the name lines up; fall back to NaN.
    # ``or []`` is unsafe here because ``mrmr_gains_`` may be an ndarray,
    # which raises ``ValueError`` on truth-test for size>1; use an explicit
    # ``None`` guard instead.
    _gains_raw = getattr(mrmr_self, "mrmr_gains_", None)
    if _gains_raw is None:
        gains_arr = np.array([], dtype=np.float64)
    else:
        gains_arr = np.asarray(_gains_raw, dtype=np.float64)

    final_names = _final_feature_order(mrmr_self)
    raw_set = set(feature_names_in)

    rows = []
    for name in final_names:
        rank = _greedy_rank_for_name(name, predictors)
        if rank >= 0 and rank < gains_arr.size:
            gain_val = float(gains_arr[rank])
        else:
            gain_val = float("nan")

        if name in raw_set and name not in recipe_by_name:
            origin = "raw"
            details: dict[str, Any] = {}
        else:
            recipe = recipe_by_name.get(name)
            if recipe is not None:
                origin, details = _origin_from_recipe(recipe)
            else:
                # Engineered name without a recipe -> fall back to roster
                # attrs. Higher-order interactions land here.
                origin = _origin_from_rosters(name, mrmr_self)
                details = {}

        rows.append(
            {
                "feature_name": name,
                "origin": origin,
                "mechanism_details": _safe_str(details),
                "mrmr_gain": gain_val,
                "support_rank": int(rank),
            }
        )

    if not rows:
        return empty
    df = pd.DataFrame(rows, columns=columns)
    return df


def populate_fe_provenance(mrmr_self: Any) -> None:
    """Run ``compute_fe_provenance`` and stash the result on the
    estimator. Wrapped here (rather than at the call site in mrmr.py's
    ``fit``) so the parent module stays under its LOC budget and the
    fallback DataFrame schema is centralised in one place.
    """
    import logging as _logging
    try:
        mrmr_self.fe_provenance_ = compute_fe_provenance(mrmr_self)
    except Exception as exc:
        _logging.getLogger("mlframe.feature_selection.filters.mrmr").warning(
            "MRMR.fit: fe_provenance_ population failed (%s); "
            "get_fe_report() will surface the empty-DataFrame message "
            "until next successful fit. Cause: %s",
            type(exc).__name__, exc,
        )
        mrmr_self.fe_provenance_ = pd.DataFrame(
            {col: pd.Series([], dtype=object) for col in _PROVENANCE_COLUMNS},
        )


def get_unlabeled_recipe_kinds(mrmr_self: Any) -> dict[str, int]:
    """PROVENANCE SELF-AUDIT (W2). List every SURVIVING engineered recipe.kind
    whose provenance origin resolved to ``"engineered_unknown"`` -- i.e. a kind
    the FE pipeline emits but ``_RECIPE_KIND_TO_ORIGIN`` does not map to a
    dedicated origin bucket.

    Turns "did we forget to register a kind?" into a measured list: when a new
    FE family ships a recipe.kind nobody added to ``_RECIPE_KIND_TO_ORIGIN``,
    its surviving columns show up here, so provenance can't silently regress.

    Returns a dict ``{kind: count}`` over the SURVIVING engineered columns (those
    in ``support_`` -- ``support_rank >= 0``) whose origin is
    ``"engineered_unknown"``. On a normal fit after commit 205baa86 this is the
    deliberate ``{"factorize": N}`` set (the generic ordinal lookup is kept
    unlabeled by design) and nothing else; any OTHER kind appearing here is an
    unregistered family -- the guardrail firing.

    Pure-read; never raises. Returns ``{}`` when the estimator is unfitted, has
    no engineered survivors, or every surviving kind is labeled. The ``factorize``
    entry is INCLUDED (it genuinely resolves to engineered_unknown); callers that
    want only the GUARDRAIL signal subtract the known-deliberate set
    ``DELIBERATELY_UNLABELED_KINDS``.
    """
    out: dict[str, int] = {}
    prov = getattr(mrmr_self, "fe_provenance_", None)
    if prov is None or not isinstance(prov, pd.DataFrame) or prov.empty:
        return out
    # Index every recipe (survivor + produced ledger) by name so we can recover
    # each engineered column's recipe.kind. Mirrors compute_fe_provenance's join.
    recipe_by_name: dict[str, Any] = {}
    for attr in ("_produced_recipes_", "_engineered_recipes_"):
        for r in (getattr(mrmr_self, attr, None) or []):
            nm = getattr(r, "name", None)
            if nm is not None:
                recipe_by_name[str(nm)] = r
    try:
        unlabeled = prov[
            (prov["origin"] == "engineered_unknown") & (prov["support_rank"] >= 0)
        ]
    except Exception:
        return out
    for name in unlabeled["feature_name"].tolist():
        recipe = recipe_by_name.get(str(name))
        kind = str(getattr(recipe, "kind", "") or "") if recipe is not None else ""
        key = kind if kind else "<no-recipe>"
        out[key] = out.get(key, 0) + 1
    return out


# Kinds that resolve to ``engineered_unknown`` BY DESIGN (not a registration
# gap). ``get_unlabeled_recipe_kinds`` surfaces these too; the guardrail signal
# is whatever appears OUTSIDE this set. Keep in sync with the deliberate
# ``engineered_unknown`` entries in ``_RECIPE_KIND_TO_ORIGIN``.
DELIBERATELY_UNLABELED_KINDS: frozenset = frozenset({"factorize"})


def get_fe_report(mrmr_self: Any) -> str:
    """Render ``fe_provenance_`` as a single human-readable string.

    The string includes a header line summarising counts per origin and
    the full DataFrame in ``to_string`` form. Returns an explanatory
    message when ``fe_provenance_`` is empty / missing rather than
    raising, so notebooks calling this on a freshly-loaded model never
    crash.
    """
    prov = getattr(mrmr_self, "fe_provenance_", None)
    if prov is None or not isinstance(prov, pd.DataFrame) or prov.empty:
        return (
            "MRMR.fe_provenance_ is empty: estimator is unfitted or "
            "the fitted attributes have been wiped. Call fit() first."
        )
    by_origin = prov.groupby("origin", dropna=False).size().sort_values(ascending=False)
    header_parts = [f"{origin}={count}" for origin, count in by_origin.items()]
    header = "MRMR FE provenance: " + ", ".join(header_parts)
    table = prov.to_string(index=False)
    return header + "\n" + table
