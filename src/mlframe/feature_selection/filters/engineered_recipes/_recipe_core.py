"""The frozen ``EngineeredRecipe`` dataclass + its array-aware ``extra`` equality helper.

``EngineeredRecipe`` is the leaf of this package: every other submodule + the
``__init__`` facade import it from here, and it imports nothing from a sibling,
so the package import graph stays acyclic. It round-trips cleanly through pickle
/ ``sklearn.clone`` (no closures / fitted estimators captured).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


def _extra_equal(a: dict, b: dict) -> bool:
    """Array-aware dict equality for ``EngineeredRecipe.extra``. Plain ``dict.__eq__`` returns an ndarray (not bool) when values are arrays; this helper uses
    ``np.array_equal`` for arrays and ``==`` otherwise.

    2026-05-30 Wave 9.1 fix (loop iter 45): three correctness gaps:
      1. ``np.array_equal(va, vb)`` returns False on NaN-containing
         arrays because NaN != NaN. Persisted recipes whose lookups /
         diagnostics contained NaN (factorize/target_encoding lookups,
         cluster_aggregate's ``pca_var_ratio`` when PCA degenerates)
         failed pickle round-trip equality and ``sklearn.clone`` ==
         fitted checks.
      2. Scalar NaN in the else branch (``va != vb`` is True for
         ``nan != nan``) had the same defect.
      3. Nested list-of-arrays raised ``ValueError: truth value
         ambiguous`` from ``va != vb`` instead of returning bool -
         leaking an exception out of ``__eq__``.
    Fix: NaN-aware array equality via ``equal_nan=True``; NaN-aware
    scalar equality; defensive fallback that returns False on
    ambiguous truth-value errors instead of raising.
    """
    if a.keys() != b.keys():
        return False
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
            if not (isinstance(va, np.ndarray) and isinstance(vb, np.ndarray)):
                return False
            # NaN-aware (equal_nan=True) for float arrays; harmless for
            # integer arrays (numpy ignores the kwarg there).
            try:
                if not np.array_equal(va, vb, equal_nan=True):
                    return False
            except TypeError:
                # Older numpy without equal_nan; fall back to manual
                # NaN-aware check.
                if va.shape != vb.shape:
                    return False
                _eq = (va == vb) | ((va != va) & (vb != vb))  # both NaN -> equal
                if not bool(_eq.all()):
                    return False
        else:
            # Scalar NaN: float('nan') != float('nan') is True under
            # standard comparison, so treat both-NaN as equal.
            if isinstance(va, float) and isinstance(vb, float):
                if (va != va) and (vb != vb):
                    continue
            try:
                if va != vb:
                    return False
            except ValueError:
                # Nested list-of-arrays / ambiguous truth value: be
                # conservative and report unequal rather than raising
                # from inside __eq__.
                return False
    return True


@dataclass(frozen=True, eq=False)
class EngineeredRecipe:
    """One frozen description of how to recompute an engineered column. Survives pickle / ``sklearn.clone`` (no closures or fitted estimators captured).

    Parameters
    ----------
    name
        Engineered column name (e.g. ``"mul(log(c1),sin(c2))"``). Used in transform output / ``get_feature_names_out``.
    kind
        Replay strategy: ``"unary_binary"`` (numeric pair FE), ``"factorize"`` (cat-FE k-way ordinal merge), ``"target_encoding"``.
    src_names
        Original feature names this recipe consumes. Length 2 for unary_binary, k for factorize. Must be a subset of ``feature_names_in_``.
    unary_names
        ``"unary_binary"``: the two unary fn names from ``feature_engineering.create_unary_transformations(preset)``. ``"identity"`` means no transform.
    binary_name
        ``"unary_binary"``: the binary fn name from ``feature_engineering.create_binary_transformations(preset)``.
    unary_preset / binary_preset
        Preset names captured at fit time so later registry edits don't silently change replay semantics.
    quantization
        ``None`` for raw numeric output, else ``{"nbins": int, "method": str, "dtype": str}`` matching fit-time discretization.
    factorize_nbins
        ``"factorize"``: per-source nbins captured at fit time (shape for ``merge_vars`` and bound for ``unknown_strategy`` clipping).
    unknown_strategy
        ``"factorize"`` test-time handling for unseen category values: ``"clip"`` caps at highest trained bin (default); ``"sentinel"`` adds a separate bin
        (inflates cardinality); ``"raise"`` errors out.
    """

    name: str
    # T1#3 2026-05-18 #1 Hermite recipe: ``"hermite_pair"`` kind carries
    # ``coef_a``, ``coef_b``, ``basis``, ``bin_func_name``, ``preprocess_a``,
    # ``preprocess_b``, ``degree_a``, ``degree_b`` in ``extra``. The
    # 88-min Optuna best_res is now reproducible at predict-time.
    # Layer 23 2026-05-31: ``orth_univariate`` carries (src_names=(c,),
    # extra={basis, degree}); ``orth_pair_cross`` carries
    # (src_names=(c_i, c_j), extra={basis_i, basis_j, deg_a, deg_b}). Replay
    # is closed-form from the source column(s) alone -- no y reference is
    # captured at fit time, so transform() is leakage-free by construction.
    kind: Literal["unary_binary", "factorize", "hermite_pair", "target_encoding", "cluster_aggregate", "orth_univariate", "orth_pair_cross", "orth_triplet_cross", "orth_quadruplet_cross", "orth_spline", "orth_fourier", "orth_diff_basis", "orth_cluster_basis", "mi_greedy_transform", "kfold_target_encoded", "count_encoded", "frequency_encoded", "cat_num_residual", "missing_indicator", "missingness_count", "missingness_pattern", "pairwise_ratio", "grouped_delta", "lagged_diff", "grouped_agg", "composite_group_agg", "grouped_quantile", "target_aware_group_bin", "cat_pair_cross", "numeric_rounding", "digit_extract", "temporal_expanding", "temporal_rolling", "temporal_lag", "modular", "pairwise_modular", "pairwise_integer_lattice", "row_argmax", "conditional_gate", "group_distance", "rare_category", "conditional_residual", "rankgauss", "hinge_basis", "orth_wavelet", "binned_numeric_agg", "conditional_dispersion", "cat_triple_cross", "conditional_quantile_rank", "ordinal_pattern_te", "random_fourier", "sir_direction", "lof_score", "mahalanobis_density"]
    src_names: tuple[str, ...]
    unary_names: tuple[str, ...] = ()
    binary_name: str = ""
    unary_preset: str = "minimal"
    binary_preset: str = "minimal"
    quantization: dict | None = None
    factorize_nbins: tuple[int, ...] = ()
    unknown_strategy: Literal["clip", "sentinel", "raise"] = "clip"
    # Free-form bucket for future recipe kinds (e.g. polynomial-basis Hermite carries coef_a/coef_b/degree_a/degree_b/bin_func_name).
    extra: dict = field(default_factory=dict)

    def __getstate__(self):
        """Pickle-friendly state: unwrap the ``MappingProxyType`` back
        to a plain dict so pickle can handle it. ``__setstate__``
        re-applies the post_init freeze.
        """
        state = dict(self.__dict__)
        # Convert mappingproxy -> dict for pickle.
        if "extra" in state:
            state["extra"] = dict(state["extra"])
        return state

    def __setstate__(self, state):
        # ``frozen=True`` blocks normal __dict__ writes; use
        # ``object.__setattr__`` for each key, then re-apply the
        # post_init proxy/freeze chain.
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self.__post_init__()

    def __post_init__(self):
        """2026-05-30 Wave 9.1 fix (loop iter 49): freeze ``extra``.

        ``frozen=True`` blocks attribute REBIND (``recipe.extra = {}``
        raises) but NOT in-place mutation of the dict itself. Caller-
        held references and accidental ``recipe.extra['x'] = ...`` /
        ``recipe.extra['cell_means'][:] = ...`` silently corrupted
        every subsequent ``apply_recipe`` replay and could poison any
        cache that stored the recipe as a dict/set key (hash stays the
        same on ``(kind, name)`` while ``__eq__`` flips with content).

        Four failure modes documented in the iter-49 repro:
        H.1 caller pops a required key after construction -> apply_*
            raises KeyError on what looked like a "frozen" recipe.
        H.2 in-place ndarray mutation -> apply_* returns garbage.
        H.3 hash-eq invariant violated for any recipe in a set/dict.
        H.4 cache poisoning when recipe used as dict key.

        Fix: deep-copy the ``extra`` dict at construction (severs the
        caller-held reference), freeze every ndarray inside it, then
        wrap in ``MappingProxyType`` (read-only view). Re-assigning via
        ``object.__setattr__`` because ``frozen=True`` blocks normal
        attribute writes inside ``__post_init__``.
        """
        import copy as _copy_iter49
        import types as _types_iter49
        # Deep-copy so post-construction mutation of caller's source
        # dict can't propagate into the recipe.
        _extra_copy = _copy_iter49.deepcopy(dict(self.extra))
        # Freeze every ndarray value so ``recipe.extra['x'][:] = ...``
        # raises ValueError instead of silently corrupting downstream
        # replays. Skip arrays we don't own (views).
        for _v in _extra_copy.values():
            if isinstance(_v, np.ndarray):
                if _v.flags.owndata and _v.flags.writeable:
                    try:
                        _v.flags.writeable = False
                    except Exception:  # nosec B110 - best-effort path
                        pass
        # Wrap in read-only proxy. ``MappingProxyType`` returns
        # ``TypeError`` on any ``extra['x'] = ...`` style write.
        object.__setattr__(
            self, "extra", _types_iter49.MappingProxyType(_extra_copy),
        )

    def __eq__(self, other: object) -> bool:
        """Custom ``__eq__`` handling ndarray values in ``extra`` (factorize lookup tables). ``frozen=True, eq=False`` disables the auto-generated one."""
        if not isinstance(other, EngineeredRecipe):
            return NotImplemented
        if self.kind != other.kind:
            return False
        if self.name != other.name:
            return False
        if self.src_names != other.src_names:
            return False
        if self.unary_names != other.unary_names:
            return False
        if self.binary_name != other.binary_name:
            return False
        if self.unary_preset != other.unary_preset:
            return False
        if self.binary_preset != other.binary_preset:
            return False
        if self.quantization != other.quantization:
            return False
        if self.factorize_nbins != other.factorize_nbins:
            return False
        if self.unknown_strategy != other.unknown_strategy:
            return False
        return _extra_equal(self.extra, other.extra)

    def with_extra(self, **updates) -> "EngineeredRecipe":
        """Return a NEW frozen recipe identical to this one but with ``extra`` extended by ``updates``.

        ``extra`` is a read-only ``MappingProxyType`` after ``__post_init__`` (so post-build in-place writes raise), so attaching late-bound
        metadata (e.g. the ``cat_code_maps`` table built once the raw frame is in scope) goes through this helper. Uses ``dataclasses.replace`` which
        re-runs ``__post_init__`` (deep-copy + re-freeze), preserving the immutability contract on the returned copy."""
        from dataclasses import replace as _replace
        _merged = dict(self.extra)
        _merged.update(updates)
        return _replace(self, extra=_merged)

    def __hash__(self) -> int:
        # Name-based hash (names are unique per fit), since ``extra: dict`` is mutable and would normally disable __hash__.
        # Wave 73 (2026-05-21) hardening: __eq__ (above) walks the ``extra`` dict
        # content (incl. ndarrays via np.array_equal). Hash key (kind, name) is
        # NARROWER than equality, so two recipes with same (kind, name) but
        # different ``extra`` collide on the same hash bucket but DON'T compare
        # equal. That's a valid hash-eq pair (equal-implies-equal-hash holds),
        # but it WOULD trigger an O(N) bucket scan if recipes were ever stored
        # in a set/dict-key with name collisions on different content.
        # Contract: callers MUST NOT use ``EngineeredRecipe`` instances as
        # dict/set keys when the same ``name`` can carry different ``extra``;
        # use ``recipe.name`` (the string) as the dict key instead. All current
        # callers store recipes as dict VALUES (engineered_recipes[r.name] = r).
        return hash((self.kind, self.name))
