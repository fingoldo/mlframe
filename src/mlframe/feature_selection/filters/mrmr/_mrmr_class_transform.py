"""sklearn transform-protocol surface (get_feature_names_out / get_support / usability union) for MRMR.

Pure move from ``_mrmr_class`` into a mixin. ``transform`` itself stays on the ``MRMR`` class body (so
``_SetOutputMixin.__init_subclass__`` still wraps it for ``set_output``); the helpers it and the sklearn
protocol call live here and resolve through the MRO on the concrete ``MRMR`` instance.
"""

from __future__ import annotations

import numpy as np


class _MRMRTransformMixin:
    """sklearn transform-protocol helpers for :class:`MRMR` (see module docstring)."""

    # Set by the concrete MRMR class (fit()) / sklearn base classes in the MRO; declared here so this
    # mixin type-checks on its own without requiring the full MRMR class body.
    n_features_in_: int
    support_: np.ndarray

    def transform(self, X):
        """sklearn transformer protocol placeholder; the concrete MRMR class overrides this on the class body itself (see module docstring), so calling it on the mixin directly is a programming error."""
        raise NotImplementedError  # overridden on the concrete MRMR class body (see module docstring)

    def get_feature_names_out(self, input_features=None):
        """sklearn-1.x transformer protocol. Returns the names of selected features as an ndarray of str,
        matching transform() output cols. When ``self._engineered_recipes_`` is non-empty, their names are
        appended AFTER the base-feature names; order matches transform() output column order.

        Per the sklearn protocol (BaseEstimator._check_feature_names):
        - When ``input_features`` is None: use ``feature_names_in_``.
        - When ``input_features`` is provided AND fit-time saw real names
          (DataFrame input): the two MUST match or a ``ValueError`` is raised.
          This is the Pipeline column-drift detection contract.
        - When ``input_features`` is provided AND fit-time was an ndarray:
          synthesized ``feature_N`` placeholders are opaque; honour the
          caller's names. This lets Pipelines that name columns downstream
          (e.g. ColumnTransformer + array math + name re-injection) propagate.

        Pre-fix the ``input_features``
        argument was accepted but silently ignored on every code path, so:
        (a) Pipeline column-drift detection was bypassed - mismatched
        ``input_features`` produced fit-time names with no warning;
        (b) After ndarray fit, user-supplied ``input_features`` were dropped
        and synthesized ``feature_N`` placeholders propagated to downstream
        consumers.
        """
        if not hasattr(self, "support_") or not hasattr(self, "feature_names_in_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError("This MRMR instance is not fitted yet. Call 'fit' before " "using 'get_feature_names_out'.")
        # Resolve effective fit-time feature names. If ``input_features`` was
        # provided, validate against the saved ``feature_names_in_`` (sklearn
        # column-drift protocol) - but only when fit saw real names. The
        # ndarray-fit path synthesises ``feature_N`` placeholders which the
        # caller can override.
        if input_features is not None:
            in_names = np.asarray(input_features, dtype=object)
            saved = np.asarray(self.feature_names_in_, dtype=object)
            # 1 fix (loop iter 27): use the
            # ``_feature_names_in_synthesized_`` sentinel set at fit
            # time instead of the brittle ``startswith("feature_")``
            # heuristic. The heuristic misclassified legitimate
            # DataFrame columns the user happened to name
            # ``feature_<n>`` (very common pattern after
            # ``pd.DataFrame(arr)`` + rename) and silently bypassed
            # the sklearn column-drift contract -
            # ``get_feature_names_out(['totally_wrong_A','B','C'])``
            # returned ``['totally_wrong_A']`` instead of raising.
            # Back-compat fallback for unpickled estimators without the
            # sentinel: require an EXACT regex match (anchored, ``f\d+$``
            # or the older ``feature_\d+$`` placeholder convention) AND
            # count parity, not just ``startswith``.
            synthesized = getattr(self, "_feature_names_in_synthesized_", None)
            if synthesized is None:
                import re as _re
                _placeholder = _re.compile(r"^(?:f|feature_)\d+$")
                synthesized = all(_placeholder.match(str(n)) is not None for n in saved)
            if not synthesized:
                if len(in_names) != len(saved) or not np.array_equal(in_names, saved):
                    raise ValueError(f"input_features is not equal to feature_names_in_. " f"Got {list(in_names)[:8]}, expected " f"{list(saved)[:8]}.")
                fni = saved
            else:
                # ndarray-fit case: caller's names take precedence.
                if len(in_names) != len(saved):
                    raise ValueError(f"input_features length ({len(in_names)}) does not " f"match the number of features seen at fit " f"({len(saved)}).")
                fni = in_names
        else:
            fni = np.asarray(self.feature_names_in_, dtype=object)
        support = self.support_
        # Mirror _append_engineered's legacy-recipe filter (transform drops pre-D3 pickled k-way recipes
        # lacking the chained-lookup payload). Without the SAME filter here, get_feature_names_out would
        # advertise MORE columns than transform() emits on a legacy pickle -> a width mismatch that breaks
        # sklearn Pipeline / ColumnTransformer / set_output. For freshly-fit estimators every recipe has
        # the payload, so this is a strict no-op (the list comprehension keeps all recipes).
        # ``_engineered_recipes_`` is set once per fit and never mutated afterwards, so the filtered
        # recipe list + simplified names are cached keyed by that list's object identity: a refit always
        # creates a NEW ``_engineered_recipes_`` object, so the cache self-invalidates on refit with no
        # explicit clear() needed anywhere. Repeated ``get_feature_names_out()`` calls on the same fitted
        # instance (e.g. from a hot inference-wrapper loop) then skip the recipe-filter comprehension and
        # the name-simplification pass entirely after the first call.
        _current_recipes = getattr(self, "_engineered_recipes_", [])
        _cache = getattr(self, "_engineered_names_cache_", None)
        if _cache is not None and _cache[0] is _current_recipes:
            engineered_names = _cache[1]
        else:
            from ..engineered_recipes._recipe_name_simplify import simplified_recipe_names
            _adv_recipes = [r for r in _current_recipes if r.extra.get("chain_lookups") is not None or not r.extra.get("requires_refit_for_replay")]
            # Value-preserving DISPLAY canonicalisation (e.g. abs(div(sqr(a),neg(b))) -> abs(div(sqr(a),b)));
            # transform() names its engineered columns through the SAME helper so widths/names stay in sync.
            engineered_names = simplified_recipe_names(_adv_recipes)
            self._engineered_names_cache_ = (_current_recipes, engineered_names)
        if len(support) == 0 and not engineered_names:
            return np.array([], dtype=object)
        if len(support) > 0 and isinstance(support[0], (bool, np.bool_)):
            base_names = [n for n, s in zip(fni, support) if s]
        else:
            base_names = [fni[i] for i in support]
        names = list(base_names) + engineered_names
        # USABILITY UNION (2026-06-13): when the usability-aware pass ran, transform() ALSO materialises
        # the linear + universal lists' features (deduped against the pure-MI output), so the advertised
        # names must include them or the sklearn-Pipeline width check would reject the wider transform.
        if getattr(self, "usability_aware_lists", False):
            names += [cand.name for cand in self._usability_union_extra(names)]
        return np.asarray(names, dtype=object)

    @property
    def discovered_structure_(self):
        """Read-only EDA view of the discrete STRUCTURAL relationships the four FE detectors found during ``fit`` (modular / lattice /
        argmax / conditional-gate). Assembled near-free from the frozen ``_engineered_recipes_`` metadata (op / modulus / tau / src_names)
        the operators already emitted -- no re-scan, no y. Returns a :class:`~mlframe.feature_selection.structure_discovery.StructureReport`;
        MI / lift are ``nan`` here (the fit did not freeze the scan's MI), the kind + columns + parameter are exact. For MI / lift, call the
        standalone ``discover_structure(X, y)`` instead."""
        from ...structure_discovery import structure_report_from_recipes

        recipes = getattr(self, "_engineered_recipes_", []) or []
        if isinstance(recipes, dict):
            recipes = list(recipes.values())
        n_cols = int(getattr(self, "n_features_in_", 0) or 0)
        return structure_report_from_recipes(recipes, n_columns=n_cols)

    def _usability_union_extra(self, base_names):
        """Ordered ``UsableCandidate`` list from ``support_linear_`` + ``support_universal_`` whose name
        is NOT already present in ``base_names`` (the pure-MI transform output), deduped across the two
        usability lists. The SINGLE SOURCE OF TRUTH for the union appended by both ``get_feature_names_out``
        and ``transform`` so their widths always agree."""
        seen = set(map(str, base_names))
        extra = []
        for attr in ("support_linear_", "support_universal_"):
            for cand in getattr(self, attr, None) or []:
                if cand.name not in seen:
                    seen.add(cand.name)
                    extra.append(cand)
        return extra

    # 1 fix (loop iter 43): explicit
    # ``__sklearn_is_fitted__`` and ``get_support`` so sklearn's
    # ``check_is_fitted`` / ``SelectorMixin`` consumers behave
    # correctly.
    # Pre-fix the class declared only ``BaseEstimator, TransformerMixin``
    # with no ``__sklearn_is_fitted__``, so ``check_is_fitted`` fell
    # back to a heuristic scanning for ANY trailing-underscore attr.
    # ``_mrmr_fit_impl`` sets ``feature_names_in_`` / ``n_features_in_``
    # ~700 lines BEFORE ``support_`` (line 942), so a fit() that
    # crashed mid-screen left a half-fit instance that
    # ``check_is_fitted`` accepted but ``transform`` then refused with
    # ``NotFittedError`` - confusing for any downstream gate that used
    # the canonical check.
    # Also added ``get_support`` to honour the SelectorMixin contract
    # downstream consumers (sklearn Pipeline, RFECV, monitoring hooks)
    # expect.
    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "support_") and hasattr(self, "feature_names_in_")

    def get_support(self, indices: bool = False):
        """sklearn ``SelectorMixin`` protocol: return the selected-feature boolean mask (or, when ``indices=True``, the integer positions), built from ``support_`` against ``n_features_in_``."""
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self)
        mask = np.zeros(int(self.n_features_in_), dtype=bool)
        _supp = np.asarray(self.support_, dtype=np.intp)
        if _supp.size:
            mask[_supp] = True
        return np.where(mask)[0] if indices else mask

    def _append_usability_union(self, base_out, X):
        """Append the usability lists' features (``support_linear_`` + ``support_universal_``, deduped
        against the pure-MI output and each other) to the standard transform output, and record
        ``usability_feature_groups_`` -- a ``{'nonlinear'|'linear'|'universal': [names]}`` map so a
        downstream can subset to a model family's list. The pure-MI columns keep precedence on a name
        clash; the union is what lets a LINEAR model trained on the suite's shared matrix pick up the
        engineered interaction (c*d) it needs without any per-model re-transform."""
        import pandas as pd
        from .._usability_lists import materialize_usability_features

        if not isinstance(base_out, pd.DataFrame):
            cols = list(self.get_feature_names_out())
            arr = np.asarray(base_out)
            # get_feature_names_out already includes the union names; the base ndarray is narrower
            # (pure-MI only), so name only its own width here and let the concat below add the rest.
            base_out = pd.DataFrame(arr, columns=cols[: arr.shape[1]], index=getattr(X, "index", None))

        nonlinear_names = list(base_out.columns)
        groups = {
            "nonlinear": list(nonlinear_names),
            "linear": [c.name for c in (getattr(self, "support_linear_", None) or [])],
            "universal": [c.name for c in (getattr(self, "support_universal_", None) or [])],
        }
        extra = self._usability_union_extra(nonlinear_names)
        self.usability_feature_groups_ = groups
        if not extra:
            return base_out
        mat = materialize_usability_features(extra, X)
        mat.index = base_out.index
        return pd.concat([base_out, mat], axis=1)

    def transform_usability(self, X, which: str = "linear"):
        """Materialise a USABILITY-AWARE feature space on ``X`` -- the linear-downstream selection
        produced when the estimator was fit with ``usability_aware_lists=True``.

        ``which='linear'`` -> ``support_linear_`` (the ``w->1`` usability list, for linear / additive
        models); ``which='universal'`` -> ``support_universal_`` (the blended list); ``which=
        'nonlinear'`` returns the standard pure-MI ``transform`` output (the tree list). Each entry
        is replayed from its stored ``EngineeredRecipe`` (or passed through as a raw column), so the
        returned DataFrame is the exact feature space the usability greedy scored at fit time.

        Raises if the requested list was not computed (estimator fit with the pass OFF, or a
        non-numeric target / degenerate pool left it ``None``)."""
        if which == "nonlinear":
            return self.transform(X)
        attr = {"linear": "support_linear_", "universal": "support_universal_"}.get(which)
        if attr is None:
            raise ValueError(f"transform_usability: which must be 'linear'|'universal'|'nonlinear', got {which!r}")
        candidates = getattr(self, attr, None)
        if candidates is None:
            raise AttributeError(
                f"{attr} is not available: fit MRMR with usability_aware_lists=True and a continuous " f"target to populate it (the '{which}' usability list)."
            )
        from .._usability_lists import materialize_usability_features
        return materialize_usability_features(candidates, X)
