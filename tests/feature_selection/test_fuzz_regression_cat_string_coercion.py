"""Regression sensors for fuzz-suite cat-string float-coercion crashes.

Promoted from ``tests/training/test_fuzz_suite.py`` n1000 combos that crashed
when a categorical / string column (NA-like tokens ``"NA"`` / ``"None"``,
unseen levels ``"ZZZ_UNSEEN"``, empty ``""``, unicode) reached a NUMERIC
operation in the feature-selection / FE path. Each test pins one prod call
site that previously float-coerced the raw strings and raised
``could not convert string to float`` / ``TypeError: ufunc 'isfinite' ...`` /
``unsupported operand type(s) for /: 'str' and 'int'``.

Also pins the ``nbins_strategy='quantile'`` alias acceptance (c0149).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _mixed_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "num_0": rng.standard_normal(n).astype("float32"),
            "num_1": rng.standard_normal(n).astype("float32"),
            # Categorical with NA-like / empty / unseen string levels -- the exact
            # ``weird_cat_content`` + ``inject_test_drift`` content the fuzz emits.
            "cat_0": pd.Categorical(
                [["A", "B", "NA", "None", "", "ZZZ_UNSEEN"][i % 6] for i in range(n)]
            ),
        }
    )


class TestGroupAwareRedundancyOnStringCats:
    """``group_aware._redundancy_matrix`` / ``_su_redundancy_matrix`` must not
    float-coerce raw string categorical columns (fuzz c0013)."""

    @pytest.mark.parametrize("method", ["spearman", "pearson", "su"])
    def test_cluster_features_by_correlation_tolerates_string_cats(self, method):
        from mlframe.feature_selection.filters.group_aware import (
            cluster_features_by_correlation,
        )

        X = _mixed_frame()
        # Pre-fix: pandas ``DataFrame.corr`` / ``X.to_numpy()`` raises
        # ``could not convert string to float: 'NA'`` on the cat column.
        cluster_id = cluster_features_by_correlation(X, threshold=0.9, method=method)
        assert cluster_id.shape == (X.shape[1],)
        assert cluster_id.dtype.kind in ("i", "u")


class TestBorutaPremergeOnStringCats:
    """``_premerge_collapse`` runs ``np.corrcoef`` on raw values and must
    factor-code string categoricals first (fuzz c0149)."""

    def test_premerge_collapse_tolerates_string_cats(self):
        from mlframe.feature_selection.boruta_shap._fit_explain import _premerge_collapse

        X = _mixed_frame()
        # Pre-fix: ``np.corrcoef(X.values)`` -> ``unsupported operand type(s)
        # for /: 'str' and 'int'`` on the string column.
        X_reps, members = _premerge_collapse(X, thr=0.95)
        assert set(X_reps.columns).issubset(set(X.columns))
        assert all(isinstance(v, list) for v in members.values())


class TestMrmrFeStepNumericGuard:
    """``_run_fe_step`` must drop non-numeric operands from the pair / synergy
    FE pool even when they were not flagged in ``categorical_vars`` (fuzz
    c0114 -- Hermite pair FE hit ``ufunc 'isfinite'`` on a string column)."""

    def test_non_numeric_column_indices_pandas(self):
        from mlframe.feature_selection.filters._mrmr_fe_step import (
            _non_numeric_column_indices,
        )

        X = _mixed_frame()
        idx = _non_numeric_column_indices(X, list(X.columns))
        # cat_0 is at position 2.
        assert idx == {2}

    def test_non_numeric_column_indices_all_numeric(self):
        from mlframe.feature_selection.filters._mrmr_fe_step import (
            _non_numeric_column_indices,
        )

        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3, 4]})
        assert _non_numeric_column_indices(X, list(X.columns)) == set()

    def test_mrmr_fit_with_string_cat_does_not_crash_in_fe(self):
        """End-to-end: MRMR with FE enabled on a frame whose cat column is NOT
        declared categorical must not crash in the numeric pair/Hermite FE."""
        from mlframe.feature_selection.filters.mrmr import MRMR

        n = 400
        rng = np.random.default_rng(3)
        a = rng.standard_normal(n)
        b = rng.standard_normal(n)
        y = (a * b > 0).astype(int)
        X = pd.DataFrame(
            {
                "a": a.astype("float32"),
                "b": b.astype("float32"),
                # Raw object/string column, deliberately NOT passed via
                # categorical_features so MRMR would treat it as a numeric operand.
                "cat_0": [["A", "B", "NA", ""][i % 4] for i in range(n)],
            }
        )
        sel = MRMR(
            fe_max_steps=1,
            nbins_strategy="quantile",
        )
        # Must complete without raising on the string column.
        sel.fit(X, y)
        assert hasattr(sel, "support_")


class TestMrmrOrphanRecipeReplay:
    """``_append_engineered`` must not crash when a recipe references an engineered
    source column that no surviving recipe produces (fit-time pruning dropped a
    chained producer); it emits a neutral zero-variance column instead of KeyError
    (fuzz c0094). The column width is fixed by ``get_feature_names_out`` so the
    feature cannot be physically removed; a constant 0.0 carries no signal (so the
    feature is "dropped" in effect) yet -- unlike a NaN placeholder, the prior
    behaviour -- is accepted by every downstream estimator (LogisticRegression et al.
    reject NaN), see ``_mrmr_validate_transform`` orphan-recipe handling."""

    def test_orphan_recipe_emits_neutral_zero_not_keyerror(self):
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

        sel = MRMR()
        sel.verbose = 0
        sel.feature_names_in_ = ["a", "b"]
        base = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        # Recipe whose source 'cross_missing' is absent from X and produced by no
        # other recipe -> unreconstructable. Must degrade to a neutral zero column.
        orphan = EngineeredRecipe(
            name="modular_of_cross",
            kind="modular",
            src_names=("cross_missing",),
            extra={"period": 7.0, "op": "sin"},
        )
        out = sel._append_engineered(base.copy(), base.copy(), [orphan])
        assert "modular_of_cross" in out.columns
        col = out["modular_of_cross"]
        # No KeyError (the core contract) AND a finite, signal-free, estimator-safe column:
        assert not col.isna().any(), "orphan replay must be NaN-free (NaN crashes NaN-rejecting estimators)"
        assert (col == 0.0).all(), "orphan replay must be the neutral zero-variance column"

    def test_chained_recipe_replays_out_of_order(self):
        """A consumer recorded BEFORE its producer still replays (dependency-aware)."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

        sel = MRMR()
        sel.verbose = 0
        sel.feature_names_in_ = ["a"]
        base = pd.DataFrame({"a": [10.0, 20.0, 30.0]})
        producer = EngineeredRecipe(name="round_a", kind="numeric_rounding", src_names=("a",), extra={"precision": 1.0})
        consumer = EngineeredRecipe(name="round_round_a", kind="numeric_rounding", src_names=("round_a",), extra={"precision": 1.0})
        # Consumer listed FIRST (out of topological order).
        out = sel._append_engineered(base.copy(), base.copy(), [consumer, producer])
        assert "round_round_a" in out.columns and "round_a" in out.columns
        assert out["round_round_a"].notna().all()

    def test_raw_seed_only_kind_raises_on_missing_source(self):
        """A ``mi_greedy_transform`` recipe consumes raw input columns only, so a source absent from X is a recipe-vs-X mismatch and MUST raise
        (naming the column), never silently emit NaN -- the raise-vs-NaN split is keyed on the recipe kind's data-flow contract."""
        from mlframe.feature_selection.filters.mrmr import MRMR
        from mlframe.feature_selection.filters.engineered_recipes import EngineeredRecipe

        sel = MRMR()
        sel.verbose = 0
        sel.feature_names_in_ = ["a", "b"]
        base = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        bad = EngineeredRecipe(
            name="square(missing_input)",
            kind="mi_greedy_transform",
            src_names=("missing_input",),
            extra={"transform": "square"},
        )
        with pytest.raises(KeyError) as exc_info:
            sel._append_engineered(base.copy(), base.copy(), [bad])
        assert "missing_input" in str(exc_info.value)


class TestMrmrNbinsQuantileAlias:
    """``nbins_strategy='quantile'`` is a natural user alias for quantile-spacing
    ('qs') and must be accepted by the validator + dispatcher (fuzz c0149)."""

    def test_quantile_alias_resolves_to_qs(self):
        from mlframe.feature_selection.filters._adaptive_nbins import _METHOD_ALIASES

        assert _METHOD_ALIASES["quantile"] == "qs"

    def test_quantile_in_mrmr_valid_strategies(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        assert "quantile" in MRMR._VALID_NBINS_STRATEGIES

    def test_per_feature_edges_accepts_quantile(self):
        from mlframe.feature_selection.filters._adaptive_nbins import per_feature_edges

        rng = np.random.default_rng(7)
        X = rng.standard_normal((500, 2))
        edges = per_feature_edges(X, method="quantile")
        assert len(edges) == 2

    def test_mrmr_validate_accepts_quantile_strategy(self):
        from mlframe.feature_selection.filters.mrmr import MRMR

        sel = MRMR(nbins_strategy="quantile")
        # Pre-fix: ``_validate_string_params`` raised
        # ``MRMR: nbins_strategy='quantile' is not a recognised value``.
        sel._validate_string_params()
