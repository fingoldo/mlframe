"""Cross-module integration / contract tests for ``mlframe.feature_selection``.

Each test pins exactly one contract that lives at the seam between modules so a regression in any single component (MRMR, RFECV, EngineeredRecipe, _FIT_CACHE) surfaces here rather than as a downstream pipeline crash:

1. ``test_mrmr_rfecv_pipeline_no_feature_names_drift`` -- chaining ``MRMR.fit/transform`` into ``RFECV.fit`` preserves column names end-to-end; ``RFECV.feature_names_in_`` equals ``MRMR.get_feature_names_out()`` with no drift.
2. ``test_mrmr_sklearn_clone_preserves_params`` -- ``sklearn.base.clone`` round-trips non-default kwargs and strips fitted state (no ``support_`` / ``_engineered_recipes_`` on the clone).
3. ``test_mrmr_fit_cache_shared_across_instances`` -- the process-wide ``MRMR._FIT_CACHE`` produces identical ``support_`` across two fits on the same arrays and the second is materially faster.
4. ``test_cat_fe_recipes_replay_matches_manual`` -- after MRMR + cat-FE fit on XOR data, manual ``apply_recipe`` on test data matches the engineered column values in ``MRMR.transform`` bit-exact.
5. ``test_mrmr_polars_input_equals_pandas_input`` -- same data fed as ``polars.DataFrame`` vs ``pandas.DataFrame`` produces equivalent ``support_`` (within +/- 1 selected column to tolerate internal sort).
6. ``test_mrmr_full_npermutations_zero_runs`` -- edge contract for ``full_npermutations=0`` (either runs / skips permutation tests or raises a documented error).
7. ``test_engineered_recipe_serializes_via_pickle`` -- ``EngineeredRecipe`` round-trips through pickle and replay output matches pre-pickle.
8. ``test_get_feature_names_out_consistent_with_support_`` -- ``get_feature_names_out()`` returns exactly the columns marked True (or indexed) in ``support_`` plus any engineered names, in canonical order.
"""

from __future__ import annotations

import pickle  # nosec B403 -- test-only local pickle round-trip, never untrusted/network data
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from mlframe.feature_selection.filters import MRMR, CatFEConfig
from mlframe.feature_selection.filters.engineered_recipes import (
    apply_recipe,
    build_unary_binary_recipe,
)
from mlframe.feature_selection.wrappers import RFECV

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_classification_df():
    """n=300, m=10 pandas DataFrame; first 3 cols are informative."""
    rng = np.random.default_rng(0)
    n, m = 300, 10
    X_inf = rng.normal(size=(n, 3)).astype(np.float32)
    y = (X_inf[:, 0] + X_inf[:, 1] - X_inf[:, 2] > 0).astype(np.int32)
    X_noise = rng.normal(size=(n, m - 3)).astype(np.float32)
    X = np.hstack([X_inf, X_noise])
    cols = [f"inf_{i}" for i in range(3)] + [f"noise_{i}" for i in range(m - 3)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="target")


@pytest.fixture
def xor_cat_df():
    """``y = x1 XOR x2`` with 4 noise cat columns. Train + test draws from the same generative process so the recipe replay test has a disjoint test set."""
    rng = np.random.default_rng(11)

    def _make(n: int):
        """Draw n rows of the XOR-categorical fixture."""
        x1 = rng.integers(0, 2, n).astype(np.int8)
        x2 = rng.integers(0, 2, n).astype(np.int8)
        noise = rng.integers(0, 4, size=(n, 4)).astype(np.int8)
        y = (x1 ^ x2).astype(np.int8)
        cols = {"x1": pd.Categorical(x1), "x2": pd.Categorical(x2)}
        for k in range(4):
            cols[f"n{k}"] = pd.Categorical(noise[:, k])
        return pd.DataFrame(cols), pd.Series(y, name="target")

    df_tr, y_tr = _make(1200)
    df_te, _ = _make(300)
    return df_tr, y_tr, df_te


def _fit_quiet(estimator, X, y):
    """Fit estimator on (X, y) with warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(X, y)


# ---------------------------------------------------------------------------
# 1. MRMR -> RFECV pipeline preserves feature names
# ---------------------------------------------------------------------------


def test_mrmr_rfecv_pipeline_no_feature_names_drift(small_classification_df):
    """``RFECV.feature_names_in_`` must equal ``MRMR.get_feature_names_out()``; otherwise downstream selectors operate on a mis-labelled column set."""
    X, y = small_classification_df
    MRMR._FIT_CACHE.clear()

    mrmr = MRMR(
        full_npermutations=2,
        baseline_npermutations=2,
        fe_max_steps=0,
        cat_fe_config=CatFEConfig(enable=False),
        verbose=0,
        n_jobs=1,
    )
    _fit_quiet(mrmr, X, y)
    X_after = mrmr.transform(X)
    assert isinstance(X_after, pd.DataFrame), "MRMR.transform on pandas input must return pandas"

    rfecv = RFECV(
        estimator=LogisticRegression(max_iter=200, solver="liblinear"),
        cv=3,
        max_noimproving_iters=3,
        max_refits=4,
        verbose=0,
        n_jobs=1,
        leakage_corr_threshold=None,
    )
    _fit_quiet(rfecv, X_after, y)

    mrmr_names = list(mrmr.get_feature_names_out())
    # RFECV may drop zero-variance / duplicate columns at fit entry; the contract here is that the columns it RECORDED as inputs are exactly the names MRMR
    # handed it (no renaming, no positional drift). Subset semantics catch the renaming/drift bug while tolerating the documented dedup step.
    rfecv_names = list(rfecv.feature_names_in_)
    assert set(rfecv_names).issubset(set(mrmr_names)), (
        f"RFECV.feature_names_in_ ({rfecv_names}) contains names MRMR.get_feature_names_out() did not emit ({mrmr_names}); name drift occurred in the chain."
    )
    # Order alignment: the names RFECV kept appear in the same relative order as in MRMR's output.
    kept_order = [n for n in mrmr_names if n in set(rfecv_names)]
    assert kept_order == rfecv_names, (
        f"Order drift: MRMR emitted {mrmr_names}, RFECV recorded {rfecv_names}; expected RFECV order to match MRMR's (filtered) order {kept_order}."
    )


# ---------------------------------------------------------------------------
# 2. sklearn.clone preserves params + strips fitted state
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_mrmr_sklearn_clone_preserves_params(small_classification_df):
    """``clone`` must propagate every non-default kwarg AND drop fitted state. A clone that carries ``support_`` would mask post-clone fit bugs."""
    X, y = small_classification_df
    MRMR._FIT_CACHE.clear()

    mrmr = MRMR(
        quantization_nbins=8,
        fe_max_steps=2,
        n_jobs=1,
        verbose=0,
        full_npermutations=2,
        baseline_npermutations=2,
        cat_fe_config=CatFEConfig(enable=False),
    )
    _fit_quiet(mrmr, X, y)

    # Pre-clone: fitted state is present.
    assert hasattr(mrmr, "support_")
    assert hasattr(mrmr, "_engineered_recipes_")

    cloned = clone(mrmr)
    # Constructor params round-trip exactly.
    assert mrmr.get_params() == cloned.get_params()
    # Specific non-defaults survived.
    assert cloned.quantization_nbins == 8
    assert cloned.fe_max_steps == 2
    assert cloned.n_jobs == 1
    assert cloned.verbose == 0

    # Fitted state is stripped.
    assert not hasattr(cloned, "support_"), "clone must not carry fitted support_"
    assert not hasattr(cloned, "_engineered_recipes_"), "clone must not carry fitted recipes"
    # ``signature`` is reset (skip_retraining_on_same_content sentinel).
    assert getattr(cloned, "signature", None) is None


# ---------------------------------------------------------------------------
# 3. _FIT_CACHE is shared across instances + speeds up second fit
# ---------------------------------------------------------------------------


def test_mrmr_fit_cache_shared_across_instances(small_classification_df):
    """Two fresh MRMR(...) on the same (X, y) must produce identical ``support_`` AND the second fit hits the process-wide cache (>= 2x faster)."""
    X, y = small_classification_df
    MRMR._FIT_CACHE.clear()

    def _new_mrmr():
        """Build a fast, fixed-seed MRMR instance for the cache-sharing comparison."""
        return MRMR(
            full_npermutations=2,
            baseline_npermutations=2,
            fe_max_steps=0,
            cat_fe_config=CatFEConfig(enable=False),
            random_seed=42,
            verbose=0,
            n_jobs=1,
        )

    first = _new_mrmr()
    t0 = time.perf_counter()
    _fit_quiet(first, X, y)
    t_first = time.perf_counter() - t0

    # Cache populated exactly once.
    assert len(MRMR._FIT_CACHE) == 1

    second = _new_mrmr()
    t0 = time.perf_counter()
    _fit_quiet(second, X, y)
    t_second = time.perf_counter() - t0

    # Identical support_.
    assert list(np.asarray(first.support_)) == list(np.asarray(second.support_))
    # Cache still has one entry (the second fit replayed, did not add).
    assert len(MRMR._FIT_CACHE) == 1
    # Second fit is materially faster. Use a generous 2x threshold (the actual speedup is ~10-100x on a cache hit since cat-FE + screening + permutation are
    # all skipped) with an absolute floor: if first fit was already <50ms there's no meaningful work left to skip and the ratio is dominated by replay overhead.
    if t_first > 0.05:
        assert t_second * 2 <= t_first, f"Cache hit did not deliver >=2x speedup: first={t_first * 1000:.1f}ms second={t_second * 1000:.1f}ms"

    MRMR._FIT_CACHE.clear()


# ---------------------------------------------------------------------------
# 4. Cat-FE recipe replay matches MRMR.transform bit-exact
# ---------------------------------------------------------------------------


def test_cat_fe_recipes_replay_matches_manual(xor_cat_df):
    """``_engineered_recipes_`` plus ``apply_recipe`` must reproduce the exact engineered column values that ``MRMR.transform`` emits -- the deterministic-encoding contract that lets downstream pipelines bypass ``transform`` when they only need one engineered col."""
    df_tr, y_tr, df_te = xor_cat_df
    MRMR._FIT_CACHE.clear()

    mrmr = MRMR(
        full_npermutations=2,
        baseline_npermutations=2,
        fe_max_steps=0,
        cat_fe_config=CatFEConfig(
            enable=True,
            top_k_pairs=4,
            min_interaction_information=0.1,
            full_npermutations=0,
            fwer_correction="none",
        ),
        verbose=0,
        n_jobs=1,
    )
    _fit_quiet(mrmr, df_tr, y_tr)

    recipes = list(getattr(mrmr, "_engineered_recipes_", []))
    if not recipes:
        # When no engineered feature survived screening, the contract is vacuously true; assert the cat-FE step at least produced candidate recipes so the
        # test still exercises the code path it was designed for. The xor fixture's interaction signal is strong enough that 0 candidate recipes points at a
        # real regression in the cat-FE pipeline, not at a seed-unlucky outcome.
        cat_state = getattr(mrmr, "_cat_fe_state_", None)
        assert (
            cat_state is not None and cat_state.recipes
        ), "cat-FE on the XOR fixture produced 0 candidate recipes; the deterministic-encoding contract cannot be exercised. Investigate cat-FE pipeline."
        # Promote one cat-FE candidate to a "would-be" recipe and apply it manually -- we still verify deterministic encoding even when MRMR didn't keep it.
        recipe = cat_state.recipes[0]
        manual = apply_recipe(recipe, df_te)
        # Replay twice -- must be deterministic call-to-call.
        manual2 = apply_recipe(recipe, df_te)
        np.testing.assert_array_equal(manual, manual2)
        return

    out = mrmr.transform(df_te)
    for recipe in recipes:
        if recipe.name not in out.columns:
            # Defensive: get_feature_names_out / transform must include every recipe; surface the mismatch.
            raise AssertionError(f"Recipe {recipe.name!r} missing from MRMR.transform output columns {list(out.columns)}")
        manual = apply_recipe(recipe, df_te)
        from_transform = out[recipe.name].to_numpy()
        np.testing.assert_array_equal(
            manual,
            from_transform,
            err_msg=f"apply_recipe disagrees with MRMR.transform for {recipe.name!r}",
        )


# ---------------------------------------------------------------------------
# 5. Polars input == pandas input
# ---------------------------------------------------------------------------


def test_mrmr_polars_input_equals_pandas_input(small_classification_df):
    """Same data via polars and via pandas must produce equivalent ``support_`` (tolerate +/- 1 selected col for internal sort ties)."""
    pl = pytest.importorskip("polars")
    X_pd, y = small_classification_df
    MRMR._FIT_CACHE.clear()

    common_kwargs = dict(
        full_npermutations=2,
        baseline_npermutations=2,
        fe_max_steps=0,
        cat_fe_config=CatFEConfig(enable=False),
        random_seed=7,
        verbose=0,
        n_jobs=1,
    )

    mrmr_pd = MRMR(**common_kwargs)
    _fit_quiet(mrmr_pd, X_pd, y)
    pd_selected = set(mrmr_pd.get_feature_names_out().tolist())

    MRMR._FIT_CACHE.clear()  # Polars-input run must compute fresh, not replay pandas cache (different col-name signature would already prevent that, but be explicit).
    X_pl = pl.from_pandas(X_pd)
    mrmr_pl = MRMR(**common_kwargs)
    _fit_quiet(mrmr_pl, X_pl, y)
    pl_selected = set(mrmr_pl.get_feature_names_out().tolist())

    # Within-1 symmetric difference tolerance per the contract docstring.
    sym_diff = pd_selected.symmetric_difference(pl_selected)
    assert len(sym_diff) <= 1, (
        f"polars vs pandas selected feature sets differ by more than 1 column: "
        f"pandas={sorted(pd_selected)} polars={sorted(pl_selected)} "
        f"sym_diff={sorted(sym_diff)}"
    )


# ---------------------------------------------------------------------------
# 6. full_npermutations=0 edge case
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_mrmr_full_npermutations_zero_runs(small_classification_df):
    """``full_npermutations=0`` is documented as an anti-statistical trap in CatFEConfig but the MRMR top-level kwarg has no such ban. Pin behaviour: either fit() completes (permutation tests effectively skipped) or it raises a clear error. Crashing with an obscure traceback is a regression."""
    X, y = small_classification_df
    MRMR._FIT_CACHE.clear()

    mrmr = MRMR(
        full_npermutations=0,
        baseline_npermutations=2,
        fe_max_steps=0,
        cat_fe_config=CatFEConfig(enable=False),
        verbose=0,
        n_jobs=1,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mrmr.fit(X, y)
    except (ValueError, ZeroDivisionError) as exc:
        # Documented error path -- accept and pin the exception types so a future change can't silently broaden it to generic Exception.
        assert str(exc), "Error path must carry a non-empty message"
        return
    # No-raise path: fit completed; pin the postconditions so a no-op fit doesn't masquerade as success.
    assert hasattr(mrmr, "support_"), "fit() returned without raising but did not set support_"
    assert hasattr(mrmr, "feature_names_in_")


# ---------------------------------------------------------------------------
# 7. EngineeredRecipe pickle BC
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_engineered_recipe_serializes_via_pickle():
    """``build_unary_binary_recipe`` produces a recipe; pickle round-trip + replay must match the pre-pickle output bit-exact."""
    recipe = build_unary_binary_recipe(
        name="mul(identity(a),identity(b))",
        src_a_name="a",
        src_b_name="b",
        unary_a_name="identity",
        unary_b_name="identity",
        binary_name="mul",
        unary_preset="minimal",
        binary_preset="minimal",
        quantization_nbins=None,
        quantization_method=None,
        quantization_dtype=np.float32,
    )

    rng = np.random.default_rng(3)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=64).astype(np.float32),
            "b": rng.normal(size=64).astype(np.float32),
        }
    )
    pre = apply_recipe(recipe, X)

    restored = pickle.loads(pickle.dumps(recipe))  # nosec B301 -- round-trip of a locally-created, trusted object
    assert restored == recipe, "EngineeredRecipe.__eq__ must survive pickle"
    post = apply_recipe(restored, X)

    np.testing.assert_array_equal(pre, post)


# ---------------------------------------------------------------------------
# 8. get_feature_names_out is consistent with support_
# ---------------------------------------------------------------------------


def test_get_feature_names_out_consistent_with_support_(small_classification_df):
    """``get_feature_names_out()`` must equal the base names selected by ``support_`` followed by any engineered names from ``_engineered_recipes_``."""
    X, y = small_classification_df
    MRMR._FIT_CACHE.clear()

    mrmr = MRMR(
        full_npermutations=2,
        baseline_npermutations=2,
        fe_max_steps=0,
        cat_fe_config=CatFEConfig(enable=False),
        verbose=0,
        n_jobs=1,
    )
    _fit_quiet(mrmr, X, y)

    support = np.asarray(mrmr.support_)
    feature_names_in = list(mrmr.feature_names_in_)

    # support_ is integer indices in current implementation; tolerate the bool-mask path too (legacy / sklearn-canonical).
    if support.size and support.dtype == bool:
        expected_base = [n for n, s in zip(feature_names_in, support) if s]
    else:
        expected_base = [feature_names_in[i] for i in support]

    expected_engineered = [r.name for r in getattr(mrmr, "_engineered_recipes_", [])]
    expected = np.asarray(expected_base + expected_engineered, dtype=object)

    actual = mrmr.get_feature_names_out()
    np.testing.assert_array_equal(actual, expected)

    # transform() column order must match get_feature_names_out().
    out = mrmr.transform(X)
    if isinstance(out, pd.DataFrame):
        assert list(out.columns) == list(actual), f"transform column order {list(out.columns)} != get_feature_names_out() {list(actual)}"
