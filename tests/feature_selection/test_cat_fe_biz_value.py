"""Business-value tests for cat-FE.

Per ``mlframe/CLAUDE.md`` ("Every new ML trick gets a biz_value
synthetic test"), every new ML feature must come with a quantitative
test that asserts the trick's MEASURABLE win on a synthetic where the
trick should clearly succeed. If a future code change silently breaks
the trick, the biz_value test FAILS THE WIN, not just an interface or
shape check.

These tests assert concrete numerical thresholds:

1. ``test_biz_cat_fe_recovers_xor_synergy``: cat-FE materialises a
   merged column whose II beats both x1 and x2 marginals by >= 10x
   on the canonical XOR synergy fixture.
2. ``test_biz_cat_fe_synergy_pair_beats_independent_pair``: the
   ranking ordered by II places the XOR pair STRICTLY above every
   noise pair on the same dataset.
3. ``test_biz_cat_fe_disabled_recovers_no_synergy``: with cat-FE
   disabled, no engineered features are produced even on XOR -- this
   is the contrast that justifies enabling cat-FE in the first place.

Tests use fixed seeds and tight thresholds. CI failure on a
``test_biz_*`` is treated as a real regression, NOT a flaky test
(margins are wide enough that random noise doesn't trip them).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import CatFEConfig, MRMR


@pytest.fixture
def xor_4way_dataset():
    """``y = x1 XOR x2`` with 4 noise cat columns. Mid-size n=2000,
    fixed seed for biz-value stability. Plus 2 independent uniform
    cat cols for contrast pairs (n0, n1)."""
    rng = np.random.default_rng(2024)
    n = 2000
    x1 = rng.integers(0, 2, n).astype(np.int8)
    x2 = rng.integers(0, 2, n).astype(np.int8)
    noise = rng.integers(0, 4, size=(n, 4)).astype(np.int8)
    y = (x1 ^ x2).astype(np.int8)
    cols = {"x1": pd.Categorical(x1), "x2": pd.Categorical(x2)}
    for k in range(4):
        cols[f"n{k}"] = pd.Categorical(noise[:, k])
    df = pd.DataFrame(cols)
    return df, pd.Series(y, name="target")


def _fit_cat_fe(df, y, **cfg_overrides):
    defaults = dict(
        enable=True,
        top_k_pairs=8,
        min_interaction_information=0.05,
        full_npermutations=0,  # biz tests focus on II ranking quality, not perm-test
        fwer_correction="none",
    )
    defaults.update(cfg_overrides)
    cfg = CatFEConfig(**defaults)
    mrmr = MRMR(
        full_npermutations=2, baseline_npermutations=2,
        verbose=0, n_jobs=1,
        cat_fe_config=cfg,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(df, y)
    return mrmr


# ---------------------------------------------------------------------------
# Biz value tests
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_biz_cat_fe_recovers_xor_synergy(xor_4way_dataset):
    """Floor 10x; measured (see below). On y = x1 XOR x2, the
    engineered ``kway(x1__x2)`` column has II ≈ ln(2) ≈ 0.69 while
    marginal MIs are ≈ 0. The ratio II/max(marginal) must be >= 10x.

    Captures regressions in: pair search kernel, Jakulin II formula,
    XOR fixture encoding.
    """
    df, y = xor_4way_dataset
    mrmr = _fit_cat_fe(df, y)

    state = mrmr._cat_fe_state_
    assert state is not None, "Cat-FE must run when enabled"
    assert state.recipes, f"No recipes produced; got state={state}"

    xor_diags = [
        d for name, d in state.diagnostics.items()
        if set(d["src_names"]) == {"x1", "x2"}
    ]
    assert xor_diags, \
        f"XOR pair (x1, x2) missing from diagnostics: {list(state.diagnostics.keys())}"
    diag = xor_diags[0]
    ii = diag["II"]
    max_marg = max(abs(diag["marginal_X1_MI"]), abs(diag["marginal_X2_MI"]), 1e-6)
    ratio = ii / max_marg
    assert ratio >= 10.0, (
        f"XOR pair II ({ii:.4f}) must be >=10x larger than max marginal "
        f"MI ({max_marg:.4f}); got {ratio:.2f}x"
    )


def test_biz_cat_fe_synergy_pair_beats_independent_pair(xor_4way_dataset):
    """The XOR pair (x1, x2) MUST rank strictly above every other
    candidate pair by II. Independence-driven pairs (e.g. n0, n1)
    have II ≈ 0; XOR pair has II ≈ 0.69.

    Captures regressions in: argpartition / select_on logic,
    top-K selection order.
    """
    df, y = xor_4way_dataset
    mrmr = _fit_cat_fe(df, y, top_k_pairs=16)  # widen so noise pairs surface too

    state = mrmr._cat_fe_state_
    xor_ii_max = max(
        d["II"] for name, d in state.diagnostics.items()
        if set(d["src_names"]) == {"x1", "x2"}
    )
    # Find max II among pairs that DON'T include both x1 and x2
    non_xor_ii_max = max(
        (d["II"] for name, d in state.diagnostics.items()
         if set(d["src_names"]) != {"x1", "x2"}),
        default=0.0,
    )
    assert xor_ii_max > non_xor_ii_max + 0.3, (
        f"XOR pair must dominate all non-XOR pairs by >=0.3 nat; got "
        f"xor={xor_ii_max:.4f} vs non-xor={non_xor_ii_max:.4f}"
    )


def test_biz_cat_fe_disabled_recovers_no_synergy(xor_4way_dataset):
    """Counter-test: with cat-FE EXPLICITLY disabled
    (``cat_fe_config=CatFEConfig(enable=False)``, the legacy path),
    MRMR produces ZERO recipes on the same XOR dataset. This pins the
    BC contract for users who opt back into legacy.

    2026-05-11: default flipped to enabled, so the legacy path is now
    opt-in via explicit ``CatFEConfig(enable=False)``.

    Captures regressions where the disable path still mutates
    ``_cat_fe_state_``."""
    df, y = xor_4way_dataset
    mrmr = MRMR(
        full_npermutations=2, baseline_npermutations=2,
        verbose=0, n_jobs=1,
        cat_fe_config=CatFEConfig(enable=False),  # explicit legacy opt-in
        # The XOR fixture is integer-categorical, so the SEPARATE default-on integer-lattice /
        # pairwise-modular FE families (own generators + own coverage) fire and emit e.g. il_lcm --
        # orthogonal to the CAT-FE disable contract this test pins. Disable them so the assertion
        # isolates "cat-FE disabled -> cat-FE itself produces nothing" (2026-06-15).
        fe_integer_lattice_enable=False, fe_pairwise_modular_enable=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mrmr.fit(df, y)
    assert mrmr._cat_fe_state_ is None
    assert mrmr._engineered_recipes_ == []


def test_biz_cat_fe_engineered_col_replays_on_test_data(xor_4way_dataset):
    """The engineered column produced at fit time must replay
    correctly on disjoint test data drawn from the same distribution.

    Captures regressions in: factorize-recipe lookup table, transform()
    replay path, unknown-value clipping.
    """
    df_train, y_train = xor_4way_dataset
    # Disjoint test split with same generator: take rows 1500..1999 as test
    df_test = df_train.iloc[1500:].copy()
    df_train = df_train.iloc[:1500].copy()
    y_train = y_train.iloc[:1500]

    mrmr = _fit_cat_fe(df_train, y_train)
    state = mrmr._cat_fe_state_
    # The XOR-4way dataset with fixed seed is supposed to deterministically yield recipes;
    # an empty recipe set is a real undertraining regression, not a benign data condition
    # (memory feedback_no_mask_via_canon_or_guards).
    assert state.recipes, (
        "cat_fe state.recipes is empty on XOR-4way dataset - undertraining regression. "
        "If MRMR config changed, broaden the fitting budget rather than silently skipping."
    )

    # Pick the XOR recipe and replay manually
    from mlframe.feature_selection.filters.engineered_recipes import apply_recipe

    xor_recipe = next(
        (r for r in state.recipes if set(r.src_names) == {"x1", "x2"}),
        None,
    )
    assert xor_recipe is not None, (
        "XOR pair (x1, x2) must be selected on the XOR-4way synthetic; "
        "missing selection indicates a regression in cat_fe recipe discovery."
    )

    # Replay on test
    out = apply_recipe(xor_recipe, df_test)
    assert out.shape == (len(df_test),)
    # All values must be valid post-prune class ids
    assert out.min() >= 0
    assert out.max() < xor_recipe.extra["n_uniq_post_prune"]
    # The engineered column should still discriminate y_train's law:
    # 4 unique cells of (x1, x2) map to 4 distinct merged classes.
    # Test should see same encoding on its rows.
    expected_unique = len(set(
        (int(a), int(b))
        for a, b in zip(df_test["x1"].cat.codes, df_test["x2"].cat.codes)
    ))
    assert len(set(out)) == expected_unique, \
        f"Replay produced {len(set(out))} unique classes, expected {expected_unique}"
