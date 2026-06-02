"""RC2 biz_value contracts: sample-size-aware Fleuret confirmation.

THE BUG (RC2)
-------------
With the default ``use_simple_mode=False`` (full Fleuret conditional-MI
redundancy) MRMR catastrophically UNDER-selects on small-n / high-cardinality
data. The Fleuret permutation-confidence test estimates ``I(X; Y | Z)`` over the
``(X, Y, Z)`` joint histogram. On a small-n / high-cardinality conditioning
joint that joint is severely undersampled (sklearn diabetes, n=442, s5 has 10
bins -> the (X, Y, Z) joint is ~10*10*10=1000 cells over 442 rows, ~0.4
rows/cell), so the conditional-MI estimate is dominated by finite-sample bias
and the SHUFFLED-y NULL conditional MI is ~= the REAL conditional MI. The gate
rejects on a single permuted-exceedance, so every genuine feature after the
first is rejected -> premature stop (PROVEN: full mode selected only 's5',
downstream 5-fold Ridge R2 = 0.20, vs simple-mode 9 feats R2 = 0.39).

THE FIX
-------
``confirm_candidate`` (``filters/_confirm_predictor.py``) measures the
``(X, Y, selected_vars)`` joint's rows-per-occupied-cell. When below
``fe_confirm_undersample_rows_per_cell`` (MRMR ctor default 5.0) the conditional
permutation test is finite-sample unreliable, so the candidate is confirmed by
the MARGINAL-MI permutation test instead (the X-marginal joint is far better
sampled). Dedup is preserved by the relevance-minus-redundancy gain term; pure
noise is still rejected because its marginal permutation test rejects it.

CONTRACTS PINNED (all falsifiable -- revert the fix and they fail)
------------------------------------------------------------------
* (a) diabetes full-mode: >= 6 features AND downstream 5-fold Ridge R2 >= 0.35
* (b) an exact duplicate of a selected column is still dropped
* (c) pure noise (y independent of X), small n -> very few / no spurious feature
* (d) UNIT: an undersampled conditioning joint with a marginally-significant
      candidate is CONFIRMED via the marginal fallback (and the strict legacy
      path -- threshold 0 -- rejects the same candidate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from mlframe.feature_selection.filters.mrmr import MRMR


def _raw_selected(sel, allowed_cols):
    """Selected feature names restricted to the raw input columns (FE off, but
    guard against any engineered name slipping through)."""
    return [f for f in sel.get_feature_names_out() if f in set(allowed_cols)]


# ---------------------------------------------------------------------------
# (a) REGRESSION: diabetes full-mode recovers >= 6 feats / R2 >= 0.35.
# ---------------------------------------------------------------------------
def test_rc2_diabetes_full_mode_recovers_features_and_r2():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    assert len(X) == 442  # the small-n regime that triggered the bug

    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=0, use_simple_mode=False)
    sel.fit(X, y)
    feats = _raw_selected(sel, X.columns)

    # Was 1 feature ('s5') pre-fix.
    assert len(feats) >= 6, f"under-selected: {feats}"

    Xs = X[feats].to_numpy()
    r2 = cross_val_score(Ridge(), Xs, y, cv=5, scoring="r2").mean()
    # Was R2=0.20 pre-fix; simple-mode reference ~0.40; all-10 baseline ~0.41.
    assert r2 >= 0.35, f"downstream R2 too low: {r2:.4f} with {feats}"


def test_rc2_diabetes_legacy_strict_path_still_underselects():
    """The bug is real and the threshold is the lever: with
    ``fe_confirm_undersample_rows_per_cell=0`` (strict conditional test
    everywhere, i.e. legacy behaviour) full mode still collapses to 1 feature.
    Pins that the default-ON fix -- not some unrelated change -- is what fixes
    (a)."""
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    sel = MRMR(
        verbose=0, random_seed=42, fe_max_steps=0,
        use_simple_mode=False, fe_confirm_undersample_rows_per_cell=0.0,
    )
    sel.fit(X, y)
    feats = _raw_selected(sel, X.columns)
    assert len(feats) <= 2, f"legacy strict path unexpectedly recovered: {feats}"


# ---------------------------------------------------------------------------
# (b) Exact-duplicate column is still dropped (dedup preserved by the gain).
# ---------------------------------------------------------------------------
def test_rc2_exact_duplicate_dropped():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    Xd = X.copy()
    Xd["s5_dup"] = Xd["s5"]  # exact copy of a strong feature

    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=0, use_simple_mode=False)
    sel.fit(Xd, y)
    feats = sel.get_feature_names_out()

    assert "s5_dup" not in feats, f"exact duplicate not dropped: {list(feats)}"
    # The original is still selected (we dropped the dup, not the signal).
    assert "s5" in feats


# ---------------------------------------------------------------------------
# (c) Pure noise (y independent of X), small n -> no spurious admission.
# ---------------------------------------------------------------------------
def test_rc2_pure_noise_rejected_small_n():
    rng = np.random.RandomState(0)
    n = 300
    Xn = pd.DataFrame(rng.randn(n, 8), columns=[f"n{i}" for i in range(8)])
    yn = pd.Series(rng.randn(n), name="t")  # independent of every column

    sel = MRMR(verbose=0, random_seed=42, fe_max_steps=0, use_simple_mode=False)
    sel.fit(Xn, yn)
    feats = list(sel.get_feature_names_out())

    # The marginal fallback must NOT open the floodgates to noise: a couple of
    # by-chance first-picks are tolerable, a full pile-up is not.
    assert len(feats) <= 2, f"marginal fallback admitted noise: {feats}"


# ---------------------------------------------------------------------------
# (d) UNIT: undersampled conditioning -> marginal fallback CONFIRMS a
#     marginally-significant candidate that the strict conditional path rejects.
# ---------------------------------------------------------------------------
def test_rc2_undersampled_fallback_confirms_marginal_signal_unit():
    """Focused unit test of the undersampled-conditioning fallback in
    ``confirm_candidate``.

    Construct a small-n dataset with a strong anchor feature and a second
    feature that is genuinely (marginally) related to y but COLLINEAR with the
    anchor, so conditioning on the anchor over a high-cardinality joint makes
    the conditional permutation test unreliable. With the fallback ON (default)
    the second feature is confirmed; with the strict conditional path
    (threshold 0) it is rejected -- demonstrating both the bug and the fix on a
    controlled case."""
    rng = np.random.RandomState(7)
    n = 250
    latent = rng.randn(n)
    df = pd.DataFrame({
        "anchor": latent + 0.05 * rng.randn(n),
        "collinear": latent + 0.05 * rng.randn(n),  # marginally significant, collinear with anchor
    })
    yv = pd.Series(latent + 0.3 * rng.randn(n), name="t")

    # Fallback ON (default): undersampled (anchor, y, collinear) joint -> marginal
    # confirmation -> both features survive.
    sel_on = MRMR(verbose=0, random_seed=1, fe_max_steps=0, use_simple_mode=False)
    sel_on.fit(df, yv)
    feats_on = [f for f in sel_on.get_feature_names_out() if f in ("anchor", "collinear")]

    # Strict legacy conditional path: the second collinear feature is rejected
    # by the over-rejecting permutation gate.
    sel_off = MRMR(
        verbose=0, random_seed=1, fe_max_steps=0,
        use_simple_mode=False, fe_confirm_undersample_rows_per_cell=0.0,
    )
    sel_off.fit(df, yv)
    feats_off = [f for f in sel_off.get_feature_names_out() if f in ("anchor", "collinear")]

    assert len(feats_on) >= 2, f"fallback failed to confirm marginal signal: {feats_on}"
    assert len(feats_on) > len(feats_off), (
        f"fallback (on={feats_on}) must recover more than strict path (off={feats_off})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--no-cov"])
