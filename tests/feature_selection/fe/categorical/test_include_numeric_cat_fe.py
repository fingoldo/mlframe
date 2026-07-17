"""``CatFEConfig(include_numeric=True)``: numeric columns are quantile-binned into the cat-FE
candidate pool so their pair crosses capture interactions the numeric unary/binary FE cannot
express (axis-aligned AND non-product / rotated). Validates:

* the engineered cross is built from RAW numeric sources and carries stored quantile edges;
* transform replay is LEAK-SAFE and free of train/serve skew (raw float values are binned through
  the stored edges, NOT int-cast) -- the replayed column matches an independent edge recomputation;
* NaN-bearing numeric columns are skipped (v1 has no NaN bin in the quantile-edge replay), so no
  unreplayable recipe is produced;
* the gate: ``include_numeric=False`` produces no numeric cross even on a pure-numeric frame;
* business value: on a ROTATED (diagonal) interaction -- where ``mul(a,b)`` is useless -- the
  include_numeric cross lifts a held-out LogisticRegression vs the same MRMR without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters import MRMR, CatFEConfig
from mlframe.feature_selection.filters.engineered_recipes._recipe_dispatch import apply_recipe


def _rotated_xor(n: int, seed: int):
    """Diagonal quadrant interaction: ``mul(a,b)`` cannot express it; only a 2-D bin cross can."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 1, n)
    x1 = rng.uniform(0, 1, n)
    u = (x0 - 0.5 + x1 - 0.5) / np.sqrt(2.0)
    v = (x0 - 0.5 - (x1 - 0.5)) / np.sqrt(2.0)
    logit = 3.0 * np.sign(u * v)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < p).astype(np.int64)
    cols = {"x0": x0, "x1": x1}
    for j in range(4):
        cols[f"noise{j}"] = rng.normal(0, 1, n)
    return pd.DataFrame(cols), y


def _numeric_cross_recipes(mrmr):
    recs = getattr(mrmr, "_cat_fe_state_", None)
    out = []
    for r in recs.recipes if recs else []:
        if r.kind == "factorize" and (r.extra.get("src_bin_edges")):
            out.append(r)
    return out


def test_include_numeric_builds_cross_with_stored_edges():
    df, y = _rotated_xor(3000, seed=1)
    mrmr = MRMR(
        cat_fe_config=CatFEConfig(enable=True, include_numeric=True, numeric_nbins=8),
        fe_max_steps=0,
        verbose=0,
    )
    mrmr.fit(df, y)
    crosses = _numeric_cross_recipes(mrmr)
    assert crosses, "include_numeric must produce at least one numeric-sourced factorize cross"
    r = crosses[0]
    # Sources are RAW user columns, edges stored for both numeric sources.
    assert set(r.src_names) <= set(df.columns)
    edges = r.extra["src_bin_edges"]
    for s in r.src_names:
        assert s in edges and np.asarray(edges[s]).size >= 1


def test_include_numeric_transform_is_leak_safe_no_skew():
    """Replay bins raw floats through stored edges (searchsorted), NOT int-cast.

    The apply path output must equal an independent edge-based recomputation; an int-cast bug
    (vals.astype(int64)) would map e.g. 0.0..1.0 floats all to bin 0 and diverge.
    """
    df, y = _rotated_xor(3000, seed=2)
    mrmr = MRMR(
        cat_fe_config=CatFEConfig(enable=True, include_numeric=True, numeric_nbins=6),
        fe_max_steps=0,
        verbose=0,
    )
    mrmr.fit(df, y)
    crosses = _numeric_cross_recipes(mrmr)
    assert crosses
    r = crosses[0]
    name_a, name_b = r.src_names
    nbins_a, nbins_b = r.factorize_nbins
    edges = r.extra["src_bin_edges"]
    lookup = r.extra["lookup_table"]

    # Independent edge-based recomputation on a disjoint test draw.
    df_te, _ = _rotated_xor(1500, seed=99)
    ca = np.clip(np.searchsorted(np.asarray(edges[name_a], float), df_te[name_a].to_numpy(float), side="right"), 0, nbins_a - 1)
    cb = np.clip(np.searchsorted(np.asarray(edges[name_b], float), df_te[name_b].to_numpy(float), side="right"), 0, nbins_b - 1)
    expected = lookup[ca + cb * nbins_a]

    got = apply_recipe(r, df_te)
    np.testing.assert_array_equal(np.asarray(got).ravel(), np.asarray(expected).ravel())
    # Sanity: the raw floats are non-integer, so a naive int-cast would collapse them to one bin --
    # i.e. the edge path is doing real work (more than one distinct code).
    assert len(np.unique(got)) > 1


def test_include_numeric_skips_nan_bearing_numeric_columns():
    """A NaN-bearing numeric column must not seed an (unreplayable) numeric cross in v1."""
    df, y = _rotated_xor(3000, seed=3)
    df = df.copy()
    df.loc[df.index[:50], "x0"] = np.nan  # inject NaN into a signal column
    mrmr = MRMR(
        cat_fe_config=CatFEConfig(enable=True, include_numeric=True, numeric_nbins=8),
        fe_max_steps=0,
        verbose=0,
    )
    mrmr.fit(df, y)
    for r in _numeric_cross_recipes(mrmr):
        assert "x0" not in r.src_names, "NaN-bearing x0 must be excluded from include_numeric crosses"
    # Transform must not raise and must be finite-or-handled (no crash on the NaN column).
    out = mrmr.transform(df)
    assert out is not None


def test_include_numeric_off_no_numeric_cross():
    """Gate: a pure-numeric frame with include_numeric=False produces no numeric cross."""
    df, y = _rotated_xor(3000, seed=4)
    mrmr = MRMR(
        cat_fe_config=CatFEConfig(enable=True, include_numeric=False),
        fe_max_steps=0,
        verbose=0,
    )
    mrmr.fit(df, y)
    assert not _numeric_cross_recipes(mrmr)


def test_biz_value_include_numeric_standalone_lift_at_step_level():
    """Standalone business value, measured at the cat-FE STEP level (isolates the mechanism from the
    rest of MRMR's many pre-FE families, which independently recover such interactions in the full fit).

    On a ROTATED interaction -- where ``mul(a,b)`` is useless -- the include_numeric numeric cross, fed to
    a held-out LogisticRegression alongside the raw columns, must beat the same model given raw + mul(a,b).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    from mlframe.feature_selection.filters.discretization import categorize_dataset
    from mlframe.feature_selection.filters.info_theory import merge_vars
    from mlframe.feature_selection.filters.cat_interactions import run_cat_interaction_step

    dtype = np.int32
    deltas = []
    for seed in (10, 11, 12):
        df, y = _rotated_xor(6000, seed=seed)
        cut = 3000
        Xtr, Xte = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
        ytr, yte = y[:cut], y[cut:]

        # Discretize train with the target appended (mirrors MRMR.fit's categorize call).
        Xtr_t = Xtr.copy()
        Xtr_t["__t0"] = ytr
        data, cols, nbins = categorize_dataset(df=Xtr_t, nbins_strategy=None, n_bins=10, dtype=dtype)
        tgt_idx = np.array([cols.index("__t0")], dtype=np.int64)
        classes_y, freqs_y, _ = merge_vars(factors_data=data, vars_indices=tgt_idx, var_is_nominal=None, factors_nbins=nbins, dtype=dtype)
        num_raw = {cols.index("x0"): Xtr["x0"].to_numpy(float), cols.index("x1"): Xtr["x1"].to_numpy(float)}
        cfg = CatFEConfig(enable=True, include_numeric=True, numeric_nbins=8, emit_target_encoding=False, full_npermutations=0)
        d2, c2, nb2, state = run_cat_interaction_step(
            data=data,
            cols=cols,
            nbins=nbins,
            target_indices=tgt_idx,
            classes_y=classes_y,
            classes_y_safe=classes_y.copy(),
            freqs_y=freqs_y,
            categorical_vars=[],
            cfg=cfg,
            numeric_raw_values=num_raw,
            dtype=dtype,
        )
        crosses = [r for r in state.recipes if r.kind == "factorize" and r.extra.get("src_bin_edges") and set(r.src_names) == {"x0", "x1"}]
        assert crosses, f"seed {seed}: include_numeric must build the x0__x1 cross"
        r = crosses[0]

        # The cross is an ORDINAL cell code (cell 5 vs 50 is arbitrary) -- a linear model consumes it
        # one-hot'd, exactly as a categorical engineered feature is fed downstream. (MRMR's own MI screening
        # is label-invariant and sees the signal directly; a linear downstream needs the one-hot / TE view.)
        code_tr = apply_recipe(r, Xtr).astype(np.int64).ravel()
        code_te = apply_recipe(r, Xte).astype(np.int64).ravel()
        n_cells = int(max(code_tr.max(), code_te.max())) + 1
        cross_tr = np.eye(n_cells, dtype=float)[np.clip(code_tr, 0, n_cells - 1)]
        cross_te = np.eye(n_cells, dtype=float)[np.clip(code_te, 0, n_cells - 1)]
        mul_tr = (Xtr["x0"].to_numpy() * Xtr["x1"].to_numpy()).reshape(-1, 1)
        mul_te = (Xte["x0"].to_numpy() * Xte["x1"].to_numpy()).reshape(-1, 1)
        raw_tr, raw_te = Xtr.to_numpy(), Xte.to_numpy()

        def _auc(a_tr, a_te):
            sc = StandardScaler()
            clf = LogisticRegression(max_iter=2000).fit(sc.fit_transform(a_tr), ytr)
            return roc_auc_score(yte, clf.predict_proba(sc.transform(a_te))[:, 1])

        auc_cross = _auc(np.hstack([raw_tr, cross_tr]), np.hstack([raw_te, cross_te]))
        auc_mul = _auc(np.hstack([raw_tr, mul_tr]), np.hstack([raw_te, mul_te]))
        deltas.append(auc_cross - auc_mul)
    mean_delta = float(np.mean(deltas))
    assert mean_delta > 0.10, f"include_numeric cross must beat mul(a,b) on a rotated interaction: mean delta={mean_delta:.3f} ({deltas})"
