"""The batched masked-Spearman kernel of the engineered-duplicate scan picks the same colliding kept columns as the one-pair-at-a-time ranks."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_scan import _REL_TOL, _pair_collisions


def _reference(eng_keep, fast, arr_c, fin_c, eng_arrs):
    """The former per-pair loop for pairs that are not fully finite: pandas average ranks over the jointly finite rows, Pearson of the ranks."""
    out = []
    for kept in eng_keep:
        if kept in fast:
            continue
        b_full = eng_arrs[kept]
        mask = fin_c & np.isfinite(b_full)
        if mask.sum() < 8:
            continue
        a, b = arr_c[mask], b_full[mask]
        if a.std() <= _REL_TOL * float(np.abs(a).max()) or b.std() <= _REL_TOL * float(np.abs(b).max()):
            continue
        ra = pd.Series(a).rank(method="average").to_numpy()
        rb = pd.Series(b).rank(method="average").to_numpy()
        if ra.std() <= 1e-12 or rb.std() <= 1e-12:
            continue
        rho = abs(float(np.corrcoef(ra, rb)[0, 1]))
        if np.isfinite(rho) and rho >= 0.99:
            out.append(kept)
    return out


def test_masked_pairs_match_the_pandas_rank_reference():
    """Monotone twins with NaN rows collide, independent and constant columns do not, ties included, in the original order."""
    rng = np.random.default_rng(0)
    n = 3000
    base = rng.normal(size=n)
    cols = {
        "twin_exp": np.exp(base),
        "twin_cube": base**3,
        "indep": rng.normal(size=n),
        "rounded_twin": np.round(base, 1),
        "const": np.full(n, 2.0),
        "anti": -base,
    }
    cols["twin_exp"][rng.random(n) < 0.05] = np.nan
    cols["rounded_twin"][rng.random(n) < 0.1] = np.nan
    cand = base.astype(np.float32)
    cand[rng.random(n) < 0.03] = np.nan
    arrs = {k: v.astype(np.float32) for k, v in cols.items()}
    keep = list(arrs)
    fin_c = np.isfinite(cand)
    finite_flags = {k: bool(np.isfinite(v).all()) for k, v in arrs.items()}
    got: list = []
    _pair_collisions(got, keep, set(), cand, fin_c, None, arrs, {}, finite_flags)
    ref = _reference([k for k in keep if not (finite_flags[k] and fin_c.all())], set(), cand, fin_c, arrs)
    assert got == ref
    assert "twin_exp" in got and "indep" not in got and "const" not in got
