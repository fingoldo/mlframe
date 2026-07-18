"""Wave 11 (Category 3) M4: the cross-stage engineered-column dedup in ``_fit_impl_core.py`` (~line 4700)
called ``np.corrcoef`` one pair at a time over up to ~200 engineered columns (O(K^2)). Fixed via
``_eng_dedup_batch_corr.one_vs_many_abs_corr_masked``: an APPEND-ONLY rank buffer (never re-copied) plus a
boolean active mask, so a fully-finite candidate's comparisons against every fully-finite kept column batch
in one parallel call. (A first-cut design that re-``np.vstack``ed the current kept set on every candidate
was measured as a NET LOSS -- O(K^2 * n) memcpy, same order as the corrcoef calls it replaced -- and was
replaced by this append-only design; see the module docstring.)

This pins the batched kernel against a brute-force one-vs-many ``np.corrcoef`` reference, and separately
pins the full dedup ALGORITHM (old per-pair loop vs. new masked-buffer integration) via a standalone
reproduction of both, matching the real ``_fit_impl_core.py`` block line for line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._eng_dedup_batch_corr import one_vs_many_abs_corr_masked


def _brute_force_one_vs_many(a, buf, active):
    """Brute force one vs many."""
    out = np.zeros(buf.shape[0], dtype=np.float64)
    for j in range(buf.shape[0]):
        if not active[j]:
            continue
        b = buf[j]
        if a.std() <= 1e-12 or b.std() <= 1e-12:
            continue
        r = np.corrcoef(a, b)[0, 1]
        out[j] = abs(float(r)) if np.isfinite(r) else 0.0
    return out


def test_one_vs_many_abs_corr_masked_matches_brute_force():
    """One vs many abs corr masked matches brute force."""
    rng = np.random.default_rng(0)
    n, k = 4000, 30
    buf = rng.normal(size=(k, n))
    a = buf[0] * 0.9 + rng.normal(scale=0.05, size=n)  # correlated with row 0
    active = rng.random(k) < 0.7
    active[0] = True

    expected = _brute_force_one_vs_many(a, buf, active)
    got = one_vs_many_abs_corr_masked(a, buf, active)
    assert np.allclose(expected, got, atol=1e-9)
    # inactive rows must be untouched (stay 0.0 regardless of their content)
    assert np.all(got[~active] == 0.0)


def test_one_vs_many_abs_corr_masked_empty_buffer_returns_empty():
    """One vs many abs corr masked empty buffer returns empty."""
    a = np.zeros(10)
    buf = np.empty((0, 10))
    active = np.empty(0, dtype=bool)
    out = one_vs_many_abs_corr_masked(a, buf, active)
    assert out.shape == (0,)


def _eng_dedup_prefer(cand, kept, mig_set, eng_mi):
    """Eng dedup prefer."""
    cand_mig = cand in mig_set
    kept_mig = kept in mig_set
    if cand_mig == kept_mig:
        return False
    mi_cand = eng_mi.get(cand)
    mi_kept = eng_mi.get(kept)
    if mi_cand is None or mi_kept is None:
        return False
    if mi_cand > mi_kept + 1e-12:
        return True
    if mi_cand >= mi_kept - 1e-12:
        return cand_mig and not kept_mig
    return False


def _old_dedup(X, eng_cols_appended, mig_set, eng_mi):
    """Frozen copy of the pre-fix O(K^2) per-pair np.corrcoef dedup loop."""
    eng_keep, eng_drop, eng_arrs, eng_ranks = [], set(), {}, {}
    for c in eng_cols_appended:
        if c in eng_drop:
            continue
        arr_c = np.asarray(X[c].to_numpy(), dtype=np.float64)
        fin_c = np.isfinite(arr_c)
        if not fin_c.any() or arr_c[fin_c].std() <= 1e-12:
            eng_keep.append(c)
            eng_arrs[c] = arr_c
            continue
        ranks_c = pd.Series(arr_c).rank(method="average").to_numpy()
        eng_ranks[c] = ranks_c
        colliding = []
        for kc in eng_keep:
            arr_k = eng_arrs[kc]
            mask = fin_c & np.isfinite(arr_k)
            if mask.sum() < 8:
                continue
            a, b = arr_c[mask], arr_k[mask]
            if a.std() <= 1e-12 or b.std() <= 1e-12:
                continue
            if bool(mask.all()):
                ra, rb = ranks_c, eng_ranks.get(kc)
                if rb is None:
                    rb = pd.Series(arr_k).rank(method="average").to_numpy()
                    eng_ranks[kc] = rb
            else:
                ra = pd.Series(a).rank(method="average").to_numpy()
                rb = pd.Series(b).rank(method="average").to_numpy()
            if ra.std() <= 1e-12 or rb.std() <= 1e-12:
                continue
            rho = abs(float(np.corrcoef(ra, rb)[0, 1]))
            if np.isfinite(rho) and rho >= 0.99:
                colliding.append(kc)
        if colliding:
            loses = any(not _eng_dedup_prefer(c, kc, mig_set, eng_mi) for kc in colliding)
            if loses:
                eng_drop.add(c)
            else:
                for kc in colliding:
                    eng_drop.add(kc)
                    eng_keep.remove(kc)
                    eng_arrs.pop(kc, None)
                eng_keep.append(c)
                eng_arrs[c] = arr_c
        else:
            eng_keep.append(c)
            eng_arrs[c] = arr_c
    return eng_keep, eng_drop


def _new_dedup(X, eng_cols_appended, mig_set, eng_mi):
    """Mirrors the current ``_fit_impl_core.py`` masked-buffer integration."""
    eng_keep, eng_drop, eng_arrs, eng_ranks = [], set(), {}, {}
    fully_finite: dict = {}
    n = len(X)
    rank_buf = np.empty((len(eng_cols_appended), n), dtype=np.float64)
    row_of: dict = {}
    next_row = 0
    for c in eng_cols_appended:
        if c in eng_drop:
            continue
        arr_c = np.asarray(X[c].to_numpy(), dtype=np.float64)
        fin_c = np.isfinite(arr_c)
        fully_finite[c] = bool(fin_c.all())
        if not fin_c.any() or arr_c[fin_c].std() <= 1e-12:
            eng_keep.append(c)
            eng_arrs[c] = arr_c
            continue
        ranks_c = pd.Series(arr_c).rank(method="average").to_numpy()
        eng_ranks[c] = ranks_c
        colliding = []
        fast_set = set()
        if fully_finite[c] and arr_c.shape[0] >= 8 and next_row > 0:
            active = np.zeros(next_row, dtype=np.bool_)
            row_to_kc = {}
            for kc in eng_keep:
                r = row_of.get(kc)
                if r is not None:
                    active[r] = True
                    row_to_kc[r] = kc
            if active.any():
                corrs = one_vs_many_abs_corr_masked(ranks_c, rank_buf[:next_row], active)
                for r, kc in row_to_kc.items():
                    fast_set.add(kc)
                    if corrs[r] >= 0.99:
                        colliding.append(kc)
        for kc in eng_keep:
            if kc in fast_set:
                continue
            arr_k = eng_arrs[kc]
            mask = fin_c & np.isfinite(arr_k)
            if mask.sum() < 8:
                continue
            a, b = arr_c[mask], arr_k[mask]
            if a.std() <= 1e-12 or b.std() <= 1e-12:
                continue
            if bool(mask.all()):
                ra, rb = ranks_c, eng_ranks.get(kc)
                if rb is None:
                    rb = pd.Series(arr_k).rank(method="average").to_numpy()
                    eng_ranks[kc] = rb
            else:
                ra = pd.Series(a).rank(method="average").to_numpy()
                rb = pd.Series(b).rank(method="average").to_numpy()
            if ra.std() <= 1e-12 or rb.std() <= 1e-12:
                continue
            rho = abs(float(np.corrcoef(ra, rb)[0, 1]))
            if np.isfinite(rho) and rho >= 0.99:
                colliding.append(kc)
        if colliding:
            loses = any(not _eng_dedup_prefer(c, kc, mig_set, eng_mi) for kc in colliding)
            if loses:
                eng_drop.add(c)
            else:
                for kc in colliding:
                    eng_drop.add(kc)
                    eng_keep.remove(kc)
                    eng_arrs.pop(kc, None)
                eng_keep.append(c)
                eng_arrs[c] = arr_c
                if fully_finite[c]:
                    rank_buf[next_row] = ranks_c
                    row_of[c] = next_row
                    next_row += 1
        else:
            eng_keep.append(c)
            eng_arrs[c] = arr_c
            if fully_finite[c]:
                rank_buf[next_row] = ranks_c
                row_of[c] = next_row
                next_row += 1
    return eng_keep, eng_drop


def _make_scenario(seed, n=2000, k=30, nan_frac=0.0):
    """Make scenario."""
    rng = np.random.default_rng(seed)
    cols = {}
    base_pool = [rng.normal(size=n) for _ in range(max(3, k // 5))]
    for i in range(k):
        r = rng.random()
        if r < 0.05:
            v = np.full(n, float(i))
        elif r < 0.35:
            base = base_pool[i % len(base_pool)]
            kind = i % 3
            v = base**3 + rng.normal(scale=1e-3, size=n) if kind == 0 else (-np.abs(base) + rng.normal(scale=1e-3, size=n) if kind == 1 else base.copy())
        else:
            v = rng.normal(size=n)
        cols[f"e{i}"] = v
    df = pd.DataFrame(cols)
    if nan_frac > 0:
        mask = rng.random(df.shape) < nan_frac
        df = df.mask(mask)
    names = list(cols.keys())
    mig_set = set(rng.choice(names, size=max(1, k // 6), replace=False))
    eng_mi = {n_: float(rng.random()) for n_ in names if rng.random() < 0.8}
    return df, names, mig_set, eng_mi


def test_eng_dedup_algorithm_matches_reference_across_random_configs():
    """Eng dedup algorithm matches reference across random configs."""
    n_checks = 0
    for seed in range(15):
        for nan_frac in (0.0, 0.05, 0.2):
            df, names, mig_set, eng_mi = _make_scenario(seed, nan_frac=nan_frac)
            keep_old, drop_old = _old_dedup(df, names, mig_set, eng_mi)
            keep_new, drop_new = _new_dedup(df, names, mig_set, eng_mi)
            n_checks += 1
            assert set(keep_old) == set(keep_new), f"seed={seed} nan_frac={nan_frac}"
            assert drop_old == drop_new, f"seed={seed} nan_frac={nan_frac}"
    assert n_checks == 45
