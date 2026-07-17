"""REAL, hard, end-to-end MRMR test at 100,000 rows x 10,000 columns.

WHY THIS EXISTS
---------------
Every other MRMR/FE test in the suite runs on TOY frames (hundreds to a few
thousand rows, dozens of columns). The user explicitly asked whether MRMR
actually WORKS at the scale they care about: 100k rows x 10k columns. A toy
test cannot answer that -- it cannot show that MRMR (a) COMPLETES within a
realistic wall-time budget at this width, (b) stays parsimonious (selects
<< 10k features) rather than dumping the whole frame, (c) RECOVERS the handful
of genuinely informative columns out of 10k noise columns far above chance,
(d) picks ONE representative of a redundant cluster rather than all copies,
and (e) actually HELPS a downstream model on held-out data vs an equal-count
random feature subset (the real "did feature selection do anything" test).

This module builds a realistically-structured SYNTHETIC frame with KNOWN
ground truth (so recall/parsimony are checkable) and runs MRMR end-to-end on
it. A genuinely real public wide dataset at 100k x 10k is not offline-
importable, so the structured synthetic with planted truth IS the deliverable
(stated honestly here). The frame mixes signal types specifically to stress
MRMR: several linear main effects of varying strength; pure pair interactions
(sign-products / XOR-like) with ~0 marginal correlation and only mild leakage;
a redundant CLUSTER (one true feature copied with noise several times -- MRMR
must pick ONE, not all); heavy-tailed columns; and low-cardinality integer
"categorical-like" columns. The rest are i.i.d. Gaussian noise.

MEMORY DISCIPLINE
-----------------
A 100k x 10k float32 frame is ~4 GB. It is generated CHUNKED into an on-disk
``np.memmap`` and the pandas frame is a thin view over that memmap, so the data
is never held twice. The memmap file lives under a tmp dir and is removed on
teardown.

SLOW / CI KNOB
--------------
This test is SLOW (the full config is the headline; minutes of wall-time).
Set ``MLFRAME_REAL100K_FAST=1`` to shrink to ~20k x 3k for CI -- it still
exercises every assertion path. To run the full headline config explicitly,
either leave the env var unset/0, or set ``MLFRAME_REAL100K_FULL=1`` to force
it even when a CI runner exports FAST.

SCALE / BUDGET NOTE (honest back-off)
-------------------------------------
A clean probe on the dev box (concurrent large-memmap load, interactions_max_order=1)
measured PURE MRMR at 8,000 rows x 10,000 cols = ~1,170 s (~19.5 min), and it
recovered ALL four planted linear effects with n_selected=4 (perfectly
parsimonious). The per-feature relevance+permutation screen over 10k columns is
~linear in rows, so a FULL 100,000-row x 10,000-col fit would take HOURS -- past
any reasonable budget. The test therefore keeps the FULL 10,000-COLUMN WIDTH (the
dimension the user asked about; the one that stresses needle-in-10k recall +
redundancy dedup), GENERATES the full 100,000-row frame on disk as a memmap
(proving the real-scale memory path), and FITS MRMR on a budget-sized ROW
SUBSAMPLE (~6,000 rows) of the full-width TRAIN frame -- the "largest config that
completes" the task explicitly allows. The downstream model + held-out evaluation
use the FULL train/test rows. Override the fit-row cap with
MLFRAME_REAL100K_FIT_ROWS=N to push toward the full 100k headline on a quiet box.

MEASURED RESULTS:
  -- see the ``measured`` dict printed at the end of the test run (fit wall-time,
     peak RSS delta, base-feature recall, parsimony, redundant-cluster members
     kept, and selected-vs-random downstream AUC delta), emitted to stdout via -s.
     The values are machine-dependent; the ASSERTIONS below are the portable
     contract.
"""

from __future__ import annotations

import gc
import os
import time

import numpy as np
import pandas as pd
import pytest

from tests.conftest import fast_n_estimators

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# size config
# ---------------------------------------------------------------------------
def _resolve_size() -> tuple[int, int, str]:
    """Return (n_rows, n_cols, mode). FAST shrinks to a CI-friendly size that
    still hits every assertion; FULL is the 100k x 10k headline."""
    force_full = os.environ.get("MLFRAME_REAL100K_FULL", "").strip() in ("1", "true", "True")
    fast = os.environ.get("MLFRAME_REAL100K_FAST", "").strip() in ("1", "true", "True")
    if fast and not force_full:
        return 20_000, 2_000, "fast"
    return 100_000, 10_000, "full"


def _resolve_fit_rows(mode: str, n_train: int) -> int:
    """Row count fed to MRMR.fit, sized to keep the fit inside the wall-time
    budget at full column width (see the rationale in the fit block). FAST mode
    fits all train rows (it is already small); FULL mode caps the fit rows so the
    10k-column screen completes reliably. Override with MLFRAME_REAL100K_FIT_ROWS.
    """
    env = os.environ.get("MLFRAME_REAL100K_FIT_ROWS", "").strip()
    if env.isdigit() and int(env) > 0:
        return min(int(env), n_train)
    if mode == "fast":
        return min(2_500, n_train)
    # FULL: ~6,000 rows x 10k cols completes in ~15 min on the dev box.
    return min(6_000, n_train)


# planted-signal layout. Indices are assigned into the wide frame; the rest are
# pure Gaussian noise. Kept well under the column count so the "needle in 10k"
# property holds.
SEED = 20260619


def _plant_layout(n_cols: int) -> dict:
    """Deterministic placement of the informative columns inside the wide frame.

    Returns a dict describing every planted group so the test can assert recall
    against EXACT ground-truth indices.
    """
    rng = np.random.default_rng(SEED + 999)
    # reserve a generous pool of distinct slots, then assign roles.
    n_slots = 40
    slots = rng.choice(n_cols, size=n_slots, replace=False)
    slots = [int(s) for s in slots]
    it = iter(slots)

    # several linear main effects of varying strength.
    linear = {next(it): w for w in (2.2, 1.7, 1.3, 1.0, 0.8, 0.6)}
    # pure pair interactions (sign products), ~0 marginal, mild leakage.
    pair_a1, pair_b1 = next(it), next(it)
    pair_a2, pair_b2 = next(it), next(it)
    pairs = [(pair_a1, pair_b1), (pair_a2, pair_b2)]
    # redundant cluster: one true driver copied with noise into K slots.
    cluster_driver = next(it)
    cluster_copies = [next(it) for _ in range(4)]  # 4 noisy copies of the driver
    # heavy-tailed informative columns (Student-t main effects).
    heavy = {next(it): w for w in (1.4, 1.0)}
    # low-cardinality integer "categorical-like" informative columns.
    cats = {next(it): w for w in (1.2, 0.9)}

    informative_base = set(linear) | {pair_a1, pair_b1, pair_a2, pair_b2} | {cluster_driver} | set(heavy) | set(cats)
    return dict(
        linear=linear,
        pairs=pairs,
        cluster_driver=cluster_driver,
        cluster_copies=cluster_copies,
        cluster_all=[cluster_driver, *cluster_copies],
        heavy=heavy,
        cats=cats,
        informative_base=sorted(informative_base),  # cluster copies excluded (they're redundant)
        all_planted=sorted(informative_base | set(cluster_copies)),
    )


# ---------------------------------------------------------------------------
# chunked memmap generation -- never hold the frame twice
# ---------------------------------------------------------------------------
def _build_memmap_frame(n_rows: int, n_cols: int, layout: dict, task: str, tmp_path):
    """Generate the wide frame CHUNKED into an on-disk float32 memmap and build
    a continuous-signal vector ``sig`` from the planted columns.

    Returns (df, y, mmap_path). ``df`` is a pandas frame backed by the memmap.
    """
    mm_path = os.path.join(str(tmp_path), "real100k_X.f32")
    X = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=(n_rows, n_cols))

    rng = np.random.default_rng(SEED)
    # Row-chunk sized so the transient float32 block stays small (~24 MB at
    # n_cols=10k), keeping peak RAM modest even when another process holds a
    # large memmap. The block is generated DIRECTLY as float32 (no float64
    # intermediate) via Generator.standard_normal(..., dtype=np.float32).
    chunk = max(500, min(2_000, 24_000_000 // max(1, n_cols)))
    # accumulate the continuous signal incrementally (one float64 vector, n_rows).
    sig = np.zeros(n_rows, dtype=np.float64)

    cat_cols = set(layout["cats"])
    heavy_cols = set(layout["heavy"])
    cluster_driver = layout["cluster_driver"]
    cluster_copies = set(layout["cluster_copies"])

    start = 0
    while start < n_rows:
        stop = min(start + chunk, n_rows)
        m = stop - start
        # base: i.i.d. standard normal noise for the whole block, float32 directly.
        block = rng.standard_normal((m, n_cols), dtype=np.float32)

        # heavy-tailed informative columns: overwrite with Student-t draws.
        for j in heavy_cols:
            block[:, j] = rng.standard_t(3, m).astype(np.float32)
        # low-card integer "categorical" informative columns: small integer set.
        for j in cat_cols:
            block[:, j] = rng.integers(0, 5, m).astype(np.float32)
        # cluster driver is a clean normal; copies = driver + small noise.
        drv = rng.standard_normal(m, dtype=np.float32)
        block[:, cluster_driver] = drv
        for j in cluster_copies:
            block[:, j] = drv + (0.15 * rng.standard_normal(m, dtype=np.float32))

        X[start:stop, :] = block

        # ---- build the signal contribution from this block ----
        s = np.zeros(m, dtype=np.float64)
        for j, w in layout["linear"].items():
            s += w * block[:, j]
        for ja, jb in layout["pairs"]:
            # pure sign-product interaction: ~0 marginal, needs both columns.
            s += 1.6 * np.sign(block[:, ja]) * np.sign(block[:, jb])
            # mild leakage L~0.1 so a marginal screen has a faint trail.
            s += 0.1 * block[:, ja]
        # cluster driver drives y (so all copies are relevant but redundant).
        s += 1.5 * block[:, cluster_driver]
        for j, w in layout["heavy"].items():
            # bounded transform of the heavy column so a few outliers don't dominate.
            s += w * np.tanh(block[:, j])
        for j, w in layout["cats"].items():
            s += w * block[:, j]
        sig[start:stop] = s
        del block
        start = stop

    X.flush()
    cols = [f"c{i}" for i in range(n_cols)]
    df = pd.DataFrame(X, columns=cols, copy=False)

    sig += 0.5 * rng.standard_normal(n_rows)  # observation noise
    if task == "classification":
        y = pd.Series((sig > np.median(sig)).astype(np.int8), name="y")
    else:
        y = pd.Series(sig.astype(np.float32), name="y")
    return df, y, mm_path, X


def _peak_rss_mb() -> float:
    """Current process RSS in MB via psutil, or NaN if psutil is unavailable (never fatal to the test)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return float("nan")


def _downstream_auc(X_tr, y_tr, X_te, y_te, cols, seed: int) -> float:
    """Train LightGBM + logistic on the given column subset, return the better
    held-out AUC (the union of the two model families -- the value a downstream
    user would actually realise)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xtr = np.asarray(X_tr[cols], dtype=np.float32)
    Xte = np.asarray(X_te[cols], dtype=np.float32)

    aucs = []
    try:
        import lightgbm as lgb

        gbm = lgb.LGBMClassifier(
            n_estimators=fast_n_estimators(120, fast=80),
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=2,
            verbose=-1,
        )
        gbm.fit(Xtr, np.asarray(y_tr))
        aucs.append(roc_auc_score(np.asarray(y_te), gbm.predict_proba(Xte)[:, 1]))
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        pass
    try:
        sc = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr)
        Xte_s = sc.transform(Xte)
        lr = LogisticRegression(max_iter=500, C=1.0)
        lr.fit(Xtr_s, np.asarray(y_tr))
        aucs.append(roc_auc_score(np.asarray(y_te), lr.predict_proba(Xte_s)[:, 1]))
    except Exception:  # nosec B110 -- best-effort cleanup/optional step; failure here never masks this test's own assertions
        pass
    return float(max(aucs)) if aucs else float("nan")


@pytest.mark.slow
@pytest.mark.timeout(0)  # this test enforces its OWN wall budget (max_runtime_mins); the global per-test timeout would kill the real-scale fit
def test_mrmr_real_100k(tmp_path):
    """End-to-end MRMR at real scale: completes in budget, recovers planted
    signal parsimoniously, dedups the redundant cluster, and beats a random
    equal-count feature subset on held-out AUC."""
    from mlframe.feature_selection.filters import MRMR

    n_rows, n_cols, mode = _resolve_size()
    task = "classification"
    layout = _plant_layout(n_cols)

    # Wall-time budget for the MRMR fit. ``max_runtime_mins`` is a SOFT budget:
    # MRMR finishes the in-flight relevance/confirmation pass before honouring
    # it, so the observed wall-time can run somewhat past the nominal value (the
    # budget assertion below allows headroom for this). FAST uses a small budget
    # for CI; FULL uses the headline budget.
    budget_mins = 4.0 if mode == "fast" else 18.0
    rss0 = _peak_rss_mb()

    df, y, _mm_path, X = _build_memmap_frame(n_rows, n_cols, layout, task, tmp_path)
    rss_after_build = _peak_rss_mb()

    # train / test split (held-out test the model never saw).
    rng = np.random.default_rng(SEED + 1)
    perm = rng.permutation(n_rows)
    n_te = n_rows // 4
    te_idx = np.sort(perm[:n_te])
    tr_idx = np.sort(perm[n_te:])
    df_tr, y_tr = df.iloc[tr_idx], y.iloc[tr_idx]
    df_te, y_te = df.iloc[te_idx], y.iloc[te_idx]

    # ---- choose the MRMR fit-row count to fit the budget ----
    # MEASURED on the dev box (concurrent large-memmap load), interactions_max_order=1:
    #   8,000 rows x 10,000 cols pure MRMR fit = ~1,170 s (~19.5 min).
    # The per-feature relevance+permutation screen over 10k columns is ~linear in
    # rows, so the FULL 100k x 10k fit would be ~12x that (hours) -- far past any
    # sane CI/interactive budget. Per the test design, we keep the FULL 10k-column
    # WIDTH (the dimension the user asked about and the one that stresses MRMR's
    # needle-in-10k recall + redundancy dedup), GENERATE the full 100k-row frame on
    # disk (proving the memory path works at real scale), and FIT MRMR on a
    # budget-sized ROW SUBSAMPLE of that full-width TRAIN frame. The selection is
    # rank-stable under row subsampling at this signal strength (the probe recovered
    # all planted linear effects at 8k rows), and the DOWNSTREAM model + held-out
    # evaluation still use the FULL train/test rows. This is the "largest config
    # that completes" the task explicitly allows; the back-off is documented here.
    fit_rows = _resolve_fit_rows(mode, len(tr_idx))
    if fit_rows < len(tr_idx):
        rng_fit = np.random.default_rng(SEED + 3)
        sub = np.sort(rng_fit.choice(len(tr_idx), size=fit_rows, replace=False))
        df_fit, y_fit = df_tr.iloc[sub], y_tr.iloc[sub]
    else:
        df_fit, y_fit = df_tr, y_tr

    # interactions_max_order=1: at 10k columns the O(p^2) FE grid sweep is the
    # dominant cost and is not the contract under test here (pure relevance +
    # conditional-MI redundancy selection is). The pair-interaction columns are
    # still planted with mild marginal leakage so the relevance gate can pick at
    # least one operand of each pair.
    mrmr_kwargs = dict(
        verbose=0,
        random_seed=42,
        max_runtime_mins=budget_mins,
        interactions_max_order=1,
        n_workers=1,
    )
    if mode == "fast":
        # CI speed: the per-feature permutation-confidence confirmation over
        # thousands of columns dominates wall-time. Shrink the permutation
        # counts in FAST mode -- selection identity is stable at this signal
        # strength, and the assertions (parsimony, recall, dedup, AUC delta)
        # still hold. The FULL headline keeps the default confirmation depth.
        mrmr_kwargs.update(full_npermutations=1, baseline_npermutations=1)
    sel = MRMR(**mrmr_kwargs)
    t0 = time.time()
    sel.fit(df_fit, y_fit)
    fit_s = time.time() - t0
    rss_peak = _peak_rss_mb()

    selected_idx = [int(i) for i in np.asarray(sel.support_)]
    selected_names = [str(sel.feature_names_in_[i]) for i in selected_idx]
    n_sel = len(selected_idx)

    # ---------------- ASSERTIONS ----------------
    # 1) COMPLETES in BOUNDED time. ``max_runtime_mins`` is a SOFT budget -- MRMR
    #    finishes the in-flight relevance pass before stopping, so under load the
    #    wall-time can run up to ~2x the nominal budget (a probe on a 10-min
    #    budget finished at ~19.5 min). The contract this guards is "bounded,
    #    not hours": allow 2.5x the budget plus a fixed floor. The real point is
    #    that it RETURNS a usable selection rather than running unbounded at 10k
    #    columns.
    max_allowed_s = budget_mins * 60 * 2.5 + 120
    assert fit_s <= max_allowed_s, f"MRMR fit took {fit_s:.0f}s, exceeding bounded budget {max_allowed_s:.0f}s"
    # 2) non-empty and PARSIMONIOUS (<< n_cols).
    assert n_sel >= 1, "MRMR selected nothing"
    assert n_sel <= max(50, n_cols // 100), f"MRMR not parsimonious: selected {n_sel} of {n_cols}"

    # 3) base-feature RECALL well above random baseline.
    informative = set(layout["informative_base"])
    sel_set = set(selected_idx)
    hits = informative & sel_set
    recall = len(hits) / len(informative)
    # random baseline: picking n_sel of n_cols, expected hits = n_sel*|inf|/n_cols.
    exp_random_hits = n_sel * len(informative) / n_cols
    assert len(hits) >= 3, f"recovered only {len(hits)} planted base features: {sorted(hits)}"
    assert len(hits) >= 5 * max(exp_random_hits, 1e-9), f"recall {len(hits)} not >> random expectation {exp_random_hits:.3f}"

    # 4) redundant cluster contributes AT MOST a small number (not all copies).
    cluster_all = set(layout["cluster_all"])  # driver + 4 copies = 5
    cluster_selected = cluster_all & sel_set
    assert len(cluster_selected) <= 2, (
        f"MRMR kept {len(cluster_selected)} of {len(cluster_all)} redundant cluster members (should dedup to <=2): {sorted(cluster_selected)}"
    )

    # 5) downstream model on MRMR-selected features beats an equal-count RANDOM
    #    subset on held-out AUC.
    auc_sel = _downstream_auc(df_tr, y_tr, df_te, y_te, selected_names, seed=7)
    rng_rand = np.random.default_rng(SEED + 2)
    rand_idx = rng_rand.choice(n_cols, size=n_sel, replace=False)
    rand_names = [f"c{i}" for i in rand_idx]
    auc_rand = _downstream_auc(df_tr, y_tr, df_te, y_te, rand_names, seed=7)

    assert np.isfinite(auc_sel), "downstream AUC on selected features is not finite"
    assert np.isfinite(auc_rand), "downstream AUC on random subset is not finite"
    assert auc_sel > auc_rand + 0.02, f"MRMR selection AUC {auc_sel:.4f} did not beat random-subset AUC {auc_rand:.4f} by a clear margin"
    # sanity: a real selection should give materially-better-than-chance AUC.
    assert auc_sel >= 0.60, f"selected-feature AUC {auc_sel:.4f} suspiciously low"

    measured = dict(
        mode=mode,
        n_rows=n_rows,
        n_cols=n_cols,
        fit_rows=len(df_fit),
        fit_seconds=round(fit_s, 1),
        rss_baseline_mb=round(rss0, 0),
        rss_after_build_mb=round(rss_after_build, 0),
        rss_peak_mb=round(rss_peak, 0),
        rss_delta_mb=round(rss_peak - rss0, 0),
        n_selected=n_sel,
        base_recall=round(recall, 3),
        base_hits=len(hits),
        n_informative_base=len(informative),
        cluster_members_selected=len(cluster_selected),
        auc_selected=round(auc_sel, 4),
        auc_random=round(auc_rand, 4),
        auc_delta=round(auc_sel - auc_rand, 4),
    )
    print("\n[MRMR REAL100K MEASURED] " + repr(measured))

    # free the memmap before teardown.
    del df, df_tr, df_te, X
    gc.collect()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_mrmr_real_100k(d)
