"""HybridSelector: a compute-once / share-many composition of the four mlframe selectors.

Design principles (both explicit in this class):
  1. COMPUTE-ONCE-SHARE-MANY. Three artifacts are computed a single time on the full X and then handed to the
     reused component selectors instead of each one recomputing them:
       - MI / SU / per-column bins: produced by MRMR (retain_artifacts=True) and injected into ShapProxiedFS via
         its ``precomputed=`` hook (skips ShapProxiedFS's own univariate pre-screen + re-binning).
       - permutation FEATURE IMPORTANCE: one honest held-out pass (mirrors RFECV's permutation importance). Drives
         the cheap relevance pre-screen that narrows X before the expensive wrappers, and breaks ties when emitting
         one representative per redundant cluster.
       - raw-correlation CLUSTERS: computed once; used both to pre-merge correlated columns before the BorutaShap
         gate (the measured R2b-6 win) and to de-duplicate the final selection.
  2. REUSE, DON'T REIMPLEMENT. The members are the real production selectors composed as whole objects
     (MRMR, ShapProxiedFS, BorutaShap); the hybrid only adds the shared-artifact plumbing and the combine rule.

Final selection is a CLUSTER-AWARE VOTE: a cluster is kept when >= ``vote`` members selected any of its features;
each kept cluster contributes its highest-permutation-FI member (parsimony), or all members when expand_clusters.

Adapter protocol matches fs_selectors.py: ``fit(X, y)`` / ``transform(X)`` / ``raw_selected_`` / ``n_engineered_``.
"""
from __future__ import annotations
import os
os.environ.setdefault("TQDM_DISABLE", "1")
import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from mlframe.utils.misc import rng_hygienic_fit


def _as_pandas_view(X):
    """Bridge a polars frame to an Arrow-backed pandas VIEW (zero-copy for numeric/bool/string columns) so HybridSelector's pandas-only glue
    (corr_clusters on ``X.values``, ``X.iloc`` / ``X[col]`` member feeds, ``pd.concat`` augment) operates on a real pandas frame. The internal
    members (LightGBM FI, ShapProxiedFS, BorutaShap) want pandas / ndarray anyway and would each convert independently; bridging once at the fit
    boundary captures the real column names from the polars schema and is NOT a full materialisation (numeric columns stay zero-copy Arrow views).
    Non-polars inputs pass through untouched."""
    try:
        import polars as pl
    except ImportError:
        return X
    if not isinstance(X, pl.DataFrame):
        return X
    from mlframe.training.utils import get_pandas_view_of_polars_df
    return get_pandas_view_of_polars_df(X)


# Tree-member feature-engineering operators applied per co-occurrence pair (a, b). "mul" is the product; the rich
# operators (absdiff/signed/ratio) recover NON-product interaction signal a bilinear term cannot linearize (measured:
# |a-b| logit 0.49->0.88; sign(a)|b| 0.79->0.88; products+rich on madelon +0.020 3-seed, variance halved). All are
# leak-free pure functions of X (no y, no fit) so they replay exactly at transform time. ratio uses +1 eps; outputs
# are nan/inf-sanitised. Each op-column is synergy-gated independently (kept only if more informative than both operands).
_TREE_OPS = {
    "mul": lambda a, b: a * b,
    "absd": lambda a, b: np.abs(a - b),
    "sign": lambda a, b: np.sign(a) * np.abs(b),
    "rat": lambda a, b: a / (np.abs(b) + 1.0),
}


# corr_block_threshold: above this many columns, corr_clusters switches from the full p x p correlation matrix to a
# row-block path that never materializes the dense p x p (the wide-data memory blocker: p=10k -> ~800 MB float64).
# The blocked path standardizes columns once (z-scores) and computes each rep's above-threshold |corr| neighbours by a
# single (1 x p) row-block GEMV against the standardized matrix, so peak extra memory is O(n*p + p) not O(p^2). The
# greedy assignment order is byte-identical to the full-matrix path (same rep order = first unassigned column; same
# member order = ascending column index): both read the same |corr| values, only the access pattern differs.
CORR_BLOCK_THRESHOLD = 1500


def corr_clusters(X: pd.DataFrame, thr: float = 0.92, block_threshold: int = CORR_BLOCK_THRESHOLD):
    """Greedy |Pearson| >= thr clustering. Returns (reps, members) where reps is the representative list (first
    column of each cluster) and members maps rep -> [member columns incl. the rep]. Computed once and shared.

    Vectorized inner scan: the per-pair Python ``abs(C[i,j]) >= thr`` test is replaced by a single boolean adjacency
    matrix ``A = |C| >= thr`` and an ``np.flatnonzero`` of each rep's upper row -- byte-identical to the prior greedy
    (same rep order = first unassigned column; same member order = ascending column index), but skips the full j-loop
    for the common all-singleton frame where each row has few/no above-threshold candidates (measured 3.83x on the
    p=240 mostly-singleton hybrid frame; the glue floor's single largest item, ~86% of the hybrid's own tottime).

    WIDE-DATA path (p > ``block_threshold``): the full p x p ``np.corrcoef`` materializes an O(p^2) float64 matrix
    (~800 MB at p=10k) -- the one uncapped super-linear-in-p hotspot among the FS paths. Above the threshold we never
    build the dense matrix: columns are z-standardized once, then for each rep we compute ONLY its row of correlations
    (a single 1 x p GEMV: z_i @ Z / (n-1)) and threshold that, so peak extra memory is O(n*p) for Z plus O(p) per row.
    The greedy is identical (it reads the same |corr| values for the same (i, j>i) pairs), so reps + members are
    set-identical to the full-matrix result (verified bit/set-identical on small frames in the test suite).

    cProfile/tracemalloc (2026-06-19, dev box, n*p random frame): p=6000,n=800 -> FULL np.corrcoef path 1141 ms /
    756 MB peak; BLOCKED path 1344 ms / 132 MB peak == 5.8x less memory at ~equal wall. The gap widens with p (the
    full path is O(p^2): ~800 MB at p=10k before corrcoef's internal deviation-matrix copy ~doubles it -> the OOM
    blocker), while the blocked path stays O(n*p) for Z + O(bs*p) for the slab. Small frames (p <= block_threshold)
    keep the faster full path. (p=2000: FULL 187 ms vs BLOCKED 311 ms -- the per-block matmul overhead, paid only
    above the threshold where the memory bound matters more than the sub-second wall.)"""
    cols = list(X.columns)
    p = len(cols)
    if p == 1:  # single column (mirrors the C.ndim == 0 corrcoef scalar case)
        return [cols[0]], {cols[0]: [cols[0]]}
    if p <= block_threshold:
        C = np.nan_to_num(np.corrcoef(X.values, rowvar=False))
        if C.ndim == 0:  # single column (defensive; p==1 already returned above)
            return [cols[0]], {cols[0]: [cols[0]]}
        A = np.abs(C) >= thr  # adjacency once (vectorized), replaces the per-pair Python compare
        reps, members, assigned = [], {}, np.zeros(p, dtype=bool)
        for i in range(p):
            if assigned[i]:
                continue
            assigned[i] = True
            reps.append(cols[i]); ms = [cols[i]]
            cand = np.flatnonzero(A[i, i + 1:]) + (i + 1)   # only the above-threshold j>i (ascending index preserved)
            for j in cand:
                if not assigned[j]:
                    assigned[j] = True; ms.append(cols[j])
            members[cols[i]] = ms
        return reps, members

    # --- wide-data blocked path: standardize once, compute correlations in COLUMN BLOCKS (no dense p x p) ---
    # Columns are z-standardized once (Z is O(n*p)); then a block of ``bs`` rep-rows is correlated against all p columns
    # in a single (bs x p) GEMV slab (Z[:, blk].T @ Z), so peak transient memory is O(bs*p) -- a small constant slab,
    # never the full p x p. The greedy reads each rep's slab row exactly as it would the dense matrix row, so the
    # assignment is identical to the full-matrix path. bs trades speed (bigger = fewer, larger matmuls) vs the slab's
    # memory (bs*p*8 bytes); 512 keeps the slab modest (e.g. 512*10000*8 ~= 41 MB) while recovering near-BLAS speed.
    M = np.asarray(X.values, dtype=np.float64)
    n = M.shape[0]
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    Z = M - mu
    # zero-variance columns -> corr undefined; np.corrcoef yields nan there (nan_to_num -> 0). Mirror that by leaving
    # their standardized column at 0, so every correlation INVOLVING them is 0 (< thr) -> they stay singletons, exactly
    # as nan_to_num(corrcoef) produced. Non-constant columns are divided by their std as usual.
    nz = sd > 0
    Z[:, nz] = Z[:, nz] / sd[nz]
    denom = float(n - 1) if n > 1 else 1.0
    bs = 512
    reps, members, assigned = [], {}, np.zeros(p, dtype=bool)
    for start in range(0, p, bs):
        stop = min(start + bs, p)
        # slab[r, j] = corr(col start+r, col j); only used for the upper triangle (j > rep index) below.
        slab = np.nan_to_num((Z[:, start:stop].T @ Z) / denom)  # (bs x p) transient, never p x p
        adj = np.abs(slab) >= thr
        for r in range(stop - start):
            i = start + r
            if assigned[i]:
                continue
            assigned[i] = True
            reps.append(cols[i]); ms = [cols[i]]
            if nz[i]:
                cand = np.flatnonzero(adj[r, i + 1 :]) + (i + 1)  # ascending j>i preserved (identical to full path)
                for j in cand:
                    if not assigned[j]:
                        assigned[j] = True; ms.append(cols[j])
            members[cols[i]] = ms
    return reps, members


class HybridSelector:
    """Compute-once / share-many hybrid feature selector (see module docstring)."""

    def __init__(self, vote: int = 1, prescreen: bool = True, expand_clusters: bool = False,
                 fi_guard: bool = False, corr_thr: float = 0.92, use_mrmr: bool = True,
                 use_fe: bool = True, fe_max_steps: int = 1, boruta_driver: str = "gini", anchor_fe: bool = False,
                 use_tree_member: bool = True, tree_top_k: int = 0, tree_cooccur_pairs: int = 12,
                 tree_n_estimators: int = 80, tree_max_depth: int = 3, tree_prod_gate: str = "synergy",
                 tree_rich_ops: tuple = ("mul", "absd", "sign", "rat"),
                 cooccur_weight: str = "gain", cluster_rep: str = "first",
                 mrmr_synergy_cap: int = 250,
                 hybrid_corr_max_features: int = 2000,
                 random_state: int = 42, classification: "Optional[bool]" = None, name: str = "hybrid"):
        # cooccur_weight (default "gain" -- MEASURED win on interaction beds): how the tree member ranks candidate
        # co-occurrence PAIRS proposed from GBM split co-occurrence. "count" weights a pair by raw frequency (how many
        # trees split on both a and b); "gain" weights it by the summed split GAIN the two features contribute within
        # each tree (how much they actually reduce loss). Frequency over-rewards shallow high-frequency-but-low-gain
        # splits, so gain ranks true interactions higher. See _tree_signals; "count" recoverable for replay/legacy.
        self.cooccur_weight = cooccur_weight
        # cluster_rep (default "first"): when a correlation cluster is collapsed to one representative, which member to
        # keep. "first" = the cluster's first column (default); "max_fi" = highest single mean perm-FI; "sum_fi" =
        # highest summed-per-repeat perm-FI. The FI-based reps are bed-dependent -- benched 5/9 cells but they REGRESS
        # first-column on some correlated-cluster beds (>0.005 honest-AUC), so they stay OPT-IN, not the default. See
        # _rep_member + _benchmarks/fs_hybrid/round5_innovate_cooccur_clusterrep.json.
        self.cluster_rep = cluster_rep
        # tree_rich_ops (default the full set -- MEASURED +0.020 on madelon 3-seed over products-only, variance
        # halved): the operators the tree member engineers per co-occurrence pair. Beyond the "mul" product,
        # absdiff/signed/ratio recover NON-product interaction signal (|a-b|, sign(a)|b|, a/(|b|+1)) a bilinear term
        # cannot linearize; the synergy gate keeps each only where it genuinely helps (madelon gains, synth unchanged).
        # Set to ("mul",) for products-only. See _TREE_OPS.
        self.tree_rich_ops = tuple(tree_rich_ops)
        # mrmr_synergy_cap (default 250 -- MEASURED win): the MRMR member's fe_synergy_screen_max_features. MRMR's
        # default (60) SKIPS the synergy bootstrap on frames wider than 60 cols, so on a moderate-width frame
        # (e.g. hard_synth, 220 cols) MRMR never engineers the interaction products its bootstrap would find. Raising
        # to 250 enables the bootstrap on moderate-p frames -> hybrid hard_synth +0.030 (3-seed, all seeds up,
        # ADDITIVE to the tree member), byte-identical no-op on narrow frames (synth 52<=cap, bootstrap already ran)
        # AND on very-wide frames (madelon 500>250, bootstrap still skipped -> avoids its O(p^2) cost where it finds
        # nothing: madelon's 5-dim XOR is not bilinear). 250 is the cost/benefit sweet spot. (round4_hybrid_mrmrcap_bench)
        self.mrmr_synergy_cap = mrmr_synergy_cap
        # hybrid_corr_max_features (default 2000 -- the wide-data clustering guard): corr_clusters is the one uncapped
        # super-linear-in-p hotspot (a full p x p Pearson matrix; ~800 MB float64 at p=10k). BEFORE clustering, X_aug is
        # narrowed to the columns the shared permutation-FI found relevant (FI>0) PLUS all MRMR-relevant and engineered
        # columns (those never drop, they passed their own FE gate). The relevant set is typically << p on wide noisy
        # frames, so clustering only ever sees the survivors. If the survivor set is still larger than this cap, the
        # corr_clusters blocked path (see CORR_BLOCK_THRESHOLD) keeps peak memory at O(n*p) instead of O(p^2). Columns
        # dropped here are pure-noise singletons (FI==0, not MRMR-relevant, not engineered): they would each form their
        # own singleton cluster and never be voted/selected anyway, so dropping them before clustering is selection-
        # neutral on the kept set. Set very high (or None) to cluster the full augmented frame (legacy behaviour).
        self.hybrid_corr_max_features = hybrid_corr_max_features
        self.vote = vote
        self.prescreen = prescreen
        self.expand_clusters = expand_clusters
        self.fi_guard = fi_guard
        self.corr_thr = corr_thr
        self.use_mrmr = use_mrmr
        # use_tree_member (default ON -- MEASURED win on interaction-heavy real data): a cheap shallow-GBM that
        # contributes a signal the MI-filter members structurally MISS. MRMR's marginal-MI greedy collapses on
        # interaction-heavy data (madelon: 3 features, lgbm 0.69) because the informative operands have ~0 marginal
        # MI; a depth-limited GBM branches on them anyway. The member adds two things to the shared composition:
        #   (1) a VOTE for its top-`tree_top_k` features by split importance, and
        #   (2) candidate co-occurrence PRODUCT columns (raw[a]*raw[b] for the top-`tree_cooccur_pairs` pairs that
        #       co-occur within a tree, gain-weighted) folded into the shared augmented frame.
        # Both are GATED by the shared honest permutation-FI so the member self-regulates by regime: on madelon the
        # product columns clear the FI floor and lift the hybrid 0.805 -> ~0.84 (beating standalone tree_top20+cooccur
        # 0.840 measured 3-seed); on the FE-saturated synthetic the raw products do NOT clear the floor (MRMR's
        # Hermite/prewarp transforms are what win there), so they are not voted and MRMR's win is preserved -- the
        # two FE styles fail oppositely and the composition keeps both. (round4_tree_seed_bench / _tree_rescue_validate)
        self.use_tree_member = use_tree_member
        self.tree_top_k = tree_top_k
        self.tree_cooccur_pairs = tree_cooccur_pairs
        self.tree_n_estimators = tree_n_estimators
        self.tree_max_depth = tree_max_depth
        # tree_prod_gate: which co-occurrence products earn a vote. The gate is the regime self-regulator -- it must
        # admit madelon's true interaction products (big win) while rejecting the FE-saturated bed's weak raw products
        # (which dilute, esp. distance-based downstreams). "synergy" (default): admit prod(a,b) only if its shared-FI
        # exceeds BOTH operands' FI (a genuine interaction adds information beyond its parts). "relevant_median":
        # FI >= median FI over the prescreened survivors (a high bar). "raw_median": FI >= median over ALL raw cols
        # (loose -- regresses the FE-saturated bed because its many noise features drag the median to ~0).
        self.tree_prod_gate = tree_prod_gate
        # use_fe: let the MRMR member feature-ENGINEER (fe_max_steps) and SHARE its engineered columns to every other
        # member via a once-built augmented frame X_aug=[raw|engineered]. This closes the measured ~0.05 AUC gap to
        # mrmr_fe: the bed's pure-interaction signal (inf_4*inf_5, ~0 marginal) is unrecoverable by selection alone;
        # only FE creates the term. Engineering is replayed leakage-free at transform time via the fitted MRMR member.
        self.use_fe = use_fe
        self.fe_max_steps = fe_max_steps
        # boruta_driver: importance measure for the BorutaShap member. "gini" is the default. The held-out
        # "permutation" driver was tried as a noise-leak fix (gini admits ~1-3 spurious columns that, under vote=1,
        # cap the linear downstream): MEASURED net-negative without FE (it only partially cut noise 2.33->1.33 AND
        # cost recall 0.96->0.88 -> auc 0.784->0.779, at 2.5x runtime) and merely TIED gini with FE on (0.8299 vs
        # 0.8288) at ~2x cost. The real cure for the noise-capped linear downstream is feature engineering (use_fe),
        # which lifts logit ~0.75->0.85 so residual noise no longer matters. So gini stays the default; permutation
        # remains available as the high-precision option.
        self.boruta_driver = boruta_driver
        # anchor_fe (default OFF -- MEASURED net-negative): the idea was to keep ALL of MRMR's picks verbatim and let
        # the other members only ADD, so selected ⊇ mrmr_selected. But that guarantees more FEATURES, not higher AUC:
        # benched on make_dataset (anchor 0.8348 < no-anchor 0.8356 < mrmr_fe 0.8367) AND on the hard bed (anchor
        # 0.7744 < no-anchor 0.7752) -- the kept-then-added features dilute the set. The plain cluster-vote (anchor_fe
        # =False) is the better default on both beds. Kept as an option; not the fix it was meant to be.
        self.anchor_fe = anchor_fe
        self.random_state = random_state
        # classification (default None -> sniff via type_of_target, but a 2-value FLOAT regression target sniffs as
        # "binary" and is silently fed to LGBMClassifier). Pass classification=True/False explicitly to declare the task
        # and skip value-sniffing entirely (mirrors hetero_vote / ShapProxiedFS which take an explicit classification=).
        self.classification = classification
        self.name = name

    # ------------------------------------------------------------------ shared-once artifacts
    def _shared_perm_fi(self, X, y):
        """One honest held-out permutation-importance pass (the shared FI). Mirrors RFECV's permutation driver:
        fit a single LightGBM on a train split, permute on the held-out split, return col -> mean importance."""
        from sklearn.model_selection import train_test_split
        from sklearn.inspection import permutation_importance
        import lightgbm as lgb
        # Stratify only when every class has >=2 members; a rare class (high-cardinality multiclass, count-like target)
        # would otherwise crash train_test_split with an opaque "least populated class has 1 member" error.
        _yv = np.asarray(y)
        _, _counts = np.unique(_yv, return_counts=True)
        _strat = y if _counts.min() >= 2 else None
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=self.random_state, stratify=_strat)
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.06, n_jobs=-1, verbose=-1, random_state=self.random_state)
        m.fit(Xtr, ytr)
        pi = permutation_importance(m, Xva, yva, n_repeats=4, random_state=self.random_state, n_jobs=-1)
        # store the summed-per-repeat importance too (drives cluster_rep="sum_fi"); falls back to mean*n_repeats if the
        # raw per-repeat matrix is unavailable. sum_fi prefers members whose importance is consistently high across
        # repeats, a more stable representative than a single lucky mean.
        imp = getattr(pi, "importances", None)
        sums = imp.sum(axis=1) if imp is not None else pi.importances_mean * 4.0
        self._fi_sum_ = {c: float(s) for c, s in zip(X.columns, sums)}
        return {c: float(v) for c, v in zip(X.columns, pi.importances_mean)}

    def _run_mrmr(self, X, y):
        """MRMR as a whole object -> (selected raw columns, artifact dict). Guarded: MRMR is shared infra under
        active development, so any failure degrades to (relevance-by-FI only, no precomputed) rather than crashing."""
        self._mrmr_member, self._eng_names, self._eng_rename = None, [], {}
        try:
            from mlframe.feature_selection.filters import MRMR
            # fe_strict (round-3 M3-4/M3-7): tighter synergy + engineered-MI prevalence gates than MRMR's defaults
            # (1.15 / 0.90). Measured on make_dataset: cuts spurious noise-product engineered features (~1.0 -> 0.67)
            # and HALVES the engineered set (15 -> 7) while raising standalone mrmr_fe AUC 0.831 -> 0.837 -- a cleaner,
            # more parsimonious FE substrate to share to the other members.
            fe_strict = dict(fe_synergy_min_prevalence=1.5, fe_min_engineered_mi_prevalence=0.97) if self.use_fe else {}
            # mrmr_synergy_cap (default 250; see __init__): raise the MRMR member's fe_synergy_screen_max_features so
            # its synergy bootstrap RUNS on moderate-width frames (e.g. 220 cols) where MRMR's default (60) skips it.
            # MEASURED additive to the tree member: hybrid hard_synth +0.030 (3-seed), no-op on narrow + very-wide frames.
            if self.use_fe and getattr(self, "mrmr_synergy_cap", None) is not None:
                fe_strict["fe_synergy_screen_max_features"] = int(self.mrmr_synergy_cap)
            m = MRMR(verbose=0, fe_max_steps=(self.fe_max_steps if self.use_fe else 0), n_jobs=-1,
                     random_seed=self.random_state, retain_artifacts=True, retain_bins=True, **fe_strict)  # type: ignore[arg-type]
            m.fit(X, y)
            selected = [c for c in m.get_feature_names_out() if c in X.columns]
            try:
                artifacts = m.export_artifacts()
            except Exception:
                artifacts = None
            # Capture the engineered columns (MRMR outputs not in X) so they can be SHARED to the other members via
            # the augmented frame; map their non-ASCII recipe names to LightGBM-safe eng_N names (stable order).
            if self.use_fe:
                try:
                    out_cols = list(m.transform(X.iloc[:5]).columns)
                    k = 0
                    for c in out_cols:
                        if c not in X.columns:
                            self._eng_names.append(c); self._eng_rename[c] = f"eng_{k}"; k += 1
                    self._mrmr_member = m
                except Exception:
                    self._mrmr_member, self._eng_names, self._eng_rename = None, [], {}
            return selected, artifacts
        except Exception as e:
            warnings.warn(f"HybridSelector: MRMR stage degraded ({type(e).__name__}: {e})", stacklevel=2)
            return [], None

    def _tree_signals(self, X, y):
        """One cheap depth-limited GBM on RAW X -> (importance-ranked feature list, top co-occurring raw pairs).
        Co-occurrence proxy: within each tree, every pair of split features gets a count weighted by the tree's
        total split gain -- features the tree branches on together are the interactions it exploits, a supervised
        operand proposer blind to marginal MI (the exact signal MRMR's marginal-MI greedy discards). Stored on self
        (ranked features + product pairs) so the product columns replay leakage-free at transform time."""
        self._tree_ranked_, self._tree_prod_pairs_, self._tree_prod_names_, self._tree_op_ = [], [], [], {}
        if not self.use_tree_member:
            return
        try:
            from itertools import combinations
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=self.tree_n_estimators, max_depth=self.tree_max_depth,
                                   num_leaves=2 ** self.tree_max_depth, learning_rate=0.1, n_jobs=-1,
                                   verbose=-1, random_state=self.random_state)
            m.fit(X, y)
            cols = list(X.columns)
            imp = pd.Series(m.feature_importances_, index=cols)
            self._tree_ranked_ = [c for c in imp.sort_values(ascending=False).index if imp[c] > 0]
            tdf = m.booster_.trees_to_dataframe()
            tdf = tdf[tdf["split_feature"].notna()]
            pair_w: dict = {}
            mode = str(getattr(self, "cooccur_weight", "gain")).lower()
            for _tid, g in tdf.groupby("tree_index"):
                feats = sorted(set(g["split_feature"].tolist()))
                if mode == "count":
                    # count: every co-occurring pair gets +1 -- raw split-co-occurrence frequency (legacy).
                    for a, b in combinations(feats, 2):
                        pair_w[(a, b)] = pair_w.get((a, b), 0.0) + 1.0
                else:
                    # gain: a pair is weighted by the loss reduction its two features actually contribute in this tree
                    # (sum of their per-node split gains), so true interactions outrank shallow high-frequency splits.
                    per_feat = g.groupby("split_feature")["split_gain"].sum()
                    for a, b in combinations(feats, 2):
                        pair_w[(a, b)] = pair_w.get((a, b), 0.0) + float(per_feat.get(a, 0.0)) + float(per_feat.get(b, 0.0))
            # The co-occurrence PRODUCTS are feature engineering -> gate them on use_fe (use_fe=False means no
            # engineering of ANY kind, raw selection only). The importance RANKING above is kept regardless, so the
            # top-k selection votes still work under use_fe=False if tree_top_k>0.
            if self.use_fe:
                _pair_w_sorted = sorted(pair_w.items(), key=lambda kv: kv[1], reverse=True)[: self.tree_cooccur_pairs]
                top = [(a, b) for (a, b), _ in _pair_w_sorted if a in cols and b in cols]
                # expand each co-occurrence pair to one candidate column per rich operator; each column carries its
                # operand pair (so the synergy gate scores it independently) and its op (so _augment replays it).
                ops = [o for o in self.tree_rich_ops if o in _TREE_OPS] or ["mul"]
                pairs, names = [], []
                for i, (a, b) in enumerate(top):
                    for op in ops:
                        nm = f"t{op}_{i}"
                        pairs.append((a, b)); names.append(nm); self._tree_op_[nm] = (a, b, op)
                self._tree_prod_pairs_, self._tree_prod_names_ = pairs, names
        except Exception as e:
            warnings.warn(f"HybridSelector: tree-signal stage degraded ({type(e).__name__}: {e})", stacklevel=2)
            self._tree_ranked_, self._tree_prod_pairs_, self._tree_prod_names_, self._tree_op_ = [], [], [], {}

    def _admit_tree_products(self, relevant, raw_cols):
        """Apply ``tree_prod_gate`` to decide which co-occurrence product columns earn a tree vote (the regime
        self-regulator). Uses only the already-shared FI -- no extra compute. Returns a list of admitted tprod names."""
        if not getattr(self, "_tree_prod_names_", None):
            return []
        fi, gate = self.fi_, str(self.tree_prod_gate).lower()
        prods = self._tree_prod_names_
        if gate == "raw_median":
            floor = max(float(np.median([fi.get(c, 0.0) for c in raw_cols])) if raw_cols else 0.0, 1e-12)
            return [nm for nm in prods if fi.get(nm, 0.0) >= floor]
        if gate == "relevant_median":
            rfis = [fi.get(c, 0.0) for c in relevant if c not in set(prods)]  # bar = survivors, excluding products
            floor = max(float(np.median(rfis)) if rfis else 0.0, 1e-12)
            return [nm for nm in prods if fi.get(nm, 0.0) >= floor]
        # default "synergy": a product earns its place only if it is more informative than BOTH operands alone
        # (FI[a*b] > max(FI[a], FI[b])) -- a genuine interaction adds information beyond its parts. Regime-agnostic:
        # madelon's true products clear it; the FE-saturated bed's redundant raw products (subsumed by the operands
        # or by MRMR's transforms) do not.
        out = []
        for (a, b), nm in zip(self._tree_prod_pairs_, prods):
            if fi.get(nm, 0.0) > max(fi.get(a, 0.0), fi.get(b, 0.0)):
                out.append(nm)
        return out

    def _rep_member(self, members):
        """Pick the kept representative of a (correlation) cluster per ``cluster_rep``. ``members`` is already the
        candidate subset (filtered to in-frame / relevant). "first" = the cluster's first column (legacy, arbitrary);
        "max_fi" = highest single mean perm-FI; "sum_fi" (default) = highest summed-per-repeat perm-FI (most stable).
        Empty input returns None. Pure function of the shared FI -- no extra compute."""
        if not members:
            return None
        rep = str(getattr(self, "cluster_rep", "sum_fi")).lower()
        if rep == "first":
            return members[0]
        if rep == "sum_fi":
            fsum = getattr(self, "_fi_sum_", None) or self.fi_
            return max(members, key=lambda f: fsum.get(f, 0.0))
        return max(members, key=lambda f: self.fi_.get(f, 0.0))

    def _augment(self, X):
        """Return [raw X | MRMR engineered cols | tree co-occurrence op cols], all safe-renamed and replayed
        leakage-free (MRMR cols via the fitted member; tree cols as pure ops of raw[a],raw[b] from the stored specs).
        Identity when neither FE source produced columns."""
        eng_cols, tprod_cols = {}, {}
        if self._eng_names and self._mrmr_member is not None:
            try:
                out = self._mrmr_member.transform(X)
                for c in self._eng_names:
                    eng_cols[self._eng_rename[c]] = out[c].values
            except Exception as e:
                warnings.warn(f"HybridSelector: FE augment failed ({type(e).__name__}: {e}); skipping MRMR eng cols", stacklevel=2)
                eng_cols = {}
        tree_op = getattr(self, "_tree_op_", {})
        for nm in getattr(self, "_tree_prod_names_", []):
            a, b, op = tree_op.get(nm, (None, None, "mul"))
            if a in X.columns and b in X.columns:
                v = _TREE_OPS.get(op, _TREE_OPS["mul"])(X[a].values.astype(float), X[b].values.astype(float))
                tprod_cols[nm] = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        if not eng_cols and not tprod_cols:
            return X
        base = X.reset_index(drop=True)
        extra = pd.DataFrame({**eng_cols, **tprod_cols}, index=base.index)
        aug = pd.concat([base, extra], axis=1)
        return aug.loc[:, ~aug.columns.duplicated()]

    # ------------------------------------------------------------------ reused members on the shared narrowed space
    def _run_shap(self, X, y, relevant, artifacts):
        """ShapProxiedFS fed the SHARED artifacts via precomputed= (restricted to the relevant survivor set)."""
        from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS
        precomputed = None
        # MRMR artifacts cover only RAW columns; if the relevant set carries engineered cols (use_fe), ShapProxiedFS
        # would warn + discard the precomputed entirely (feature_names mismatch). Only share artifacts when every
        # relevant column is covered (the pure-selection case) -- otherwise let the shap member recompute cleanly.
        if artifacts is not None:
            try:
                from mlframe.feature_selection.shap_proxied_fs import restrict_artifacts
                names = list(artifacts.get("feature_names", []))
                if relevant and all(c in names for c in relevant):
                    precomputed = restrict_artifacts(artifacts, [names.index(c) for c in relevant])
            except Exception:
                precomputed = None
        p = len(relevant)
        s = ShapProxiedFS(classification=True, n_splits=3, top_n=20, min_features=min(8, p),
                          prefilter_top=min(40, p), prefilter_n_estimators=60, oof_shap_n_estimators=60,
                          revalidation_n_estimators=60, n_revalidation_models=2, trust_guard=True,
                          trust_guard_n_estimators=20, cluster_features="auto", within_cluster_refine=True,
                          parsimony_tol=0.005, precomputed=precomputed, random_state=self.random_state, verbose=False)
        s.fit(X[relevant], y)
        return [c for c in s.selected_features_ if c in X.columns]

    def _run_boruta_premerge(self, X, y, relevant):
        """BorutaShap on the SHARED cluster representatives (premerge, R2b-6), then re-expand accepted reps. Uses the
        boruta_driver importance measure ("permutation" held-out by default -> ~0 noise leak, vs "gini" which admits
        the top spurious column and caps the linear downstream under vote=1)."""
        from mlframe.feature_selection.boruta_shap import BorutaShap
        from sklearn.ensemble import RandomForestClassifier
        # restrict the shared clusters to the relevant survivors; rep = highest-FI member of each restricted cluster
        rep_members = {}
        for r, ms in self.members_.items():
            keep = [m for m in ms if m in relevant]
            if keep:
                rep = self._rep_member(keep)
                rep_members[rep] = keep
        reps = list(rep_members.keys())
        if len(reps) < 2:
            reps = list(relevant)
            rep_members = {c: [c] for c in reps}
        driver = str(self.boruta_driver).lower()
        # held-out permutation needs the 30% split (train_or_test="test"); gini works in-bag.
        tot = "test" if driver == "permutation" else "train"
        b = BorutaShap(model=RandomForestClassifier(n_estimators=80, n_jobs=-1, random_state=self.random_state),
                       importance_measure=self.boruta_driver, permutation_n_repeats=2, classification=True,
                       n_trials=50, percentile=95, train_or_test=tot, verbose=False, random_state=self.random_state)
        b.fit(X[reps], y)
        accepted_reps = [c for c in b.selected_features_ if c in reps]
        expanded = []
        for r in accepted_reps:
            expanded.extend(rep_members.get(r, [r]))
        return [c for c in dict.fromkeys(expanded) if c in X.columns]

    # ------------------------------------------------------------------ fit / transform
    @rng_hygienic_fit
    def fit(self, X, y):
        # Polars frames carry their column names in a plain ``list`` (no ``.has_duplicates`` / ``.iloc`` / ``.values``), so the pandas-only glue below
        # would crash at the first ``X.columns.has_duplicates``. Bridge to an Arrow-backed pandas VIEW once at the boundary (zero-copy for numeric cols),
        # which captures the REAL polars column names. The defensive LGBM ``feature_names_in_`` setter shim guards the inner LightGBM fits against the
        # LightGBM 4.x + sklearn read-only-property write-path defect when a non-pandas input slips through to a member.
        from mlframe.training import _patch_lgb_feature_names_in_setter
        _patch_lgb_feature_names_in_setter()
        X = _as_pandas_view(X)
        # Duplicate column names make ``X[label]`` return a DataFrame (not a Series) and crash the member selectors (MRMR / LightGBM FI / cluster expand) that assume unique names. Surface a clear error at fit entry.
        if hasattr(X, "columns") and X.columns.has_duplicates:
            dup_names = X.columns[X.columns.duplicated()].unique().tolist()
            raise ValueError(
                f"HybridSelector.fit: duplicate column names not supported: {dup_names[:10]}. "
                f"De-duplicate (e.g. ``X.loc[:, ~X.columns.duplicated()]`` or rename) before fitting."
            )
        # HybridSelector is intrinsically CLASSIFICATION-only: the shared FI uses a stratified split + LGBMClassifier, and
        # the ShapProxiedFS / BorutaShap members are wired with classification=True (binary). A continuous / multi-output
        # target previously crashed deep inside sklearn's stratify ("least populated class has 1 member") or the LGBM 1d
        # check -- opaque tracebacks. Detect the unsupported target shape up front and raise an actionable error.
        from sklearn.utils.multiclass import type_of_target
        y_arr = np.asarray(y)
        if y_arr.ndim > 1 and not (y_arr.ndim == 2 and y_arr.shape[1] == 1):
            raise ValueError(
                f"HybridSelector supports a single-output classification target only; got y with shape {y_arr.shape} "
                f"(multilabel / multi-output is not supported)."
            )
        if self.classification is False:
            # Caller explicitly declared a regression task: HybridSelector is classification-only, so reject up front
            # instead of value-sniffing (a 2-value float regression target would otherwise sniff as "binary" and be fed
            # to LGBMClassifier silently). This is the whole point of the explicit knob.
            raise ValueError(
                "HybridSelector supports classification targets only; got classification=False (regression). "
                "For regression use MRMR / RFECV / GroupAwareMRMR with a regression estimator."
            )
        if self.classification is None:
            try:
                _ttype = type_of_target(y_arr.ravel())
            except Exception:
                _ttype = "unknown"
            if _ttype not in ("binary", "multiclass", "unknown"):
                raise ValueError(
                    f"HybridSelector supports classification targets only (binary / multiclass); got a "
                    f"'{_ttype}' target. For regression use MRMR / RFECV / GroupAwareMRMR with a regression estimator."
                )
            if _ttype in ("binary", "multiclass") and np.asarray(y).dtype.kind == "f":
                # A float target that sniffs as classification (e.g. exactly 2 distinct float values) is ambiguous --
                # it may be a low-cardinality regression target. Warn so a silent classifier-on-regression run is visible.
                warnings.warn(
                    "HybridSelector: target dtype is float but was value-sniffed as a classification target "
                    f"('{_ttype}'); pass classification=True to confirm or classification=False if this is regression.",
                    stacklevel=2,
                )
        # STAGE 0 -- MRMR FIRST (it engineers the shared FE columns), then build the augmented frame X_aug once;
        # every downstream shared artifact + member then operates on raw+engineered features.
        self.mrmr_selected_, self.artifacts_ = self._run_mrmr(X, y) if self.use_mrmr else ([], None)
        if not self.use_mrmr:
            self._mrmr_member, self._eng_names, self._eng_rename = None, [], {}
        # tree signals on RAW X (importance ranking + co-occurrence product pairs). The candidate product columns are
        # folded into the augmented frame and scored by the ONE shared FI pass, THEN GATED: only gate-admitted products
        # survive -- rejected ones are pruned from the frame AND the replay pairs, so they never reach the clusters,
        # members, vote, or transform. This makes tree_prod_gate actually BIND (it is the regime self-regulator: on
        # interaction-heavy data the true products clear synergy/FI; on FE-saturated data the redundant raw products do
        # not, so they are dropped and MRMR's engineered transforms win unchallenged -- no dilution of that regime).
        self._tree_signals(X, y)
        X_aug = self._augment(X)  # all candidate products present, for honest FI scoring
        fi_full = self._shared_perm_fi(X_aug, y)  # the single (expensive) shared FI pass
        if self.use_tree_member and getattr(self, "_tree_prod_names_", None):
            self.fi_ = fi_full  # _admit_tree_products reads self.fi_
            raw_cols = list(X.columns)
            survivor_proxy = [c for c in raw_cols if fi_full.get(c, 0.0) > 0.0] or raw_cols  # bar for relevant_median
            admitted = set(self._admit_tree_products(survivor_proxy, raw_cols))
            rejected = [nm for nm in self._tree_prod_names_ if nm not in admitted]
            if rejected:
                kept = [(p, nm) for p, nm in zip(self._tree_prod_pairs_, self._tree_prod_names_) if nm in admitted]
                self._tree_prod_pairs_ = [p for p, _ in kept]
                self._tree_prod_names_ = [nm for _, nm in kept]
                self._tree_op_ = {nm: self._tree_op_[nm] for nm in self._tree_prod_names_ if nm in self._tree_op_}
                X_aug = X_aug.drop(columns=[c for c in rejected if c in X_aug.columns])
        cols = list(X_aug.columns)
        # engineered = MRMR-engineered + the ADMITTED tree products (rejected ones are already pruned). Both are
        # candidate engineered columns exempt from the raw-FI prescreen (they passed their own FE/synergy gate).
        engineered = set(self._eng_rename.values()) | set(getattr(self, "_tree_prod_names_", []))
        self._Xaug_, self._y_ = X_aug, y  # stashed for combine-rule variants/diagnostics (benchmark composition class)

        # shared artifacts: restrict the already-computed FI to the kept columns (no recompute), cluster the kept frame
        self.fi_ = {c: fi_full.get(c, 0.0) for c in cols}
        # WIDE-DATA CLUSTERING GUARD (hybrid_corr_max_features): narrow X_aug to the relevance-survivor set BEFORE the
        # O(p^2) corr_clusters. Survivors = FI>0 OR MRMR-relevant OR engineered (the exact prescreen rule below). The
        # dropped columns are pure-noise singletons (FI==0, not MRMR, not engineered) -- they would each cluster alone
        # and never be voted/selected, so omitting them is selection-neutral on the kept set while bounding clustering
        # cost on wide frames. self.fi_ retains the FULL frame so _rep_fi / cluster_rep still see every column.
        mrmr_set_pre = set(self.mrmr_selected_)
        cluster_cols = [c for c in cols if self.fi_.get(c, 0.0) > 0.0 or c in mrmr_set_pre or c in engineered]
        cap = getattr(self, "hybrid_corr_max_features", None)
        if len(cluster_cols) < 2 or not self.prescreen:
            cluster_cols = cols  # too few survivors / prescreen off -> cluster the full frame
        elif cap is not None and len(cluster_cols) > cap:
            # survivor set still over the cap: keep the engineered + MRMR cols (must survive) and top-FI raw fill.
            must = [c for c in cluster_cols if c in engineered or c in mrmr_set_pre]
            rest = [c for c in cluster_cols if c not in engineered and c not in mrmr_set_pre]
            # Full sort key (-fi, col) so ties on importance break deterministically by column name instead of
            # depending on the input order -- otherwise two equal-FI columns at the cap boundary could be kept
            # or dropped non-reproducibly across runs / platforms.
            rest = sorted(rest, key=lambda c: (-self.fi_.get(c, 0.0), c))[: max(cap - len(must), 0)]
            cluster_cols = [c for c in cols if c in set(must) | set(rest)]  # preserve original column order
        self._cluster_cols_ = cluster_cols
        X_cluster = X_aug[cluster_cols] if len(cluster_cols) != len(cols) else X_aug
        self.reps_, self.members_ = corr_clusters(X_cluster, self.corr_thr)
        # columns dropped from clustering become their own singleton cluster (selection-neutral: FI==0 noise) so the
        # downstream cluster_of_ / vote still has an entry for every column it might encounter.
        for c in cols:
            if c not in self.members_:
                self.reps_.append(c); self.members_[c] = [c]
        self.cluster_of_ = {f: r for r, ms in self.members_.items() for f in ms}

        # STAGE 1 -- shared relevance pre-screen (held-out permutation FI > 0, OR MRMR-relevant, OR engineered:
        # engineered columns already passed MRMR's FE gate, so they are never dropped by the raw-FI prescreen)
        if self.prescreen:
            mrmr_set = set(self.mrmr_selected_)
            relevant = [c for c in cols if self.fi_.get(c, 0.0) > 0.0 or c in mrmr_set or c in engineered]
        else:
            relevant = list(cols)
        if len(relevant) < 2:
            relevant = list(cols)
        self.relevant_ = relevant

        # STAGE 2 -- reuse the member selectors on the shared, narrowed augmented space.
        # bench-attempt-rejected (2026-06-08, member-parallelism): _run_shap (~7s) and _run_boruta_premerge (~17s)
        # are independent (no shared mutable state) so they LOOK parallelizable, but each already runs n_jobs=-1
        # internally (sklearn RF / njit / LGBM) and SATURATES all cores on its own. Probed on the 8-core dev box
        # (hard_synth 1500x220): a 2-thread overlap won only 1.02x (22.34s -> 21.95s; bit-identical) -- no real
        # overlap, the inner pools time-slice the same cores. loky/process parallelism would be WORSE here: the
        # members are unequal (17s vs 7s) and RF/LGBM scale ~linearly with cores, so capping each to 4 cores to fit
        # two processes roughly doubles the 17s member to ~33s > the 24s sequential sum, and it multiplies the
        # RAM-tight ~8GB footprint + risks RNG non-determinism (re-seeded children break the bit-identical combine).
        # So the members stay SEQUENTIAL: the wall is irreducible sub-selector compute (already optimized upstream),
        # not glue. The hybrid's OWN glue is ~0.05s (<0.15% of wall); see corr_clusters for the one glue micro-opt.
        member_sel = {}
        member_sel["mrmr"] = [c for c in (self.mrmr_selected_ + sorted(engineered)) if c in relevant] or list(relevant)
        try:
            member_sel["shap"] = self._run_shap(X_aug, y, relevant, self.artifacts_)
        except Exception as e:
            warnings.warn(f"HybridSelector: shap member degraded ({type(e).__name__}: {e})", stacklevel=2)
            member_sel["shap"] = []
        try:
            member_sel["boruta"] = self._run_boruta_premerge(X_aug, y, relevant)
        except Exception as e:
            warnings.warn(f"HybridSelector: boruta member degraded ({type(e).__name__}: {e})", stacklevel=2)
            member_sel["boruta"] = []
        # TREE member: votes for its top-k features by split importance, plus the (already gate-admitted, already in
        # the frame) co-occurrence PRODUCT columns. The product gating happened up front (tree_prod_gate), so the
        # surviving products are exactly the regime-appropriate ones; here the member simply casts their vote.
        if self.use_tree_member and self._tree_ranked_:
            tree_votes = [c for c in self._tree_ranked_[: self.tree_top_k] if c in relevant]
            tree_votes += [nm for nm in getattr(self, "_tree_prod_names_", []) if nm in cols]
            tree_votes = [c for c in dict.fromkeys(tree_votes) if c in cols]
            if tree_votes:  # only register the member when it actually votes (dormant under
                member_sel["tree"] = tree_votes  # use_fe=False + tree_top_k=0 -> no key, keeps the raw-only contract)
        self.member_selections_ = member_sel

        # STAGE 3 -- cluster-aware vote over the shared clusters (pure, deterministic; extracted for unit testing)
        selected = self._combine(member_sel, cols)
        self.raw_selected_ = [c for c in cols if c in set(selected)] or cols[:1]
        self.n_engineered_ = sum(1 for c in self.raw_selected_ if c in engineered)
        # sklearn-style fitted attributes (raw_selected_ may include engineered eng_N names from X_aug)
        self.feature_names_in_ = list(X.columns)
        self.n_features_in_ = X.shape[1]
        self.selected_features_ = list(self.raw_selected_)
        return self

    def _combine(self, member_sel, cols):
        """Cluster-aware vote: a cluster is kept when >= ``vote`` members picked any of its features; each kept
        cluster contributes its highest-shared-FI member (or all members under ``expand_clusters``). Pure function
        of the shared state (self.fi_ / self.members_ / self.cluster_of_) and the per-member selections, so it is
        unit-testable without the expensive member fits."""
        if self.anchor_fe and member_sel.get("mrmr"):
            return self._combine_anchored(member_sel, cols)
        cluster_votes = defaultdict(set)
        for m, sel in member_sel.items():
            for f in sel:
                r = self.cluster_of_.get(f)
                if r is not None:
                    cluster_votes[r].add(m)
        support = {r: len(voters) for r, voters in cluster_votes.items()}

        def _rep_fi(r):
            return max((self.fi_.get(m, 0.0) for m in self.members_[r] if m in cols), default=0.0)

        # FI-CREDIBILITY GUARD (off by default -- MEASURED net loss): a cluster confirmed by only ONE member is
        # admitted only if its shared-permutation-FI clears the credibility floor = the median rep-FI of the
        # CONSENSUS clusters (>= 2 members). The intent was to drop the single-member noise leak while keeping
        # real single-member features. Benched (round2_hybrid_bench.py, 3 seeds on make_dataset): it DOES cut
        # noise 2.33 -> 0.33, but the single-member BASE features are exactly the weaker-FI ones, so they fall
        # below the consensus-FI median too -> base_recall collapses 0.958 -> 0.792 and auc_mean drops 0.784 ->
        # 0.772 (it degenerates to the vote=2 consensus set). The leaked noise's AUC cost is small (lgbm tolerates
        # it) while the lost recall is not. So fi_guard stays OFF by default; keep it only when a downstream needs
        # a cleaner, lower-recall set. Uses the already-shared FI, no extra compute.
        consensus_fis = [_rep_fi(r) for r, s in support.items() if s >= 2]
        floor = float(np.median(consensus_fis)) if (self.fi_guard and consensus_fis) else float("-inf")
        chosen = []
        for r, s in support.items():
            if s >= max(self.vote, 2):  # consensus -> always keep
                chosen.append(r)
            elif s >= self.vote and _rep_fi(r) >= floor:  # single-member -> must clear the credibility floor
                chosen.append(r)
        if not chosen:  # guard/vote too strict for this dataset -> fall back to any-member union of clusters
            chosen = list(cluster_votes.keys())
        selected = []
        for r in chosen:
            ms = [m for m in self.members_[r] if m in cols]
            if not ms:
                continue
            if self.expand_clusters:
                selected.extend(ms)
            else:
                selected.append(self._rep_member(ms))
        return selected

    def _combine_anchored(self, member_sel, cols):
        """Anchor on the MRMR/FE substrate (see anchor_fe). Keep ALL of MRMR's picks verbatim (raw + engineered),
        then ADD the clusters MRMR missed that >= ``vote`` of the OTHER members (shap/boruta) confirm. selected is a
        superset of mrmr_selected, so AUC >= mrmr_fe by construction; the other members only add complementary recall."""
        mrmr_sel = [c for c in member_sel.get("mrmr", []) if c in cols]
        selected = list(dict.fromkeys(mrmr_sel))  # FE substrate kept verbatim (engineered preserved)
        anchored_clusters = {self.cluster_of_.get(c) for c in mrmr_sel}
        others = [m for m in member_sel if m != "mrmr"]
        votes = defaultdict(set)
        for m in others:
            for f in member_sel[m]:
                r = self.cluster_of_.get(f)
                if r is not None and r not in anchored_clusters:
                    votes[r].add(m)
        for r, voters in votes.items():
            if len(voters) >= self.vote:  # add only clusters the other members confirm
                ms = [m for m in self.members_[r] if m in cols]
                if not ms:
                    continue
                if self.expand_clusters:
                    selected.extend(ms)
                else:
                    selected.append(self._rep_member(ms))
        return list(dict.fromkeys(selected))

    def transform(self, X):
        # replay FE so engineered selections (eng_N) are available, then slice to the selected set
        X = _as_pandas_view(X)
        X_aug = self._augment(X)
        keep = [c for c in self.raw_selected_ if c in X_aug.columns]
        return X_aug[keep] if keep else X_aug.iloc[:, :1]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        """Selected output feature names (raw survivors + engineered eng_N columns)."""
        return np.asarray(list(self.raw_selected_), dtype=object)

    def get_support(self, indices: bool = False):
        """Boolean mask (or integer indices) over the ORIGINAL input columns marking the selected RAW features.
        Engineered (eng_N) selections are not original columns, so they are excluded from the mask; use
        get_feature_names_out() / selected_features_ to see engineered survivors too."""
        names = list(self.feature_names_in_)
        mask = np.array([c in set(self.raw_selected_) for c in names], dtype=bool)
        return np.flatnonzero(mask) if indices else mask

    def __getstate__(self):
        """Drop the transient fit-time data (X_aug / y stashed only for combine-rule diagnostics) so a fitted
        HybridSelector pickles small and does not retain training data. The fitted MRMR member is kept (needed to
        replay feature engineering at transform time)."""
        state = dict(self.__dict__)
        state.pop("_Xaug_", None)
        state.pop("_y_", None)
        return state
