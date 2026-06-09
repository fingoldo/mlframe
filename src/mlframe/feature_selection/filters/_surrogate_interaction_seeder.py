"""Surrogate-GBM split-co-occurrence interaction seeder (FE/MRMR backlog idea #6).

The univariate-MI ``seed_count`` gate that the prospective-pair sweep and the
triplet/quadruplet FE stages use to pick source columns is BLIND to pure synergy:
a zero-marginal interaction operand (``y = sign(x_a * x_b * x_c) + noise`` -- every
marginal MI ~= 0) is never ranked top-N, so the pair is never enumerated and the
triple is never seeded. The interaction needle is MISSED.

This proposer fits ONE fast shallow gradient-boosted surrogate (LightGBM, ~100
depth-3 trees) on the already-discretised screening matrix, then walks every
root-to-leaf path and tallies a DEPTH-DISCOUNTED SPLIT-GAIN co-occurrence weight for
every (a, b) PAIR and (a, b, c) TRIPLE that co-occur on a path. A zero-marginal
synergy operand still appears as a split partner CONDITIONED on its co-splitter on
the tree path, so co-occurrence ranks the true interaction members at the top even
though their marginals are ~0 -- reaching the prospective pool / ``triplets=`` where
``seed_count`` never would. 3-way is FREE via path co-occurrence. Cost is
``O(n * trees * depth)`` to fit + ``O(trees * depth^2)`` to tally -- INDEPENDENT of
``p^2`` / ``p^3``, the scaling lever for large-p frames.

SELF-GATE (the proposer only GENERATES; downstream maxT floors gate admission). The
seeder emits NOTHING unless the surrogate carries genuine joint signal: its
out-of-fold score must beat a PERMUTED-y surrogate baseline (same hyper-params, y
shuffled). On pure noise the real and permuted OOF scores tie, the gate fails, and the
pool is not polluted -- the order-2 / order-3 Westfall-Young maxT floors then gate every
emitted candidate as the outer best-of-pool guard.

Integration: called at the top of ``MRMR._run_fe_step`` to populate ``_seeded_pairs``
(merged into the prospective pool, bypassing the univariate seed_count) and
``_seeded_triplets`` (fed to the order-3-floored triplet FE). Opt-in /
fastest-default-routed by ``fe_gbm_seeder_enable`` + the ``(n, p)`` cost gate below.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

import numpy as np

logger = logging.getLogger("mlframe.feature_selection.filters.mrmr")


def _walk_paths_tally(
    node,
    path,
    pair_w: dict,
    triple_w: dict,
    feat_w: dict,
    depth_discount: float,
) -> None:
    """Recursively walk a LightGBM ``tree_structure`` dict, accumulating
    depth-discounted split-gain co-occurrence weights into ``pair_w`` / ``triple_w``
    (and per-feature into ``feat_w``).

    ``path`` is the list of ``(feature, weight)`` splits from the root to the current
    internal node. At each internal split we add ``weight = split_gain * depth_discount ** depth``
    so a deep, low-gain split contributes less than a shallow high-gain one, then for
    every ANCESTOR split already on the path we credit the (ancestor, current) PAIR and
    every (ancestor_i, ancestor_j, current) TRIPLE with the MIN of the two/three split
    weights (a conjunction is only as strong as its weakest leg -- a pair co-occurs only
    where BOTH splits are taken). Leaves carry no split and terminate the recursion.
    """
    if not isinstance(node, dict) or "split_feature" not in node:
        return  # leaf
    feat = int(node["split_feature"])
    depth = len(path)
    gain = float(node.get("split_gain", 0.0) or 0.0)
    if gain < 0.0:
        gain = 0.0
    w = gain * (depth_discount ** depth)
    feat_w[feat] = feat_w.get(feat, 0.0) + w

    # Pairs: (ancestor, current). Triples: (ancestor_i, ancestor_j, current).
    for i, (fa, wa) in enumerate(path):
        if fa != feat:
            key = (fa, feat) if fa < feat else (feat, fa)
            pair_w[key] = pair_w.get(key, 0.0) + min(wa, w)
        for fb, wb in path[i + 1:]:
            trio = tuple(sorted({fa, fb, feat}))
            if len(trio) == 3:
                triple_w[trio] = triple_w.get(trio, 0.0) + min(wa, wb, w)

    child_path = path + [(feat, w)]
    _walk_paths_tally(node["left_child"], child_path, pair_w, triple_w, feat_w, depth_discount)
    _walk_paths_tally(node["right_child"], child_path, pair_w, triple_w, feat_w, depth_discount)


def _fit_surrogate_and_oof(
    disc_X: np.ndarray,
    y: np.ndarray,
    *,
    is_classification: bool,
    n_estimators: int,
    max_depth: int,
    num_leaves: int,
    learning_rate: float,
    min_data_in_leaf: int,
    n_threads: int,
    random_seed: int,
    shuffle_y: bool,
):
    """Fit one shallow LightGBM surrogate with a single 70/30 ordered OOF split and
    return ``(booster, oof_score)``. ``oof_score`` is OOF accuracy (classification) or
    OOF R^2 (regression) -- the self-gate statistic compared against the permuted-y run.
    When ``shuffle_y`` the labels are permuted (the permuted-y baseline). Returns
    ``(None, nan)`` if LightGBM is unavailable or the fit degenerates."""
    try:
        import lightgbm as lgb
    except Exception:
        return None, float("nan")

    n = disc_X.shape[0]
    rng = np.random.default_rng(random_seed)
    yv = np.asarray(y).copy()
    if shuffle_y:
        rng.shuffle(yv)

    # Deterministic 70/30 split (ordered, mirrors the cheap %-stride splits used
    # elsewhere in the FE gates); the OOF score is the honest held-out estimate.
    idx = rng.permutation(n)
    cut = int(n * 0.7)
    tr, te = idx[:cut], idx[cut:]
    if len(tr) < max(2 * min_data_in_leaf, 20) or len(te) < 10:
        return None, float("nan")

    Xtr, Xte = disc_X[tr], disc_X[te]
    ytr, yte = yv[tr], yv[te]

    params = dict(
        num_leaves=int(num_leaves),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        min_data_in_leaf=int(min_data_in_leaf),
        verbose=-1,
        num_threads=int(n_threads),
        deterministic=True,
        force_row_wise=True,
        seed=int(random_seed),
        feature_pre_filter=False,
    )
    if is_classification:
        n_cls = int(np.unique(yv).size)
        if n_cls < 2:
            return None, float("nan")
        if n_cls == 2:
            params["objective"] = "binary"
        else:
            params["objective"] = "multiclass"
            params["num_class"] = n_cls
    else:
        params["objective"] = "regression"

    try:
        ds = lgb.Dataset(Xtr, label=ytr, free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=int(n_estimators))
        pred = booster.predict(Xte)
        if is_classification:
            if pred.ndim == 2:
                yhat = np.argmax(pred, axis=1)
            else:
                yhat = (pred >= 0.5).astype(yte.dtype)
            score = float(np.mean(yhat == yte))
        else:
            ss_res = float(np.sum((yte - pred) ** 2))
            ss_tot = float(np.sum((yte - np.mean(yte)) ** 2)) + 1e-12
            score = 1.0 - ss_res / ss_tot
        return booster, score
    except Exception:
        logger.warning("GBM surrogate seeder: LightGBM fit failed; emitting nothing.", exc_info=True)
        return None, float("nan")


def surrogate_gbm_interaction_seeds(
    disc_X: np.ndarray,
    y: np.ndarray,
    candidate_indices,
    *,
    is_classification: bool,
    top_k_pairs: int = 12,
    top_k_triples: int = 8,
    # DEPTH-4 default (NOT the backlog's "depth-3" first guess): a depth-3 axis-aligned
    # tree can only FORM a 3-way conjunction at the leaf with NO purity margin, so the
    # split gains on the operands of a hard 3-way needle (``sign(a*b*c)``) stay buried in
    # noise and the true triple never ranks. Depth 4 gives the tree a 4th split to clean up
    # the 3-way cell, lifting the operand split-gains above chance. Bench (D:/Temp/tune_3way.py,
    # n=4000 p=200 sign-product 3-way): md=3 -> needle-triple rank MISS (op-recall 0.67);
    # md=4 nl=16 ne=150 -> needle triple is the RANK-0 co-occurrence triple (op-recall 1.00);
    # md=5+ OVER-fragments (op-recall drops to 0.33) -- so 4 is the sweet spot, NOT "deeper is
    # better". The 2-way needle + pure-noise self-gate are unaffected by the depth choice.
    n_estimators: int = 150,
    max_depth: int = 4,
    num_leaves: int = 16,
    learning_rate: float = 0.1,
    min_data_in_leaf: int = 20,
    depth_discount: float = 0.5,
    self_gate_margin: float = 0.0,
    n_threads: int = 4,
    random_seed: int = 0,
) -> tuple[list, list, dict]:
    """Propose interaction PAIR + TRIPLE seeds from a surrogate-GBM's split co-occurrence.

    ``disc_X`` is the ordinal-encoded screening matrix (n, p); ``candidate_indices`` the
    COLUMN INDICES (into ``disc_X``) eligible to seed (the same numeric operand pool the
    pair sweep considers -- so seeds map back to real operands). ``y`` is the ordinal
    target. Returns ``(seeded_pairs, seeded_triplets, info)``:

      * ``seeded_pairs``    -- list of ``(a, b)`` index tuples, top-``top_k_pairs`` by
        depth-discounted split-gain co-occurrence (a < b).
      * ``seeded_triplets`` -- list of ``(a, b, c)`` index tuples, top-``top_k_triples``.
      * ``info``            -- diagnostics dict (``oof_real``, ``oof_perm``, ``gated``,
        ``n_pairs``, ``n_triples``, plus the raw weight maps for provenance).

    SELF-GATE: returns EMPTY pair/triple lists (``gated=False``) unless the real OOF score
    beats the permuted-y OOF baseline by > ``self_gate_margin`` -- on pure noise the two
    tie, so nothing is emitted and the pool is not polluted. Only ``candidate_indices``
    columns are eligible to be seeded (splits on a non-candidate column are ignored)."""
    info: dict = {"oof_real": float("nan"), "oof_perm": float("nan"), "gated": False,
                  "n_pairs": 0, "n_triples": 0}
    cand_set = set(int(c) for c in candidate_indices)
    if len(cand_set) < 2 or disc_X.shape[0] < 30:
        return [], [], info

    booster, oof_real = _fit_surrogate_and_oof(
        disc_X, y, is_classification=is_classification, n_estimators=n_estimators,
        max_depth=max_depth, num_leaves=num_leaves, learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf, n_threads=n_threads, random_seed=random_seed,
        shuffle_y=False,
    )
    if booster is None or not np.isfinite(oof_real):
        return [], [], info
    _, oof_perm = _fit_surrogate_and_oof(
        disc_X, y, is_classification=is_classification, n_estimators=n_estimators,
        max_depth=max_depth, num_leaves=num_leaves, learning_rate=learning_rate,
        min_data_in_leaf=min_data_in_leaf, n_threads=n_threads, random_seed=random_seed + 1,
        shuffle_y=True,
    )
    info["oof_real"] = float(oof_real)
    info["oof_perm"] = float(oof_perm) if np.isfinite(oof_perm) else float("nan")

    # SELF-GATE: real surrogate must beat the permuted-y baseline. If the permuted run
    # failed (nan), require a positive absolute OOF (R^2 > 0 / acc above a degenerate
    # baseline) so a degenerate permuted run does not auto-pass a noise frame.
    perm_ref = info["oof_perm"] if np.isfinite(info["oof_perm"]) else (0.0 if not is_classification else None)
    if perm_ref is None:
        # classification with failed permuted run: fall back to majority-class rate.
        vals, cnts = np.unique(np.asarray(y), return_counts=True)
        perm_ref = float(cnts.max() / cnts.sum())
    if not (oof_real > perm_ref + float(self_gate_margin)):
        logger.info(
            "GBM surrogate seeder SELF-GATE failed: OOF real=%.4f <= permuted/baseline=%.4f (+margin %.3f); "
            "emitting NO seeds (no genuine joint signal -> pool not polluted).",
            oof_real, perm_ref, self_gate_margin,
        )
        return [], [], info
    info["gated"] = True

    pair_w: dict = {}
    triple_w: dict = {}
    feat_w: dict = {}
    md = booster.dump_model()
    for tinfo in md.get("tree_info", []):
        ts = tinfo.get("tree_structure")
        if ts is not None:
            _walk_paths_tally(ts, [], pair_w, triple_w, feat_w, depth_discount)

    # Restrict to candidate operands on BOTH legs/all-three legs.
    pair_w = {k: v for k, v in pair_w.items() if k[0] in cand_set and k[1] in cand_set and v > 0.0}
    triple_w = {k: v for k, v in triple_w.items()
                if all(i in cand_set for i in k) and v > 0.0}

    seeded_pairs = [k for k, _ in sorted(pair_w.items(), key=lambda kv: kv[1], reverse=True)[:int(top_k_pairs)]]
    seeded_triplets = [k for k, _ in sorted(triple_w.items(), key=lambda kv: kv[1], reverse=True)[:int(top_k_triples)]]
    info["n_pairs"] = len(seeded_pairs)
    info["n_triples"] = len(seeded_triplets)
    info["pair_weights"] = pair_w
    info["triple_weights"] = triple_w
    info["feature_weights"] = feat_w
    logger.info(
        "GBM surrogate seeder: OOF real=%.4f > permuted=%.4f -> emitting %d pair + %d triple "
        "co-occurrence seed(s) (top by depth-discounted split-gain), bypassing the univariate "
        "seed_count gate.",
        oof_real, perm_ref, len(seeded_pairs), len(seeded_triplets),
    )
    return seeded_pairs, seeded_triplets, info
