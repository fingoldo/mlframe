"""Interaction-aware coalition value for ShapProxiedFS (lever #5).

The plain coalition value ``base + sum_{j in S} phi_j`` folds each pairwise interaction into the two
features' main effects, so it cannot tell that dropping ONE partner of an interacting pair destroys
the joint signal -- and for a pure interaction (XOR) each partner's main effect is ~0, so the
informative pair looks like noise. Using SHAP *interaction* values fixes this: by the additivity of
interaction values ``phi_j = sum_k Phi_jk``, the coalition value restricted to S is

    base + sum_{i in S} sum_{k in S} Phi_ik   (interactions WITH excluded features dropped)

which correctly credits a subset for the joint signal it can actually produce. This is the user's
research ``get_total_contributions`` formula, done as the search objective.

Interaction tensors are O(P^2) memory and slower to compute, so this is opt-in and bounded to a small
proxy-column count (post-clustering / pre-screen, P is already small). We compute the tensor with one
in-sample model (a ranking refinement; the honest re-validation downstream stays OOF-disjoint), and
search with an exhaustive interaction scan for small P plus a greedy-forward pass that recovers
interacting pairs the main-effect search would miss.
"""

from __future__ import annotations

import itertools
from typing import Any, cast

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import _unwrap_estimator, _fit_one
from mlframe.feature_selection.shap_proxied_fs._shap_proxy_objective import proxy_loss, resolve_metric

# Min proxy width at which the custom numba interaction kernel is selected over the ``shap`` library.
# Measured (xgboost regr, 1000 rows, 200 trees): numba beats shap's ``shap_interaction_values`` at
# every width from P=10 (~1.5x) upward, so the interaction crossover is far LOWER than the main-effect
# one (the shap interaction call is much heavier than its main-effect call). We keep a small floor so
# the numba JIT warmup is not paid on trivial widths. Tunable per-HW via kernel_tuning_cache.
_INTERACTION_NUMBA_MIN_FEATURES = 8

# Min problem size (rows * P^2 interaction-tensor cells) at which the cupy/CUDA interaction kernel is
# preferred over numba on a CUDA box. The GPU kernel carries large per-thread local-memory scratch (it
# spills on small GPUs) so it only pays off once the per-sample tree*cond_feat work amortises the
# upload + spill; below this we stay on numba (no JIT/launch overhead win). Tunable via
# kernel_tuning_cache key ``interaction_gpu_min_cells``.
_INTERACTION_GPU_MIN_CELLS = 3_000 * 40 * 40  # ~4.8M cells (e.g. 3000 rows x P=40)


def _interaction_numba_min_features() -> int:
    """Crossover width for routing interactions to the numba kernel, from kernel_tuning_cache if set."""
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = cast(Any, ktc).lookup("shap_proxy_treeshap")
            if isinstance(entry, dict) and entry.get("interaction_numba_min_features"):
                return int(entry["interaction_numba_min_features"])
    except Exception:  # nosec B110 - best-effort path
        pass
    return _INTERACTION_NUMBA_MIN_FEATURES


def _interaction_gpu_min_cells() -> int:
    """Min ``n * P^2`` to route the interaction tensor to the GPU kernel, from kernel_tuning_cache if set."""
    try:
        from mlframe.feature_selection.filters import get_kernel_tuning_cache

        ktc = get_kernel_tuning_cache()
        if ktc is not None:
            entry = cast(Any, ktc).lookup("shap_proxy_treeshap")
            if isinstance(entry, dict) and entry.get("interaction_gpu_min_cells"):
                return int(entry["interaction_gpu_min_cells"])
    except Exception:  # nosec B110 - best-effort path
        pass
    return _INTERACTION_GPU_MIN_CELLS


def _broadcast_base(base, n: int) -> np.ndarray:
    """Return the expected-value base as a ``(n,)`` float64 vector for the coalition-margin contract.

    IX3: the in-sample single-model TreeSHAP base is a scalar today, and ``np.full(n, float(base))``
    relied on that -- it would raise (ndim>0) or silently collapse if a future kernel ever returned a
    per-row base. Handle both: a scalar broadcasts to ``(n,)``; an already per-row ``(n,)`` base is
    accepted as-is (cast + length-checked) so per-row attributions are NOT collapsed to base[0]."""
    arr = np.asarray(base, dtype=np.float64)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=np.float64)
    arr = arr.ravel()
    if arr.shape[0] == n:
        return np.ascontiguousarray(arr)
    if arr.shape[0] == 1:
        return np.full(n, float(arr[0]), dtype=np.float64)
    raise ValueError(f"interaction base has length {arr.shape[0]}, expected scalar or (n={n},)")


def _interaction_tensor_numba(est, X, *, classification):
    """Custom numba TreeSHAP interaction tensor for a supported xgboost estimator, else ``None``.

    Returns ``(Phi (n,P,P) float64, base (n,) float64)`` in margin / log-odds space, matching the shap
    library path's contract; ``None`` if the model is unsupported (caller falls back to ``shap``)."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions import interaction_tensor_numba

    base_est = _unwrap_estimator(est)
    ensemble = extract_ensemble(base_est)
    if ensemble is None:
        return None
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    Phi, _phi, base = interaction_tensor_numba(ensemble, Xv)
    return Phi, _broadcast_base(base, Phi.shape[0])


def _interaction_tensor_gpu(est, X, *, classification):
    """cupy/CUDA TreeSHAP interaction tensor for a supported xgboost estimator, else ``None``.

    Same ``(Phi (n,P,P), base (n,))`` contract as ``_interaction_tensor_numba``. Returns ``None`` (so the
    caller can fall back to numba then shap) if the model is unsupported or the ensemble exceeds the GPU
    kernel's depth/width scratch caps."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import extract_ensemble
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import interaction_tensor_gpu

    base_est = _unwrap_estimator(est)
    ensemble = extract_ensemble(base_est)
    if ensemble is None:
        return None
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    Phi, _phi, base = interaction_tensor_gpu(ensemble, Xv)
    return Phi, _broadcast_base(base, Phi.shape[0])


def compute_interaction_tensor(model_template, X, y, *, classification, rng=None, backend="auto"):
    """Fit one model on X and return ``(Phi, base)`` where Phi is (n, P, P) SHAP interaction values
    (positive class for binary) and base is the scalar expected value broadcast to (n,).

    ``backend`` ("auto" default) routes between the cupy/CUDA TreeSHAP interaction kernel (on a CUDA box
    once the problem is large enough), the custom numba TreeSHAP interaction kernel (for supported
    xgboost / lightgbm models on wider proxy widths) and the ``shap`` library (always-correct fallback).
    The GPU and
    numba kernels match each other to ~machine eps and ``shap.shap_interaction_values`` to ~1e-4; both
    are far faster on wide proxies, where the interaction tensor is the search hotspot.

    ``"shap"`` / ``"treeshap_numba"`` / ``"treeshap_gpu"`` force a path. The GPU path falls back to numba
    (depth/width cap, missing cupy or device hiccup); the numba path falls back to shap for unsupported
    models."""
    rng = np.random.default_rng(0) if rng is None else rng
    est = _fit_one(model_template, X, y, classification, int(rng.integers(0, 2**31 - 1)))

    use_gpu = False
    use_numba = False
    if backend == "treeshap_gpu":
        use_gpu = True
    elif backend == "treeshap_numba":
        use_numba = True
    elif backend == "auto":
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap import is_supported_lightgbm, is_supported_xgboost

        base_est = _unwrap_estimator(est)
        supported = is_supported_xgboost(base_est) or is_supported_lightgbm(base_est)
        P = X.shape[1]
        if supported and P >= _interaction_numba_min_features():
            use_numba = True
            # Prefer the GPU interaction kernel on a CUDA box once the tensor is large enough that the
            # per-sample tree*cond-feat work amortises the upload + local-mem spill.
            try:
                from mlframe.feature_selection.shap_proxied_fs._shap_proxy_treeshap_interactions_gpu import gpu_interactions_available

                if gpu_interactions_available() and X.shape[0] * P * P >= _interaction_gpu_min_cells():
                    use_gpu = True
            except Exception:  # nosec B110 - optional dependency import guard
                pass

    if use_gpu:
        try:
            out = _interaction_tensor_gpu(est, X, classification=classification)
            if out is not None:
                return out
        except Exception:  # nosec B110 - optional/best-effort path, rationale documented
            pass  # device/cupy/cap hiccup -> numba then shap (never lose the result)
        out = _interaction_tensor_numba(est, X, classification=classification)
        if out is not None:
            return out
    elif use_numba:
        out = _interaction_tensor_numba(est, X, classification=classification)
        if out is not None:
            return out
        # Unsupported despite routing -> fall through to shap.

    import shap

    explainer = shap.TreeExplainer(_unwrap_estimator(est), feature_perturbation="tree_path_dependent")
    Phi = explainer.shap_interaction_values(X)
    base = explainer.expected_value
    if isinstance(Phi, list):  # binary -> positive class
        was_binary_list = len(Phi) == 2  # capture BEFORE reassigning Phi (len(Phi) below would read n_rows, not class count)
        Phi = Phi[1] if was_binary_list else Phi[0]
        base = base[1] if (np.ndim(base) > 0 and was_binary_list) else (base[0] if np.ndim(base) > 0 else base)
    Phi = np.asarray(Phi, dtype=np.float64)
    if Phi.ndim == 4:  # (n, P, P, classes) -> positive class
        Phi = Phi[:, :, :, -1]
    base = float(np.asarray(base, dtype=np.float64).ravel()[0]) if np.ndim(base) > 0 else float(base)
    n = Phi.shape[0]
    return Phi, np.full(n, base, dtype=np.float64)


def interaction_margin(Phi: np.ndarray, base: np.ndarray, idx) -> np.ndarray:
    """``base + sum_{i,k in S} Phi_ik`` -- coalition value keeping only within-subset interactions."""
    idx = np.asarray(idx, dtype=np.int64)
    if idx.size == 0:
        return np.asarray(np.asarray(base).copy())
    sub = Phi[:, idx][:, :, idx]
    return np.asarray(base + sub.sum(axis=(1, 2)))


def interaction_subset_loss(Phi, base, y, idx, metric) -> float:
    return proxy_loss(interaction_margin(Phi, base, idx), y, metric)


def interaction_top_n(
    Phi, base, y, *, classification, metric=None, min_card=1, max_card=None, top_n=30,
    exhaustive_max=16,
):
    """Rank subsets by the interaction-aware coalition loss. Exhaustive for small P; always also runs
    a greedy-forward pass (which recovers interacting pairs a main-effect search would miss)."""
    metric = resolve_metric(classification, metric)
    P = Phi.shape[1]
    max_card = P if max_card is None else min(max_card, P)
    cache: dict[tuple[int, ...], float] = {}

    def loss(idx):
        key = tuple(sorted(int(i) for i in idx))
        if not key:
            return float("inf")
        v = cache.get(key)
        if v is None:
            v = interaction_subset_loss(Phi, base, y, list(key), metric)
            cache[key] = v
        return v

    if P <= exhaustive_max:
        for r in range(min_card, max_card + 1):
            for comb in itertools.combinations(range(P), r):
                loss(comb)
    # Always run greedy-forward to surface interacting pairs (cheap, dedup via cache); when P is too
    # wide to enumerate exhaustively this is the only pass that runs.
    current: tuple[int, ...] = ()
    best = float("inf")
    remaining = set(range(P))
    while remaining and len(current) < max_card:
        cand, cl = None, float("inf")
        for j in remaining:
            l = loss(current + (j,))
            if l < cl:
                cl, cand = l, j
        if cand is None or cl >= best:
            break
        current = tuple(sorted(current + (cand,)))
        best = cl
        remaining.discard(cand)

    items = [(v, k) for k, v in cache.items() if np.isfinite(v)]
    items.sort(key=lambda t: t[0])
    return items[:top_n]


# ===========================================================================================
# su_seeded_interactions (lever A4-4): a CHEAP pairwise-SU synergy screen that ranks candidate
# interaction PAIRS at O(P)+O(K) cost, then runs the interaction objective on ONLY the top-K
# synergistic pairs -- NEVER the O(P^2) TreeSHAP interaction tensor that gates ``interaction_aware``
# to <= max_interaction_features (no-op on wide proxies).
#
# WHY this works where the additive coalition proxy is blind: the additive coalition value
# ``base + sum_{j in S} phi_j`` folds each pairwise interaction into the two operands' MAIN effects.
# For a PURE interaction (y = sign(a*b), XOR, ...) each operand's marginal signal is ~0, so the
# informative pair is invisible to the main-effect SHAP search. A pairwise-SU SYNERGY score
#     synergy(a, b ; y) = SU(joint_bin(a, b) ; y) - max( SU(a ; y), SU(b ; y) )
# is HIGH exactly for such pairs (the joint bin carries about y far beyond either marginal) and ~0
# for two strong-but-non-interacting marginals (the joint adds nothing past the better marginal). The
# screen reuses the SAME discrete-SU primitive (``compute_su_from_classes``) and per-column binning
# (``_column_marginal``) the cluster-SU kernels run, so there is no new estimator machinery.
#
# SNR GATE (critical): on data where a real interaction sits BELOW the spurious-pair synergy floor
# (e.g. hard_synth's ia*ib buried in 200 noise cols) the screen MUST skip it rather than seed noise.
# We estimate the null synergy floor by a target-shuffle permutation: re-run the joint-vs-y SU on
# permuted y for a sample of pairs, take a high quantile of that null synergy, and keep only observed
# pairs whose synergy clears ``null_q + z * null_std`` (and a small absolute floor). When NOTHING
# clears the gate the path NO-OPs cleanly (no product columns engineered, no candidates merged), so
# the additive default is never regressed.
#
# bench-attempt-rejected (2026-06-04): the dense O(P^2) TreeSHAP interaction tensor
# (``compute_interaction_tensor`` + ``interaction_top_n``) is the prior ``interaction_aware`` path;
# it is gated to phi<=16 because the (n, P, P) tensor is prohibitive on wide proxies. This screen is
# the cheap replacement -- do NOT route su_seeded_interactions through the dense tensor.
# ===========================================================================================


def _su_bin(col: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin a numeric column to dense 0..K-1 int64 bin ids (constant -> all-zero one bin).

    Mirrors the discretiser the cluster-SU path uses for on-the-fly binning, but inlined so the screen
    has no dependency on the heavier ``categorize_dataset`` plumbing (it only needs a pair of columns
    at a time). Degenerate columns (<=2 distinct quantile edges) collapse to a single bin so their
    marginal entropy is 0 and they contribute SU=0.
    """
    col = np.asarray(col, dtype=np.float64)
    n = col.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    try:
        edges = np.unique(np.quantile(col, np.linspace(0.0, 1.0, n_bins + 1)))
    except Exception:
        return np.zeros(n, dtype=np.int64)
    if edges.size <= 2:
        # too few distinct values to bin meaningfully -> rank-dense fallback (handles low-card ints)
        uniq, inv = np.unique(col, return_inverse=True)
        if uniq.size <= 1:
            return np.zeros(n, dtype=np.int64)
        return inv.astype(np.int64)
    ids = np.clip(np.digitize(col, edges[1:-1]), 0, edges.size - 2).astype(np.int64)
    # densify in case some interior bins are empty
    _, ids = np.unique(ids, return_inverse=True)
    return ids.astype(np.int64)


def _su_target_bin(y: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin the target: small-cardinality (classification / few uniques) used as-is, else quantile."""
    y = np.asarray(y).ravel()
    uniq = np.unique(y)
    if uniq.size <= 20:
        # dense relabel so class labels are contiguous 0..C-1 (compute_su_from_classes indexes by id)
        _, inv = np.unique(y, return_inverse=True)
        return inv.astype(np.int64)
    return _su_bin(y.astype(np.float64), n_bins)


def _su_marginals(ids: np.ndarray):
    """``(classes_int64, freqs_float64)`` for ``compute_su_from_classes`` from dense bin ids."""
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import _column_marginal

    return _column_marginal(ids)


def su_synergy_screen(
    X,
    y,
    *,
    n_bins: int = 8,
    top_k: int = 8,
    max_screen_cols: int = 120,
    snr_z: float = 3.0,
    snr_null_quantile: float = 0.99,
    snr_abs_floor: float = 1e-3,
    n_permutations: int = 3,
    importance: np.ndarray | None = None,
    rng=None,
):
    """Rank feature PAIRS by SU synergy and return only those clearing a permutation-null SNR gate.

    Parameters
    ----------
    X : pandas.DataFrame
        The (proxy / unit) columns to screen. Only the leading ``max_screen_cols`` by ``importance``
        (or by marginal SU(.;y) when ``importance`` is None) enter the O(K^2) pair scan, so the screen
        stays O(P) marginal binning + O(min(P, max_screen_cols)^2) cheap discrete-SU pair work.
    y : array-like
        Target. Classification / low-cardinality targets are used as-is; continuous targets are
        quantile-binned to ``n_bins``.
    n_bins : int
        Quantile bins per feature column for the discrete SU estimator.
    top_k : int
        Maximum number of surviving synergistic pairs to return (after the SNR gate).
    max_screen_cols : int
        Cap on the columns entering the pair scan (bounds the O(P^2) pair count cheaply).
    snr_z, snr_null_quantile, snr_abs_floor, n_permutations :
        SNR gate. The null synergy distribution is estimated by re-scoring a sample of pairs against
        ``n_permutations`` independent target shuffles; the gate is
        ``max(snr_abs_floor, quantile(null, snr_null_quantile) + snr_z * std(null))``. A pair is kept
        iff its observed synergy clears the gate. Noise-buried interactions (synergy below the floor)
        are correctly skipped -> the caller NO-OPs.
    importance : np.ndarray | None
        Per-column ranking weight (e.g. mean |phi|) used to pick the ``max_screen_cols`` screen pool.
        ``None`` ranks by marginal SU(col; y).
    rng : np.random.Generator | None
        Source for the permutation shuffles (default: seeded 0).

    Returns
    -------
    kept : list[tuple[float, float, str, str]]
        ``(synergy, joint_su, col_a, col_b)`` for the surviving pairs, best-synergy first, len <= top_k.
        EMPTY when nothing clears the gate (the no-op signal).
    info : dict
        Diagnostics: ``gate`` (the SNR threshold), ``null_quantile``/``null_std``, ``n_screened_cols``,
        ``n_pairs``, ``best_synergy``, ``n_kept``.
    """
    from mlframe.feature_selection.filters.info_theory import compute_su_from_classes

    rng = np.random.default_rng(0) if rng is None else rng
    cols = list(X.columns)
    if len(cols) < 2:
        return [], dict(
            gate=float("inf"), n_screened_cols=len(cols), n_pairs=0, best_synergy=float("nan"), n_kept=0, null_quantile=float("nan"), null_std=float("nan")
        )

    yb = _su_target_bin(np.asarray(y), n_bins)
    y_cls, y_freq = _su_marginals(yb)

    # Per-column bins + marginals (O(P)). Reuses the cluster-SU column-marginal primitive.
    bins = {c: _su_bin(X[c].values, n_bins) for c in cols}
    nb = {c: (int(bins[c].max()) + 1 if bins[c].size else 1) for c in cols}
    marg_pack = {c: _su_marginals(bins[c]) for c in cols}
    marg_su = {}
    for c in cols:
        cls_c, freq_c = marg_pack[c]
        if freq_c.size <= 1:
            marg_su[c] = 0.0
        else:
            marg_su[c] = float(compute_su_from_classes(cls_c, freq_c, y_cls, y_freq))

    # Restrict the pair scan to the most promising columns (importance, else marginal SU).
    if importance is not None and len(importance) == len(cols):
        rank_key = {c: float(importance[i]) for i, c in enumerate(cols)}
    else:
        rank_key = marg_su
    pool = sorted(cols, key=lambda c: -rank_key.get(c, 0.0))[: int(max_screen_cols)]

    def _joint_marg(col_a, col_b):
        """``(joint_class_ids, joint_freqs)`` for the (a, b) pair WITHOUT a per-pair np.unique.

        The Cantor id ``ids_a * nb_b + ids_b`` already lives in ``[0, nb_a*nb_b)``, so bincounting to
        that fixed length (``_column_marginal`` with the hint) gives a contiguous probability vector
        directly; empty joint cells just get freq 0 and are skipped by the SU kernel. This drops the
        argsort-bound ``np.unique`` that dominated the O(P^2) pair scan (bench: ~4.6s -> ~0.5s of the
        argsort hotspot at P=120).
        """
        from mlframe.feature_selection.shap_proxied_fs._shap_proxy_cluster_su import _column_marginal

        nb_b = nb[col_b]
        raw = bins[col_a].astype(np.int64) * nb_b + bins[col_b].astype(np.int64)
        return _column_marginal(raw, n_bins_hint=nb[col_a] * nb_b)

    def _pair_synergy(col_a, col_b, target_cls, target_freq):
        jcls, jfreq = _joint_marg(col_a, col_b)
        if jfreq.size <= 1:
            return 0.0, 0.0
        joint_su = float(compute_su_from_classes(jcls, jfreq, target_cls, target_freq))
        syn = joint_su - max(marg_su[col_a], marg_su[col_b])
        return syn, joint_su

    pairs = []
    for col_a, col_b in itertools.combinations(pool, 2):
        syn, joint_su = _pair_synergy(col_a, col_b, y_cls, y_freq)
        pairs.append((syn, joint_su, col_a, col_b))
    pairs.sort(key=lambda t: -t[0])

    # --- SNR gate: target-shuffle permutation null on the synergy ----------------------------------
    # Re-score a bounded sample of pairs against shuffled targets so the null reflects the SAME joint
    # binning + cardinality as the observed pairs (a permutation of y destroys any real dependence but
    # preserves the spurious synergy a high-cardinality joint can manufacture on finite samples).
    null_vals: list[float] = []
    if pairs:
        sample_n = min(len(pairs), 40)
        sample_pairs = pairs[:sample_n]
        n = yb.shape[0]
        # Precompute each sample pair's joint marginals ONCE (they do not depend on the permutation,
        # only y is shuffled) instead of re-binning per permutation.
        sample_joint = []
        for _syn, _jsu, col_a, col_b in sample_pairs:
            jcls, jfreq = _joint_marg(col_a, col_b)
            if jfreq.size > 1:
                sample_joint.append((jcls, jfreq))
        for _ in range(max(1, int(n_permutations))):
            perm = rng.permutation(n)
            yb_perm = yb[perm]
            yp_cls, yp_freq = _su_marginals(yb_perm)
            for jcls, jfreq in sample_joint:
                joint_su_perm = float(compute_su_from_classes(jcls, jfreq, yp_cls, yp_freq))
                # marginal SU under permutation is ~0 for both operands; the spurious synergy is
                # dominated by the joint's finite-sample SU vs shuffled y, so use the joint directly
                # as the null synergy proxy (max(marg) under shuffle is also pure noise).
                null_vals.append(joint_su_perm)

    if null_vals:
        null_arr = np.asarray(null_vals, dtype=np.float64)
        null_q = float(np.quantile(null_arr, float(snr_null_quantile)))
        null_std = float(null_arr.std())
    else:
        null_q = 0.0
        null_std = 0.0
    gate = max(float(snr_abs_floor), null_q + float(snr_z) * null_std)

    kept = [(syn, jsu, a, b) for syn, jsu, a, b in pairs if syn >= gate]
    kept = kept[: int(top_k)]
    info = dict(
        gate=gate,
        null_quantile=null_q,
        null_std=null_std,
        n_screened_cols=len(pool),
        n_pairs=len(pairs),
        best_synergy=float(pairs[0][0]) if pairs else float("nan"),
        n_kept=len(kept),
    )
    return kept, info


def sparse_interaction_candidates(
    model_template,
    X_proxy,
    y,
    kept_pairs,
    *,
    classification,
    metric=None,
    min_card=1,
    max_card=None,
    top_n=30,
    rng=None,
):
    """Run the interaction objective on ONLY the ``kept_pairs`` -- a SPARSE interaction path.

    Engineers one product column ``a * b`` per surviving synergistic pair, fits ONE in-sample model on
    ``[X_proxy | products]`` and computes its (additive) SHAP matrix, then ranks candidate subsets by
    the additive coalition value over the AUGMENTED column set. Because each product column carries the
    pair's joint signal as a MAIN effect, the additive search now sees the interaction the original
    main-effect proxy was blind to -- WITHOUT ever materialising the dense O(P^2) interaction tensor
    (only ``len(kept_pairs)`` extra columns are added).

    Returns ``(candidates, product_to_operands)`` where:
      * ``candidates`` is a list of ``(loss, idx_tuple)`` in AUGMENTED proxy-column index space
        (original proxy columns keep their indices; product columns are appended at the tail), and
      * ``product_to_operands`` maps each appended product index -> ``(idx_a, idx_b)`` in original
        proxy-column space, so the caller can expand a selected product into its operand columns.

    Returns ``([], {})`` when ``kept_pairs`` is empty (the SNR-gate no-op).
    """
    from mlframe.feature_selection.shap_proxied_fs import _shap_proxy_heuristics as H
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_explain import compute_shap_matrix
    from mlframe.feature_selection.shap_proxied_fs._shap_proxy_search import brute_force_top_n

    if not kept_pairs:
        return [], {}

    rng = np.random.default_rng(0) if rng is None else rng
    base_cols = list(X_proxy.columns)
    col_to_idx = {c: i for i, c in enumerate(base_cols)}
    n_base = len(base_cols)

    aug = X_proxy.copy()
    product_to_operands: dict[int, tuple[int, int]] = {}
    prod_names: list[str] = []
    for k, pair in enumerate(kept_pairs):
        # accept (synergy, col_a, col_b) or (synergy, joint_su, col_a, col_b)
        col_a, col_b = (pair[-2], pair[-1])
        if col_a not in col_to_idx or col_b not in col_to_idx:
            continue
        nm = f"_suprod_{k}__{col_a}__x__{col_b}"
        # IX2: a single NaN/inf product (overflow on large-magnitude operands, or a NaN operand) poisons
        # the whole in-sample SHAP fit on the augmented frame. Sanitise to finite values up front so one
        # bad pair cannot take down the entire sparse-interaction ranking; nan->0, +-inf->float bounds.
        prod = X_proxy[col_a].values.astype(np.float64) * X_proxy[col_b].values.astype(np.float64)
        aug[nm] = np.nan_to_num(prod, nan=0.0, posinf=float(np.finfo(np.float64).max), neginf=float(np.finfo(np.float64).min))
        product_to_operands[n_base + len(prod_names)] = (col_to_idx[col_a], col_to_idx[col_b])
        prod_names.append(nm)

    if not prod_names:
        return [], {}

    # ONE in-sample model on the augmented frame (a ranking refinement; honest re-validation downstream
    # stays OOF-disjoint and retrains only on the ORIGINAL operand columns).
    phi_aug, base_aug, y_aug = compute_shap_matrix(
        model_template, aug, np.asarray(y), classification=classification, out_of_fold=False, n_splits=2, n_models=1, rng=rng, n_jobs=1
    )

    P = phi_aug.shape[1]
    max_card_eff = P if max_card is None else min(max_card, P)
    # Bound the additive search the same way the main optimiser does: brute force at small widths,
    # greedy-forward otherwise. We only need candidate subsets that INCLUDE the product columns to
    # surface, so greedy-forward (which adds the highest-marginal product columns first) suffices and
    # is cheap; brute force is used only when the augmented width is small.
    metric_name = resolve_metric(classification, metric)
    if metric_name == "auc":
        # the njit brute-force / greedy kernels are logloss/brier only; AUC is not rank-equivalent in
        # the hot loop. Fall back to logloss for the sparse-interaction RANKING only (the honest
        # re-validation downstream still uses the user's metric on the operand columns).
        metric = None
    if P <= 18:
        cands = brute_force_top_n(
            phi_aug, base_aug, y_aug, classification=classification, metric=metric, min_card=min_card, max_card=max_card_eff, top_n=top_n, parallel=(P >= 14)
        )
    else:
        cands = H.greedy_forward(phi_aug, base_aug, y_aug, classification=classification, metric=metric, max_card=max_card_eff, top_n=top_n)
    return cands, product_to_operands
