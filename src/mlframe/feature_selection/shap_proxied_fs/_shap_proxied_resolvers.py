"""Per-HW resolver helpers + dispatcher constants for :class:`ShapProxiedFS`.

Leaf module: the brute-force / cluster-SU / adaptive-prescreen defaults plus the
``kernel_tuning_cache``-backed resolvers that let a wider/narrower box shift the dispatcher
boundaries without code changes. No back-import to the selector parent.
"""

from __future__ import annotations

import numpy as np

_EXACT_OPTIMIZERS = {"bruteforce", "bruteforce_gpu"}
_HEURISTIC_OPTIMIZERS = {"beam", "greedy_forward", "greedy_backward", "multistart", "genetic", "annealing", "gradient"}

# Brute-force dispatcher gates (iter56 + iter57 audit). Both are overridable per HW via
# ``pyutilz.performance.kernel_tuning.cache`` so a wider/narrower box can shift the boundary without
# touching code.
#
# Two distinct knobs, two distinct effects:
#
# 1. ``brute_force_max_features`` (default 28): cap on ``phi.shape[1]`` AFTER prescreen. The
#    prescreen step (shap_proxied_fs.py near line 866) narrows the candidate pool to this many
#    columns whenever the user runs ``optimizer="auto"|"bruteforce"|"bruteforce_gpu"``. The pool
#    feeds whichever optimizer the dispatcher ends up picking - brute_force when feasible, beam
#    otherwise - so this knob widens or narrows the CANDIDATE SPACE all optimizers see, not just
#    brute force.
#
# 2. ``brute_force_n_sub_gate`` (default 80M): cap on the EXHAUSTIVE subset count
#    ``total_subsets(n, min_card, max_card)`` that brute force would enumerate. The dispatcher
#    uses this to decide whether brute_force is feasible AT the post-prescreen n. When it isn't,
#    the dispatcher falls back to beam.
#
# Default-config behaviour at ``max_features=None`` (the default):
#   total_subsets(n, 1, None) = 2^n - 1 (kernel treats None as n_features).
#   - n in {1..26}: 2^n - 1 <= 67M, under the 80M gate -> brute force dispatches.
#   - n in {27, 28}: 134M, 268M, over the 80M gate -> beam dispatches.
#
# So at ``max_features=None`` the EFFECTIVE brute-force ceiling is n=26, NOT n=28. The cap of 28
# only unlocks the brute-force path when the user ALSO pins ``max_features<=12`` (sum C(28, 1..12)
# = 76.7M, under the 80M gate). At n=27,28 with default ``max_features=None`` the cap acts as a
# prescreen-pool widener that beam consumes - iter56's measured recall/wall gain came from beam at
# the wider pool, NOT from brute force. The cap is named after the brute-force kernel because
# that is the optimizer the dispatcher PREFERS at small n; at n=27,28 with default max_features
# the dispatcher correctly falls back to beam over the 28-column prescreen pool.
#
# Sizing rationale for the gate: 80M at the iter30 parallel kernel (~5M subsets/s on the 8-core
# dev box) caps a single brute-force search at ~16s wall. The next power-of-2 (n=28 with
# max_features=None: 268M subsets, ~54s wall) is beyond what the dispatcher should pick without
# the user explicitly opting in via ``optimizer="bruteforce"``.
#
# Iter58 beam-width sweep (``_benchmarks/bench_iter58_beam_width_sweep.py``) measured caps
# {22, 28, 32, 40} at C3 (snr=8) and C3_hard (snr=2). cap28 recall-dominates or ties all wider
# caps in both regimes; cap32 LOST a recall hit at C3_hard (15/20 vs cap28's 16/20), and cap40
# was slower without improving recall. Hypothesis: widening the prescreen pool past 28 lets
# noise features into beam's input, occasionally pushing the chosen subset off the truly
# informative one when the SHAP ranking is noisy. Default stays at 28.
_DEFAULT_BRUTE_FORCE_MAX_FEATURES = 28
_DEFAULT_BRUTE_FORCE_N_SUB_GATE = 80_000_000

# iter75: when auto-mode picks the SU clustering backend without precomputed bins, ShapProxiedFS
# bins X on-the-fly via MRMR's ``categorize_dataset`` and then runs pairwise SU. The pairwise scan
# is O(f^2) and despite the iter67-74 speedups (33x cumulative at width=2000) it is still slower
# than Pearson's vectorised correlation matrix at very wide search widths; above the cap the
# auto-mode falls back to Pearson rather than pay the SU cost. 2000 is the calibration point at
# which the iter69 column-major + iter71 fused-setup + iter73 popcount kernels keep SU within ~3x
# of Pearson on the dev box; per-HW tunable via kernel_tuning_cache key
# ``mlframe.shap_proxied_fs.cluster_su_auto_max_features``.
_DEFAULT_CLUSTER_SU_AUTO_MAX_FEATURES = 2000

# Iter59 adaptive-prescreen-width thresholds. The lever is a recall-protection device for low-SNR
# regimes: at low SHAP rank-stability across OOF folds, the top features past the strongly-informative
# core are essentially noise -- pulling them into the prescreen pool injects noise into beam's input
# and can perturb the chosen subset off the truly informative one. So we NARROW (never widen) the
# prescreen cap when stability drops. High-stability regimes keep the existing cap untouched.
#
# Thresholds (stability = median pairwise Spearman of per-fold mean |phi| feature rankings):
#   stability >= 0.8  -> use default cap (current behaviour, no regression risk)
#   0.6 <= stability < 0.8 -> cap = max(20, default - 4)  (mild narrow)
#   stability < 0.6   -> cap = max(16, default - 8)       (aggressive narrow)
#
# Overridable per-HW via kernel_tuning_cache key ``mlframe.shap_proxied_fs.adaptive_prescreen_stability_thresholds``
# which accepts a list of ``[stability_threshold, cap_delta]`` pairs sorted descending by threshold.
_DEFAULT_ADAPTIVE_PRESCREEN_THRESHOLDS = (
    (0.8, 0),  # stability >= 0.8: no narrowing, keep default cap
    (0.6, -4),  # 0.6 <= stability < 0.8: narrow by 4
    (-1.0, -8),  # stability < 0.6: narrow by 8 (catches negative correlations too)
)
_ADAPTIVE_PRESCREEN_FLOOR = 16  # never narrow below this regardless of stability
# Knee-ladder gate: minimum peak gap above the diagonal of the cumulative-importance curve before the
# knee narrows the cap. Below this the importance is spread evenly enough (dense signal) that the full
# cap is kept. Calibrated so a uniform curve (gap ~0) keeps the cap and a clearly front-loaded curve
# (top-k dominant, gap >> 0) narrows. Per-HW tunable via kernel_tuning_cache is unnecessary (pure
# distribution shape, hardware-independent); exposed as a module constant for bench recalibration.
_KNEE_SPARSITY_GATE = 0.10


# Adaptive trust-guard anchor budget. The fixed 30 anchors are sparse on very wide search frames
# (p>>): with a fixed budget the per-anchor coverage of a p=10000 column space is the same as a p=20
# space, so the trust signal weakens exactly where the proxy is least trustworthy. The adaptive count
# scales sub-linearly with the post-prefilter search width so wide frames get a denser guard while
# narrow frames keep a cheap budget: n_anchors = clip(round(c*sqrt(p)), lo, hi). sqrt keeps the cost
# bounded (p=10000 -> ~6*100 capped at 100, not 10000). c=6.0 reproduces ~30 at the p=25 small-frame
# regime the fixed default was tuned on (6*5=30) and lifts to the 100 ceiling for p>=278. An explicit
# integer ``n_anchors=`` always overrides. Per-HW tunable via kernel_tuning_cache key
# ``mlframe.shap_proxied_fs.adaptive_n_anchors_params`` -> [c, lo, hi].
_DEFAULT_ADAPTIVE_N_ANCHORS_C = 6.0
_ADAPTIVE_N_ANCHORS_LO = 10
_ADAPTIVE_N_ANCHORS_HI = 100


def _resolve_adaptive_n_anchors(
    n_search_cols: int, *, c: float = _DEFAULT_ADAPTIVE_N_ANCHORS_C, lo: int = _ADAPTIVE_N_ANCHORS_LO, hi: int = _ADAPTIVE_N_ANCHORS_HI
) -> int:
    """Self-tuning anchor count for the trust guard: ``clip(round(c*sqrt(p)), lo, hi)``.

    ``p`` is the post-prefilter proxy search width (``phi.shape[1]``). Reads override params from
    kernel_tuning_cache key ``mlframe.shap_proxied_fs.adaptive_n_anchors_params`` as ``[c, lo, hi]``.
    """
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        params = kernel_tuning_cache.get("mlframe.shap_proxied_fs.adaptive_n_anchors_params", default=None)
        if params:
            c, lo, hi = float(params[0]), int(params[1]), int(params[2])
    except Exception:
        pass
    p = max(1, int(n_search_cols))
    n = int(round(float(c) * float(np.sqrt(p))))
    return int(min(int(hi), max(int(lo), n)))


def _resolve_knee_prescreen_cap(importance, default_cap: int, *, floor: int = _ADAPTIVE_PRESCREEN_FLOOR) -> tuple[int, dict]:
    """Data-driven prescreen cap from the sorted |phi| importance curve (knee detection).

    Replaces the hardcoded stability ladder with a distribution read: a DENSE-signal frame (many
    columns carry comparable importance) keeps the full ``default_cap``; a SPARSE-signal frame (a few
    dominant columns, long noise tail) prunes harder. The knee is the kneedle-style point of maximum
    drop below the diagonal of the normalised cumulative-importance curve over the top ``default_cap``
    candidates. ``cap = clip(knee+1, floor, default_cap)`` -- this lever only ever NARROWS, mirroring
    the stability ladder's contract, so a dense frame is never penalised. Returns (cap, info).
    """
    imp = np.sort(np.abs(np.asarray(importance, dtype=np.float64)))[::-1]
    head = imp[: int(default_cap)]
    info = dict(mode="knee", default_cap=int(default_cap))
    if head.shape[0] < 3 or not np.isfinite(head).all() or head[0] <= 0:
        info["effective_cap"] = int(default_cap)
        info["knee"] = None
        return int(default_cap), info
    # Normalised cumulative-importance curve; concave for sparse signal (mass front-loaded).
    cum = np.cumsum(head)
    cum = cum / cum[-1]
    x = np.arange(1, head.shape[0] + 1, dtype=np.float64) / head.shape[0]
    # Distance above the y=x diagonal; the max-distance index is the kneedle knee.
    diff = cum - x
    knee_idx = int(np.argmax(diff))
    # Sparsity score: peak gap above the diagonal. ~0 => uniform/dense importance (mass spread evenly
    # across the cap) => KEEP the full default cap; large => front-loaded/sparse => narrow to the knee.
    sparsity = float(diff[knee_idx])
    if sparsity < _KNEE_SPARSITY_GATE:
        info.update(effective_cap=int(default_cap), knee=int(knee_idx), sparsity=sparsity)
        return int(default_cap), info
    cap = int(min(int(default_cap), max(int(floor), knee_idx + 1)))
    info.update(effective_cap=int(cap), knee=int(knee_idx), sparsity=sparsity)
    return cap, info


def _resolve_brute_force_max_features(default: int = _DEFAULT_BRUTE_FORCE_MAX_FEATURES) -> int:
    """Per-HW brute-force cap from ``pyutilz.performance.kernel_tuning.cache`` (key
    ``mlframe.shap_proxied_fs.brute_force_max_features``), falling back to the module default."""
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        value = kernel_tuning_cache.get("mlframe.shap_proxied_fs.brute_force_max_features", default=default)
        return int(value)
    except Exception:
        return default


def _resolve_brute_force_n_sub_gate(default: int = _DEFAULT_BRUTE_FORCE_N_SUB_GATE) -> int:
    """Per-HW feasibility cap on enumerated subset count (key
    ``mlframe.shap_proxied_fs.brute_force_n_sub_gate``). Above this the dispatcher falls through
    to ``beam`` regardless of ``brute_force_max_features``."""
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        value = kernel_tuning_cache.get("mlframe.shap_proxied_fs.brute_force_n_sub_gate", default=default)
        return int(value)
    except Exception:
        return default


def _resolve_cluster_su_auto_max_features(
    default: int = _DEFAULT_CLUSTER_SU_AUTO_MAX_FEATURES,
) -> int:
    """Per-HW upper width at which ``cluster_backend='auto'`` still picks SU when no precomputed
    bins are supplied. Above this the auto path falls back to Pearson (the pairwise SU O(f^2)
    scan no longer amortises the iter73-era 33x-over-naive cost into a wall-clock win vs the
    vectorised Pearson |corr|). Reads kernel_tuning_cache key
    ``mlframe.shap_proxied_fs.cluster_su_auto_max_features``."""
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        value = kernel_tuning_cache.get("mlframe.shap_proxied_fs.cluster_su_auto_max_features", default=default)
        return int(value)
    except Exception:
        return default


def _resolve_adaptive_prescreen_thresholds():
    """Return the (stability, delta) threshold list, ordered descending by stability.

    Reads ``mlframe.shap_proxied_fs.adaptive_prescreen_stability_thresholds`` from kernel_tuning_cache
    when present (expected as an iterable of (stability, delta) pairs). Falls back to the module
    default. Always coerced to a tuple of (float, int) pairs sorted by descending stability.
    """
    raw = _DEFAULT_ADAPTIVE_PRESCREEN_THRESHOLDS
    try:
        from pyutilz.performance.kernel_tuning import cache as kernel_tuning_cache

        cached = kernel_tuning_cache.get("mlframe.shap_proxied_fs.adaptive_prescreen_stability_thresholds", default=None)
        if cached:
            raw = cached
    except Exception:
        pass
    pairs = [(float(s), int(d)) for s, d in raw]
    pairs.sort(key=lambda p: -p[0])
    return tuple(pairs)


def _resolve_adaptive_prescreen_width(stability: float, default_cap: int, floor: int = _ADAPTIVE_PRESCREEN_FLOOR) -> int:
    """Resolve the prescreen pool width from the measured cross-fold SHAP rank stability.

    Returns ``max(floor, default_cap + delta)`` where ``delta`` is read from the first threshold
    matching ``stability >= threshold`` in the descending-sorted table. The default table never adds
    a positive delta, so this lever can only NARROW the pool, never widen it past ``default_cap``.
    Conservative by design: high-stability regimes (the existing working configurations) keep the
    current cap untouched, and only the low-SHAP-rank-stability case (where the rank tail past the
    strongly-informative core is noise) sees a narrower pool that excludes that noise tail.
    """
    table = _resolve_adaptive_prescreen_thresholds()
    delta = 0
    for thr, d in table:
        if stability >= thr:
            delta = d
            break
    return max(int(floor), int(default_cap) + int(delta))
