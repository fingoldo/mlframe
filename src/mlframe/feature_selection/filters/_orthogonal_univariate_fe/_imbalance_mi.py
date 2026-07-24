"""Confidence-weighted / class-balanced MI for FE relevance under class imbalance (backlog idea #18).

bench-rejected (2026-06-10): default OFF. The premise -- that plain plug-in
``MI(candidate; y)`` under-RANKS a feature that separates the rare class -- does
NOT hold for the estimator as implemented. Inverse-prior class balancing applies
a near-uniform MULTIPLICATIVE rescale to every feature's MI (Kendall tau 0.989 vs
plain across 40 imbalanced frames; min 0.956), so it almost never changes a
rank-based selection. Over 120 imbalanced frames (n=20000, prior ~0.5-6%) the
balanced top-5 selection differed from plain in only 13; across those 13 flips
the downstream rare-class metrics were a coin-flip and net NEGATIVE: mean dAP
-0.0037 (improved 6/13), mean drecall@3% -0.0122. The stratified-balanced-
subsample alternative reorders more (63/120) only because subsampling injects
estimator variance, and is also net-negative downstream (mean tree-dAP -0.0027,
improved 26/63). So the correction does not recover rare-class-discriminative
features and does not improve a downstream classifier's rare-class recall/AP.
Mechanistically: a high-precision rare detector ALWAYS carries higher plain MI
than a broad/weak majority-correlated decoy (verified analytically for binary
features), so plain MI compresses the rare signal's MAGNITUDE (a 0.78-AUC rare
separator -> plain MI ~0.004 vs balanced ~0.13, ~30x) but preserves its RANK and
the eng/raw uplift SIGN -- and rank/ratio gates are invariant to the uniform
rescale. Kept here (keep-all-kernels rule) as an opt-in scorer for ad-hoc
imbalance experiments, reachable via ``MLFRAME_FE_IMBALANCE_MI=on``. Numbers:
``D:/Temp/imbalmi_results.md``; regression coverage:
``tests/feature_selection/test_imbalance_mi.py``.

Mechanism (when forced on). Plain plug-in MI is dominated by the majority class:
when the positive prior is far below 0.5, a feature that separates the RARE class
contributes little to the marginal MI (the rare class carries little probability
mass). When the minority prior is below a threshold AND there are enough rare-
class samples to estimate the rebalanced histogram reliably (gate on ``n_rare``,
NOT just the prior -- too few rare rows = an unreliable rebalanced estimate, fall
back to plain MI), compute a **class-balanced MI**: inverse-prior reweight the
joint histogram so every class contributes equal total probability mass to the MI
sum.

The gate is two-sided inert:
  - balanced data (prior ~0.5)             -> weights are ~uniform -> MI == plain MI;
  - imbalanced but n_rare below the floor  -> fall back to plain MI (byte-identical);
  - imbalanced AND n_rare >= floor         -> class-balanced MI (the correction).

Default ``off`` (the bench-rejection above). Override via
``MLFRAME_FE_IMBALANCE_MI`` (``auto`` / ``on`` / ``off``):
  - ``off``  (default) -> always plain MI;
  - ``on``   -> force the prior/n_rare gate + balanced MI when it fires;
  - ``auto`` -> same gate as ``on`` (opt-in alias; NOT wired as the default).
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np

logger = logging.getLogger(__name__)

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is a hard dep in this repo
    _NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        """No-op ``numba.njit`` stand-in for the (hard-dep-violating) environment without numba: returns the function unchanged whether called bare (``@njit``) or with kwargs (``@njit(cache=True)``)."""
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def deco(fn):
            """Identity decorator used when ``njit`` is invoked with arguments."""
            return fn
        return deco

    def prange(n):
        """Plain-Python fallback for ``numba.prange`` when numba is unavailable: behaves as ``range``."""
        return range(n)


# --- Gate thresholds -------------------------------------------------------
# Chosen per the idea + the "rare-imbalance-needs-large-n" memory:
#  * PRIOR threshold: only correct when the minority class is genuinely a
#    minority. 0.30 = "materially imbalanced"; above it the plain-MI majority
#    domination is mild and the reweight buys nothing (and risks inflating the
#    rare class's finite-sample MI bias).
#  * N_RARE floor: the rebalanced histogram up-weights the rare class by
#    ~1/prior; with too few rare rows that estimate is high-variance and would
#    promote chance-separating features. 150 rare rows is the documented
#    "enough to estimate" floor (the WIN fixture has ~200 positives at 1%; the
#    GATE-control fixture has ~20 and must fall back). Overridable via env.
_PRIOR_THRESHOLD = float(os.environ.get("MLFRAME_FE_IMBALANCE_PRIOR", "0.30"))
_N_RARE_FLOOR = int(os.environ.get("MLFRAME_FE_IMBALANCE_N_RARE", "150"))


def _imbalance_mode() -> str:
    """``off`` (DEFAULT -- bench-rejected) | ``on`` / ``auto`` (opt-in: force the gate).

    Default is ``off`` so ``_mi_classif_batch`` is byte-for-byte the plain-MI path
    unless a user explicitly opts in (``MLFRAME_FE_IMBALANCE_MI=on``). ``auto`` is
    kept as an alias for ``on`` (the gate still self-disables on balanced data and
    below the n_rare floor) but is NOT the unset default -- see the module
    docstring for the rejection numbers.
    """
    flag = os.environ.get("MLFRAME_FE_IMBALANCE_MI", "").strip().lower()
    if flag in ("1", "true", "on", "yes"):
        return "on"
    if flag in ("auto",):
        return "on"
    # unset / "off" / "0" / anything else -> plain MI (bench-rejected default)
    return "off"


def compute_class_weights(y: np.ndarray) -> np.ndarray | None:
    """Decide whether to class-balance MI for this ``y``; return per-class weights or ``None``.

    Returns ``None`` (=> caller uses plain MI, byte-identical) when:
      * the override is ``off`` (the DEFAULT -- bench-rejected; the common path);
      * y is not a discrete classification target with >=2 populated classes;
      * the minority prior is above ``_PRIOR_THRESHOLD`` -- not imbalanced
        enough to bother (the no-regression / balanced-data case);
      * the rare-class count is below ``_N_RARE_FLOOR`` -- too few rare rows
        to estimate a reliable rebalanced histogram (the GATE-control case).

    When it DOES correct (only under the opt-in ``on``/``auto`` mode), returns
    ``w`` of shape ``(n_classes,)`` with ``w[c] = (1/K) / p(c)`` so every class
    contributes equal total weight ``1/K`` to the reweighted joint distribution
    (inverse-prior balancing). On balanced data ``w`` is ~uniform => reweighted
    MI == plain MI, which is why the correction is inert when not needed.
    """
    mode = _imbalance_mode()
    if mode == "off":
        return None

    y_arr = np.asarray(y)
    n = y_arr.shape[0]
    if n == 0:
        return None
    # Must be an integer-labelled discrete target. Non-integer / float y is a
    # regression target routed elsewhere; never reweight it.
    if not np.issubdtype(y_arr.dtype, np.integer):
        return None
    y_min = int(y_arr.min())
    if y_min < 0:
        return None
    n_classes = int(y_arr.max()) + 1
    if n_classes < 2:
        return None
    # X_EDGE_CASES_BEST_PRACTICES-3 fix: no upper bound existed on n_classes before
    # allocating `counts` -- a caller-controlled integer y whose max value is large (a row-id/timestamp
    # column mistakenly typed/passed as a classification target, or genuinely tens of millions of sparse
    # integer "classes") allocates a np.bincount array of that size. This module is opt-in
    # (MLFRAME_FE_IMBALANCE_MI) but the allocation-size gap is the same class INFO_THEORY_A-2/B-7 flagged
    # for sibling dense-histogram builders. A real classification target essentially never exceeds a few
    # thousand distinct classes; reject clearly rather than silently allocating gigabytes.
    _MAX_CLASSES = 100_000
    if n_classes > _MAX_CLASSES:
        logger.warning(
            "compute_class_weights: y has %d distinct integer values (max=%d) -- treating as NOT a "
            "classification target (likely a mistyped continuous/id column) and skipping reweighting. "
            "If this is genuinely a %d-class problem, this opt-in module is not intended for that scale.",
            n_classes, int(y_arr.max()), n_classes,
        )
        return None

    counts = np.bincount(y_arr, minlength=n_classes).astype(np.float64)
    populated = counts > 0
    n_populated = int(populated.sum())
    if n_populated < 2:
        return None

    rare_count = float(counts[populated].min())
    minority_prior = rare_count / float(n)

    # Two-sided gate (applied for the opt-in on/auto modes): correct only when
    # imbalanced enough AND with enough rare rows to estimate reliably.
    if minority_prior >= _PRIOR_THRESHOLD:
        return None  # balanced / mild imbalance -> plain MI (no-regression)
    if rare_count < _N_RARE_FLOOR:
        return None  # too few rare rows -> plain MI (gate fallback)

    # Inverse-prior weights: each populated class -> equal total mass 1/K_pop.
    # Empty classes get weight 0 (they contribute nothing anyway).
    w = np.zeros(n_classes, dtype=np.float64)
    inv_k = 1.0 / float(n_populated)
    for c in range(n_classes):
        if counts[c] > 0.0:
            w[c] = inv_k / counts[c]
    return w


@njit(cache=True, fastmath=True, parallel=True)
def _class_balanced_mi_batch_njit(
    X_cols: np.ndarray,
    y: np.ndarray,
    class_w: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """Inverse-prior class-balanced plug-in MI of each column of X_cols vs discrete y.

    Identical histogram/binning recipe to ``_plugin_mi_classif_batch_njit`` (equi-
    frequency quantile bins via argsort), but each sample of class ``c``
    contributes weight ``class_w[c]`` to the joint/marginal histograms instead
    of 1. The reweighted distribution gives every class equal total mass, so a
    feature that separates the rare class scores on its true discriminative
    power rather than its (tiny) marginal contribution. Returns MI in nats on
    the reweighted distribution.

    parallel=True over columns mirrors the plain batch kernel.
    """
    n = X_cols.shape[0]
    k = X_cols.shape[1]
    n_classes = class_w.shape[0]
    out = np.zeros(k, dtype=np.float64)

    # Total reweighted mass is column-independent: W = sum_i class_w[y_i].
    W = 0.0
    for i in range(n):
        W += class_w[y[i]]
    if W <= 0.0:
        return out
    inv_W = 1.0 / W

    for j in prange(k):
        col = X_cols[:, j]
        sort_idx = np.argsort(col)
        x_binned = np.empty(n, dtype=np.int64)
        base = n // n_bins
        rem = n % n_bins
        pos = 0
        for b in range(n_bins):
            size = base + (1 if b < rem else 0)
            for _ in range(size):
                x_binned[sort_idx[pos]] = b
                pos += 1

        hist_xy = np.zeros((n_bins, n_classes), dtype=np.float64)
        hist_x = np.zeros(n_bins, dtype=np.float64)
        hist_y = np.zeros(n_classes, dtype=np.float64)
        for i in range(n):
            b = x_binned[i]
            c = y[i]
            wc = class_w[c]
            hist_xy[b, c] += wc
            hist_x[b] += wc
            hist_y[c] += wc

        mi = 0.0
        for b in range(n_bins):
            wx = hist_x[b]
            if wx <= 0.0:
                continue
            log_px = math.log(wx * inv_W)
            for c in range(n_classes):
                wxy = hist_xy[b, c]
                wy = hist_y[c]
                if wxy <= 0.0 or wy <= 0.0:
                    continue
                p_xy = wxy * inv_W
                mi += p_xy * (math.log(p_xy) - log_px - math.log(wy * inv_W))
        if mi < 0.0:
            mi = 0.0
        out[j] = mi
    return out
