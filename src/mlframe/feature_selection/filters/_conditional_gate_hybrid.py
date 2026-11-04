"""Detection and hybrid-FE entry points for the row-argmax and conditional-gate operators.

Carved out of ``_conditional_gate_fe`` for file size; the parent re-exports every name defined here.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np


from mlframe.feature_selection.filters._y_encoding import encode_y_for_classif_mi

from ._pairwise_modular_fe import _mi

logger = logging.getLogger(__name__)

# The gate-grid build kernels live in a sibling (file-size carve only); re-exported here because the parent
# is their only caller and `_conditional_gate_fe._gate_select_grid_njit` is the name benchmarks reference.
from mlframe.utils.env_flags import env_int

from ._conditional_gate_naming import (
    build_conditional_gate_recipe,
    build_row_argmax_recipe,
    engineered_name_conditional_gate,
    engineered_name_row_argmax,
)

# Quantile grid for the tau scan: skip the extreme tails (a tau at q<=0.05 / q>=0.95 leaves one branch nearly empty, so the
# gate degenerates to a single column already on the raw list). 17 interior quantiles is enough to land near a true tau.
# Margin the engineered column's MI must beat its operand baseline by (mirrors _pairwise_modular_fe._MIN_MARGIN). Below it the
# selector can already recover the signal from a raw / cheap-op column, so the engineered column adds no genuine structure.
_MIN_MARGIN = 0.02

# Row threshold above which _build_feats fuses the per-tau (n, 17) mask/select block into one njit(prange) kernel
# instead of the numpy per-tau loop. The isolated build kernel wins at all n, but an earlier reject found
# it LOSES end-to-end at small n (build is a tiny fraction of the scan + its prange contends with the MI prange);
# gated ON only for large n where the build is a real fraction of the scan (validated end-to-end). Env-overridable.
_GATE_BUILD_NJIT_MIN_N = env_int("MLFRAME_GATE_BUILD_NJIT_MIN_N", 20000, minimum=0)

# Absolute floor the engineered MI must clear ABOVE the permutation-null band (not just `> null_hi`); mirrors _pairwise_modular_fe._MIN_NULL_MARGIN.
# Guards the cardinality-inflation false positive on a few-class y (a ~10-bin regression/quantized target), where a select/mask column's
# plug-in MI can sit ~0.01 nats above a z=3 null on noise; a true regime/argmax hit clears the null by a wide margin.
_MIN_NULL_MARGIN = 0.05

from ._conditional_gate_fe_helpers import (  # noqa: F401  -- carved helpers
    _TAU_QUANTILES,
    _scan_gate_column,
)
from typing import TYPE_CHECKING
from typing import Any
if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from .engineered_recipes import EngineeredRecipe


from . import _conditional_gate_fe as _cgf

logger = logging.getLogger(__name__)


def detect_row_argmax(
    X: pd.DataFrame | pl.DataFrame, y: np.ndarray, cols: Optional[Sequence[str]] = None, *,
    top_k: int = 4, nbins: int = 12, seed: int = 0,
) -> list[dict[str, Any]]:
    """Cheap-first scan; returns the list of responded row-argmax hits (each a dict with the source triple, MI + margin), capped
    at ``top_k``. Empty when nothing responds - a non-argmax frame detects nothing."""
    hits = _cgf.cheap_row_argmax_scan(X, y, cols, nbins=nbins, seed=seed)
    out = []
    for h in hits:
        if not h.responded:
            continue
        out.append({"cols": h.cols, "feat_mi": h.feat_mi, "operand_floor": h.operand_floor, "margin": h.margin_over_operands})
        if len(out) >= int(top_k):
            break
    return out

def detect_conditional_gate(
    X: pd.DataFrame | pl.DataFrame, y: np.ndarray, cols: Optional[Sequence[str]] = None, *,
    top_k: int = 4, nbins: int = 12, seed: int = 0,
) -> list[dict[str, Any]]:
    """Cheap-first scan; returns the list of responded conditional-gate hits (each a dict with mode, source columns, frozen tau,
    MI + margin), capped at ``top_k``. Empty when nothing responds - the hardened baseline keeps smooth / ordinary_mul silent."""
    hits = _cgf.cheap_conditional_gate_scan(X, y, cols, nbins=nbins, seed=seed)
    out = []
    for h in hits:
        if not h.responded:
            continue
        out.append({"mode": h.mode, "cols": h.cols, "tau": h.tau, "feat_mi": h.feat_mi, "baseline_mi": h.baseline_mi, "margin": h.margin_over_baseline})
        if len(out) >= int(top_k):
            break
    return out

# Recipe plumbing: emit frozen EngineeredRecipe objects so the MRMR selector materialises, scores, selects, and replays the
# argmax / gate column at predict time identically. Replay is a pure function of X (argmax index; gate np.where / mask with the
# FROZEN tau) - no y reference, so it is leak-free + deterministic + train/test bit-identical.

def _top_marginal_mi_columns(X, y: np.ndarray, cols: Sequence[str], k: int, *, nbins: int) -> list[str]:
    """The ``k`` columns of ``cols`` with the largest marginal binned MI against ``y``, in their original order."""
    yi = encode_y_for_classif_mi(np.asarray(y))
    scores = np.array([_mi(np.asarray(X[c]), yi, nbins=nbins) for c in cols], dtype=np.float64)  # Intentional float64: ranking statistic
    keep = set(np.argsort(-scores, kind="stable")[:k].tolist())
    return [c for i, c in enumerate(cols) if i in keep]


def hybrid_row_argmax_fe_with_recipes(
    X: pd.DataFrame | pl.DataFrame, y: np.ndarray, *,
    cols: Optional[Sequence[str]] = None,
    top_k: int = 4, nbins: int = 12, seed: int = 0,
    max_cols: int = 30,
) -> tuple[list[str], list[EngineeredRecipe]]:
    """Detect responded row-argmax structure and emit it as frozen, replayable ``EngineeredRecipe`` objects.

    Column-count BUDGET GUARD (the wide-frame cost is the budgeted C(p,3) triple sweep + the batched MI; see
    ``_benchmarks/bench_conditional_gate_wideframe``): when the number of eligible columns exceeds ``max_cols`` the whole sweep is
    SKIPPED (logged, never silent).

    Returns ``(appended_names, recipes)`` - the materialised columns are NOT concatenated here (the MRMR caller appends them under
    its own RAM-safe path); each recipe replays the exact argmax column from X alone, leak-free."""
    if cols is None:
        elig = [c for c in X.columns if _cgf._is_argmax_eligible(np.asarray(X[c]))]
    else:
        elig = [c for c in cols if c in X.columns and _cgf._is_argmax_eligible(np.asarray(X[c]))]

    p = len(elig)
    if p > int(max_cols):
        # Every eligible column is ranked by marginal MI and the triple budget is spent on the strongest triples (cheap_row_argmax_scan), so a wide
        # frame is swept in full instead of being skipped or cut to a subset of columns.
        logger.info(
            "row_argmax FE: %d eligible columns exceeds max_cols=%d; sweeping all of them with the triple budget spent on the strongest.",
            p, int(max_cols),
        )

    hits = _cgf.cheap_row_argmax_scan(X, y, elig, nbins=nbins, seed=seed, _cols_prefiltered=True)
    appended: list[str] = []
    recipes = []
    seen: set[str] = set()
    for h in hits:
        if not h.responded:
            continue
        name = engineered_name_row_argmax(h.cols)
        if name in seen:
            continue
        seen.add(name)
        appended.append(name)
        recipes.append(build_row_argmax_recipe(name=name, cols=h.cols))
        if len(appended) >= int(top_k):
            break
    return appended, recipes

def hybrid_conditional_gate_fe_with_recipes(
    X: pd.DataFrame | pl.DataFrame, y: np.ndarray, *,
    cols: Optional[Sequence[str]] = None,
    top_k: int = 4, nbins: int = 12, seed: int = 0,
    k_gate: int = 8, k_operand: int = 10,
    max_cols: int = 200,
    subsample_n: int = 0,
) -> tuple[list[str], list[EngineeredRecipe]]:
    """Detect responded conditional-gate structure and emit it as frozen, replayable ``EngineeredRecipe`` objects (tau frozen).

    RELEVANCE-PRUNED candidate set: the sweep cost is O(k_operand^2 * k_gate * tau-scan), FLAT in p - the gate column c is drawn
    from the top-``k_gate`` columns by raw MI vs y and the operands a, b from the top-``k_operand`` (see ``cheap_conditional_gate_scan``).
    With k=8/10 the added cost is a small constant whether p=30 or p=300, so the prior O(p^3) blow-up that forced this OFF is gone.
    ``max_cols`` (default 200) no longer skips the sweep: it is logged when exceeded, and the relevance pruning bounds the cost. The detector gates vs the HARDENED best-existing-op baseline so smooth / ordinary_mul
    controls stay silent (0 FP at p=30 over 3 seeds).

    Returns ``(appended_names, recipes)`` - the materialised columns are NOT concatenated here (the MRMR caller appends them under
    its own RAM-safe path); each recipe replays the exact gate column from X + the frozen tau alone, leak-free."""
    if cols is None:
        elig = [c for c in X.columns if _cgf._is_argmax_eligible(np.asarray(X[c]))]
    else:
        elig = [c for c in cols if c in X.columns and _cgf._is_argmax_eligible(np.asarray(X[c]))]

    p = len(elig)
    if p > int(max_cols):
        # The sweep is relevance-pruned (top-k gate and operand columns by raw MI), so its cost does not grow with p: a wide frame is swept
        # in full rather than skipped. The planted gate is lost at p=260 with the skip and found without it (D:/Temp/gate_cap.py).
        logger.info(
            "conditional_gate FE: %d eligible columns exceeds max_cols=%d; sweeping all of them over the relevance-pruned candidate set.",
            p, int(max_cols),
        )

    hits = _cgf.cheap_conditional_gate_scan(
        X, y, elig, nbins=nbins, seed=seed, k_gate=k_gate, k_operand=k_operand, subsample_n=subsample_n,
        _cols_prefiltered=True,
    )
    appended: list[str] = []
    recipes = []
    seen: set[str] = set()
    for h in hits:
        if not h.responded:
            continue
        name = engineered_name_conditional_gate(h.mode, h.cols, h.tau)
        if name in seen:
            continue
        seen.add(name)
        appended.append(name)
        recipes.append(build_conditional_gate_recipe(name=name, mode=h.mode, cols=h.cols, tau=h.tau))
        if len(appended) >= int(top_k):
            break
    return appended, recipes
