"""Helpers carved out of ``_conditional_gate_fe`` to keep that module under its size budget."""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# The gate-grid build kernels live in a sibling (file-size carve only); re-exported here because the parent
# is their only caller and `_conditional_gate_fe._gate_select_grid_njit` is the name benchmarks reference.
from mlframe.utils.env_flags import env_int

ROW_ARGMAX_PREFIX = "argmax"
CONDITIONAL_GATE_PREFIX = "gate"

GATE_MODES = ("select", "mask")

# Quantile grid for the tau scan: skip the extreme tails (a tau at q<=0.05 / q>=0.95 leaves one branch nearly empty, so the
# gate degenerates to a single column already on the raw list). 17 interior quantiles is enough to land near a true tau.
_TAU_QUANTILES = tuple(np.round(np.linspace(0.1, 0.9, 17), 4))

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


_TAU_QUANTILES = tuple(np.round(np.linspace(0.1, 0.9, 17), 4))


def _scan_gate_column(gate_cols, arrs, operand_cols, _add):
    """Score the conditional-gate candidates for one gate column."""
    from mlframe.feature_selection.filters._fe_deadline import fe_deadline_passed

    for cgate in gate_cols:
        # Optional-enrichment wall-clock budget: stop the O(k_gate * k_operand^2) gate sweep once
        # MRMR.fit's deadline passes; flush whatever candidates are already queued and return the hits
        # found so far. No-op without a budget (mirrors the orth-univariate/pair-cross/extra-basis
        # generators' internal deadline check).
        if fe_deadline_passed():
            break
        cv = arrs[cgate]
        taus = np.quantile(cv, _TAU_QUANTILES)
        others = [cn for cn in operand_cols if cn != cgate]
        # mask: one active column a (cols = (a, c)); baseline over {a, c}.
        for a in others:
            av = arrs[a]
            _add("mask", (a, cgate), (cv, av), taus, (a, cgate))
        # select: ordered (a, b), cols = (a, b, c); baseline over {a, b, c}.
        for a in others:
            for b in others:
                if a == b:
                    continue
                av, bv = arrs[a], arrs[b]
                _add("select", (a, b, cgate), (cv, av, bv), taus, (a, b, cgate))
