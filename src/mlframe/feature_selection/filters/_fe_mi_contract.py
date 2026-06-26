"""Shared MI source-of-truth contract for the FE candidate-scoring paths (2026-06-26).

The MRMR feature-engineering candidate scoring runs on two SEPARATE, independently-optimised backends
-- a CPU njit path and a GPU cupy path -- that MUST select the IDENTICAL feature set. This leaf module
is the single place that pins the few decisions on which that identity depends, so neither backend can
drift from the other. It imports ONLY numpy + stdlib (no FE-pipeline imports) so any module can depend
on it without a cycle.

Two empirically-grounded facts motivate the contract (probed 2026-06-26 on F2):
  1. Equi-frequency RANK binning (``hermite_fe._quantile_bin_njit``: argsort -> n/nbins rows per bin) and
     equi-frequency PERCENTILE-EDGE binning (``_usability_njit_pool._qbin_into`` / GPU
     ``_gpu_resident_select._radix_select_interior_edges``: np.quantile lerp edges + searchsorted) produce
     BIT-IDENTICAL bin codes -- and therefore bit-identical plug-in MI -- on continuous data with NO ties
     (measured max|drift| 0.0 across 400 columns). They diverge ONLY at TIED / duplicate values on a bin
     boundary: rank splits equal values arbitrarily across two bins, edge keeps them together. So the only
     columns at which the two backends' MI can disagree are low-cardinality / discrete / indicator columns.
  2. End to end, F2 already selects byte-identical features under CPU-rank and GPU-edge -- because its
     engineered forms are smooth. The contract exists to keep that true for the tied-column case too, and
     under the new GPU-resident relevance path (full residency, where relevance MI moves onto the GPU).

THE CONTRACT (applies to the FE BATCHER paths -- the new CPU and GPU resident scorers)
------------
* BINNING: both BATCHER backends use equi-frequency PERCENTILE-EDGE binning so they bin tied values
  identically and select the same forms. The orth/basis family uses the GPU's convention -- FIXED
  ``n_bins-1`` interior edges (``cp.percentile``/``_radix_select_interior_edges``, NO dedup) +
  ``searchsorted(side='right')`` -- whose bit-faithful CPU twin is
  ``_fe_edge_mi.plugin_mi_classif_batch_edge_njit`` (parity verified to ~1e-9 on continuous AND tied,
  ``test_fe_edge_mi_parity``). The usability pool uses ``_qbin_into`` (np.quantile lerp + np.unique dedup)
  on both its CPU and GPU sides already. Each family's CPU and GPU binning are identical; the conventions
  differ BETWEEN families, which is fine -- the contract is per-family CPU==GPU, not one global binning.

  SCOPE NOTE: this edge convention governs the BATCHER paths. The LEGACY per-family orth CPU scorer
  (``_orth_mi_backends._mi_classif_batch_numba``) intentionally keeps RANK binning as its default, because
  switching that default to edge regresses a razor-edge redundancy pin (``test_private_raw_a_kept...``):
  the ~1e-12 rank-vs-edge MI perturbation, even on continuous data, pushes a spurious cross-signal form
  over the admission gate. Making edge the legacy default (so legacy CPU == GPU too) needs that
  redundancy/admission gate hardened with a tolerance band first; tracked separately.
* MI ESTIMATOR + REDUCTION ORDER: whatever estimator a family uses (plain plug-in for the orthogonal/basis
  families; Miller-Madow for the usability pool), BOTH backends compute the SAME one with the SAME
  occupied-cell reduction order (ascending bin, ascending class). The plain-vs-MM choice is per-family and
  is NEVER changed by routing a family onto the GPU.
* TIEBREAKER QUANTISATION: the selection key ``_pairs_gates._select_single_best`` uses EXACT MI as its
  third tiebreaker leg, decisive among forms whose banded MI AND linear usability tie. A cross-backend fp
  reduction-order difference (~1e-12) -- or any residual sub-noise binning difference -- could otherwise
  flip that leg and pick a different form on GPU than on CPU. ``quantize_mi_tiebreak`` snaps the leg to a
  grid far coarser than that drift yet far finer than any genuine within-band MI gap, so the leg is
  hardware-independent; genuinely-distinct MIs still order correctly, sub-grid ties fall through to the
  deterministic name key.
"""
from __future__ import annotations

import math

# Quantisation grid for the EXACT-MI tiebreaker leg, in nats. Chosen so that
#   cross-backend drift (unified binning -> fp reduction order, ~1e-12)  <<  QUANTUM  <<  genuine MI gaps.
# The plug-in MI tie band is ``(k_x-1)(k_y-1)/2n`` ~ 1e-3 on the FE quantiser, and the tightest genuine
# within-band gap the leg must still resolve (F2 mixed: 0.1180 vs 0.1167) is ~1.3e-3 -- both >> 1e-7 --
# while fp reduction-order drift between two summation orders of the same histogram is ~1e-12 << 1e-7.
# So 1e-7 collapses ONLY sub-noise ties (pure fp jitter) to the deterministic name key and never merges
# two forms that differ by a meaningful amount.
_MI_TIEBREAK_QUANTUM: float = 1e-7


def quantize_mi_tiebreak(mi: float, quantum: float = _MI_TIEBREAK_QUANTUM) -> float:
    """Snap an MI value to the tiebreaker grid so the exact-MI selection leg is hardware-independent.

    Used ONLY for the tiebreaker comparison key, never for the reported/stored MI. Two MIs that differ by
    less than ``quantum`` (cross-backend fp jitter or sub-noise binning difference) map to the SAME key and
    are resolved by the next, deterministic leg; two MIs that differ by a genuine amount (>> quantum) keep
    their order. ``round`` (banker's rounding) is deterministic and platform-stable for these magnitudes.
    """
    if quantum <= 0.0:
        return float(mi)
    m = float(mi)
    if not (m == m):  # NaN guard: leave NaN as-is (max() over a NaN key is already undefined upstream)
        return m
    return round(m / quantum) * quantum
