"""Reject-tracker for the composite-discovery ``mi_nbins`` default.

A multi-scenario bench (``bench_mi_bin_nbins_ranking.py``) measured rank-recovery of the known-truth MI ordering across
nbins in {8,12,16,24,32}. Verdict: KEEP nbins=16. A coarser grid (nbins=8) helps only at small n, but production callers
pass n>=20k where it ties OR LOSES -- on fine multi-lobe dependence the coarse grid blurs structure and mis-ranks. These
tests pin that verdict so a future "just lower the default for speed" cannot silently degrade ranking on prod-shape data.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MLFRAME_NO_CUDA_AUTOCONFIG", "1")
os.environ.setdefault("MLFRAME_KEEP_BROKEN_CUPY", "1")

from mlframe.training._composite_target_discovery_config import CompositeTargetDiscoveryConfig
from mlframe.training.composite.discovery.screening import _mi_pair_bin


def _concordance(true: np.ndarray, est: np.ndarray) -> float:
    k = len(true)
    correct = total = 0
    for i in range(k):
        for j in range(i + 1, k):
            if true[i] == true[j]:
                continue
            total += 1
            if (est[i] - est[j]) * (true[i] - true[j]) > 0:
                correct += 1
    return correct / total if total else 1.0


def _multimodal_fine_pairs(rng: np.random.Generator, n: int):
    """Multi-lobe (k-band) dependence: MI rises ~log(k) with band count; finer structure a coarse grid blurs."""
    kbands = [1, 2, 3, 4, 6, 8]
    pairs, true = [], []
    for kb in kbands:
        x = rng.uniform(0, 1, size=n)
        y = np.floor(x * kb) + 0.05 * rng.normal(size=n)
        pairs.append((x, y))
        true.append(np.log(kb))
    return pairs, np.array(true)


def test_default_mi_nbins_is_16():
    """The shipped default must stay 16 (the bench-validated ranking optimum at prod n)."""
    assert CompositeTargetDiscoveryConfig().mi_nbins == 16


def test_nbins16_beats_coarse_on_fine_structure_at_prod_n():
    """On prod-shape n=20k fine multi-lobe data, nbins=16 must rank-recover better than the coarse nbins=8 challenger.

    Measured (bench): concordance 16 -> 1.000 vs 8 -> 0.933. This is the cell that REJECTS lowering the default to 8.
    Fails on a config that lowered the default toward a coarse grid.
    """
    concs16, concs8 = [], []
    for seed in (0, 1, 2):
        rng = np.random.default_rng(100 + seed)
        pairs, true = _multimodal_fine_pairs(rng, 20_000)
        est16 = np.array([_mi_pair_bin(x, y, nbins=16) for (x, y) in pairs])
        est8 = np.array([_mi_pair_bin(x, y, nbins=8) for (x, y) in pairs])
        concs16.append(_concordance(true, est16))
        concs8.append(_concordance(true, est8))
    mean16, mean8 = float(np.mean(concs16)), float(np.mean(concs8))
    assert mean16 >= 0.98, f"nbins=16 fine-structure concordance regressed: {mean16:.3f}"
    assert mean16 > mean8 + 0.02, f"nbins=16 must beat coarse nbins=8 on fine structure: {mean16:.3f} vs {mean8:.3f}"


@pytest.mark.parametrize("nbins", [8, 12, 16, 24, 32])
def test_nbins16_never_loses_on_gaussian_ladder_at_prod_n(nbins):
    """On a monotone Gaussian-MI ladder at prod n, every grid (incl. 16) recovers the order perfectly -- so coarser bins
    buy NOTHING where they would (the easy case), and the only differentiator is fine structure (handled above)."""
    rng = np.random.default_rng(7)
    rhos = [0.1, 0.25, 0.4, 0.55, 0.7, 0.85]
    pairs, true = [], []
    for rho in rhos:
        x = rng.normal(size=20_000)
        y = rho * x + np.sqrt(1 - rho * rho) * rng.normal(size=20_000)
        pairs.append((x, y))
        true.append(-0.5 * np.log(1 - rho * rho))
    est = np.array([_mi_pair_bin(x, y, nbins=nbins) for (x, y) in pairs])
    assert _concordance(np.array(true), est) == 1.0
