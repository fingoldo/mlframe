"""Fast decision-equivalence guard for the permutation-fallback prefilter (2026-07-19).

Regression test for ``_permutation_prefilter_reject`` (``_mdlp_validated_split.py``): the
prefilter must never change which splits get accepted relative to the unfiltered permutation
path -- it is a reject-only speed shortcut, not a looser or stricter significance test. See
``_benchmarks/bench_mdlp_prefilter_hybrid.py`` for the full A/B methodology and wall-time
measurement (warm-JIT, median-of-3, interleaved order).
"""
from __future__ import annotations

from mlframe.feature_selection.filters._benchmarks.bench_mdlp_prefilter_hybrid import run_fast_subset


def test_prefilter_never_changes_accepted_edges():
    """Every scenario x seed combination in the fast subset must produce byte-for-byte identical
    accepted bin edges with and without the prefilter -- any mismatch means the reject-only
    shortcut silently turned into a decision-altering one, a real correctness regression."""
    results = run_fast_subset()
    assert results
    mismatches = [r for r in results if not r.edges_match]
    assert not mismatches, [(r.scenario, r.seed, r.n_bins_with, r.n_bins_without) for r in mismatches]
