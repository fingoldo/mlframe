"""FE_ROOT_B-4 (2026-08-05 audit): knn_gradient_features wraps a batched np.linalg.solve for ALL query
rows in one try/except LinAlgError -- the audit's concern was that one singular neighbourhood (e.g.
duplicate/quantized coordinates) could set gradient features to NaN for the ENTIRE batch, not just the
offending row. Empirically, the function's existing fixed 1e-12 ridge already keeps LAPACK's solve from
raising in most degenerate cases -- but that same ridge means a rank-deficient neighborhood "succeeds"
with a spurious near-zero gradient in the collapsed direction instead of erroring, silently masking that
no reliable local gradient exists there (a real, verified bug in its own right: pre-fix, a query whose
neighborhood is a duplicate-coordinate cluster returned a finite grad_norm=0.0, not NaN).

Fixed by detecting (near-)singular per-query design matrices up front via their batched singular values
(condition number), substituting a well-conditioned placeholder just for those rows so the batched solve
never raises, then overwriting exactly those rows with NaN afterward -- every other query keeps its
normal, fully-vectorized batched solve, and a degenerate neighborhood now reports the honest NaN.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.spatial import knn_gradient_features


def test_singular_neighborhood_does_not_nan_other_queries():
    """One query's neighborhood is exactly degenerate (duplicate reference points collapse the design
    matrix's direction); a second, well-separated query has a normal, well-conditioned neighborhood with
    an exact linear gradient. The degenerate query must be NaN (not a spurious finite 0.0, which is what
    the pre-fix ridge-regularized solve silently returned); the well-conditioned query must recover the
    true gradient either way, confirming the new per-row singularity check does not disturb it."""
    k = 5
    rng = np.random.default_rng(0)

    # Duplicate cluster: k identical reference points -> for a query AT that same point, x_diff is
    # identically zero for every neighbor, collapsing the design matrix's gradient directions (confirmed
    # via direct SVD: singular values [5, 1e-12, 1e-12], condition number ~5e12, well above any
    # legitimate well-conditioned neighborhood's ratio).
    dup_ref = np.zeros((k, 2))
    dup_labels = rng.normal(size=k)

    # Well-conditioned cluster, far away, with an EXACT linear label (no noise) so WLS recovers the
    # gradient to near machine precision regardless of which k points/weights are picked.
    good_ref = rng.uniform(10.0, 20.0, size=(60, 2))
    good_labels = 2.0 * good_ref[:, 0] + 3.0 * good_ref[:, 1]

    ref_coords = np.vstack([dup_ref, good_ref])
    ref_labels = np.concatenate([dup_labels, good_labels])

    q_coords = np.array(
        [
            [0.0, 0.0],  # query 0: neighborhood is the duplicate cluster -> exactly singular
            [15.0, 15.0],  # query 1: neighborhood is the well-conditioned cluster
        ]
    )

    out = knn_gradient_features(q_coords, ref_coords, ref_labels, k=k)

    assert np.isnan(out["grad_norm"][0]), "degenerate-neighborhood query must be NaN (documented behaviour)"

    assert np.isfinite(out["grad_norm"][1]), "the OTHER, well-conditioned query must not be collaterally NaN'd"
    assert out["grad_axis_0"][1] == pytest.approx(2.0, abs=1e-6)
    assert out["grad_axis_1"][1] == pytest.approx(3.0, abs=1e-6)
    assert out["wls_residual_std"][1] == pytest.approx(0.0, abs=1e-6)


def test_all_well_conditioned_queries_unaffected_by_the_new_guard():
    """Sanity: with no singular neighborhoods at all, results are unaffected by the new detection path."""
    k = 8
    rng = np.random.default_rng(1)
    ref_coords = rng.uniform(0.0, 50.0, size=(200, 2))
    ref_labels = 1.5 * ref_coords[:, 0] - 0.5 * ref_coords[:, 1]
    q_coords = rng.uniform(5.0, 45.0, size=(20, 2))

    out = knn_gradient_features(q_coords, ref_coords, ref_labels, k=k)

    assert np.isfinite(out["grad_norm"]).all()
    np.testing.assert_allclose(out["grad_axis_0"], 1.5, atol=1e-6)
    np.testing.assert_allclose(out["grad_axis_1"], -0.5, atol=1e-6)
