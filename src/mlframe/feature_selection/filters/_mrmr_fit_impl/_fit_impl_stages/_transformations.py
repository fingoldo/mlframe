"""Construction of the unary/binary transformation registries the FE pair search draws from."""

from __future__ import annotations

# --- imports (managed) ---
import numpy as np
from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_core import logger
# --- end imports ---


def _build_fe_transformations(self, fe, verbose):
    """Build the unary/binary transformation registries for the FE pair search (row-wise pure ops only, plus seeded random polynomials) and fresh engineered/checked-pair sets."""
    from mlframe.feature_selection.filters.mrmr import (
        create_binary_transformations,
        create_unary_transformations,
    )

    unary_transformations = create_unary_transformations(preset=fe.unary_preset)
    binary_transformations = create_binary_transformations(preset=fe.binary_preset)
    # REPLAY-SAFETY: exclude ops that are NOT row-wise pure functions from FE
    # pair candidates. Their value at a row depends on OTHER rows (``np.gradient``: grad1/grad2) or
    # on a whole-column statistic recomputed at apply time (``logn`` uses ``x - np.min(x)``), so a
    # recipe built on them silently produces DIFFERENT values on a row-slice / test frame
    # (slice-replay corruption - the same class as the smart_log BUG2 fix). They appear only in the
    # non-default "maximal" preset; dropping them here means they are never selected as engineered
    # features, while the create_*_transformations registry stays intact (other callers + the
    # registry-coverage test are unaffected). On the default "minimal" preset this is a no-op.
    _FE_NON_ROWWISE_PURE = ("grad1", "grad2", "logn")
    unary_transformations = {k: v for k, v in unary_transformations.items() if k not in _FE_NON_ROWWISE_PURE}
    binary_transformations = {k: v for k, v in binary_transformations.items() if k not in _FE_NON_ROWWISE_PURE}
    if fe.max_polynoms:
        # Generated polynomial coefficients are appended directly to unary_transformations under "poly_<coef>" keys;
        # no separate registry is needed. Use a seeded local Generator so the polynomial recipes are reproducible
        # across reruns with the same ``random_seed`` - prior code used the global ``np.random`` stream, breaking
        # determinism whenever any earlier suite stage advanced it.
        _poly_rng = np.random.default_rng(self.random_seed)
        for _ in range(fe.max_polynoms):
            length = int(_poly_rng.integers(3, 9))
            coef = np.empty(shape=length, dtype=np.float32)
            for i in range(length):
                coef[i] = _poly_rng.normal((1.0 if i == 1 else 0.0), scale=0.05)

            unary_transformations["poly_" + str(coef)] = coef

    if verbose > 2:
        logger.info("nunary_transformations: %s", f"{len(unary_transformations):_}")
        logger.info("nbinary_transformations: %s", f"{len(binary_transformations):_}")

    engineered_features: set = set()
    checked_pairs: set = set()
    return binary_transformations, checked_pairs, engineered_features, unary_transformations
