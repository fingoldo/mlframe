"""Coverage test: count_encoding and frequency_encoding both survive the cat-FE floor-drop rescue when the raw
categorical column is ALSO selected (so the encoding must clear a genuine held-out-R^2 usability bar, not just
exist).

This was written while root-causing a pre-existing ``test_fe_families_polars_parity.py`` flake where
``count_encoding`` failed to engineer any surviving column on one seed. The actual, dominant root cause turned out
to be that fixture's ``eff`` mapping being near-linear in the raw category id (fixed there: an exact-zero-
population-correlation ``eff``) -- with an info-equivalent ``cat__freq`` twin surviving on a different seed purely
by luck of that seed's draw, not because of any difference in how the two encodings were evaluated.

Along the way this surfaced a real, independently-reproducible numerical fact worth defending against regardless:
the rescue's re-add probe used to read a candidate's value from ``data[:, idx]`` (the nbins-QUANTIZED screening
code) while its baseline design used full-precision continuous values for the same already-selected columns.
Quantile bin-edge digitization is not exactly scale-invariant for a duplicate-heavy column (see
``test_quantile_tie_scale_invariance.py`` for an isolated pin of that fact) -- pure floating-point luck could make
that quantization loss matter for SOME fixture even though it isn't the culprit here. Fixed in
``_fit_impl_core.py``'s cat-FE floor-drop protection block to prefer the raw continuous value (matching what the
baseline already does), with the quantized code only as a fallback. This file's two tests below did not
empirically regress when that specific line was reverted on the current fixture (the R^2 margin here is wide
enough that quantization noise doesn't flip the outcome) -- they are kept as behavioral coverage for the rescue
mechanism itself, not as a pin of that one line.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr import MRMR


def _categorical(seed, n=1500):
    """Same fixture (and the same validated seeds, 21/22) as ``test_fe_families_polars_parity.py``'s
    ``_categorical`` -- see that docstring for why ``eff`` must have exactly zero population correlation with the
    raw category id. MRMR's screen dynamics are sensitive enough to fixture/seed interaction that this combination
    is validated for these specific seeds, not claimed robust for an arbitrary one."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, 8, n)
    eff = np.array([-3, 0, 1, 2, 3, 0, -1, -2], dtype=float)[cat]
    num = rng.standard_normal(n)
    noise = rng.standard_normal((n, 2))
    y = (eff + 0.5 * num + rng.standard_normal(n) > 0).astype(int)
    data = {"cat": cat.astype("int64"), "num": num, "n0": noise[:, 0], "n1": noise[:, 1]}
    return pd.DataFrame(data), pd.Series(y)


_SCORER_KW = dict(
    # univariate_basis OFF (unlike the polars-parity suite's shared fixture config): it produces a competing
    # composite feature (e.g. ``add(sqrt(cat),rint(num))``) that partially subsumes the same residual signal the
    # count/frequency encoding needs to clear the floor-drop rescue's R^2 threshold, whose margin here is thin
    # enough that the split between "subsumed by the composite" and "recoverable only by the encoding" varies by
    # execution context (measured: this exact fixture at seed=21 is unreliable with univariate_basis on; seeds 20
    # and 22 are deterministic across 3 repeated runs with it off). This file's only job is deterministic coverage
    # of the rescue mechanism in isolation, not reproducing the full polars-parity kitchen-sink FE config.
    fe_univariate_basis_enable=False,
    fe_hybrid_orth_pair_enable=False,
    fe_hybrid_orth_triplet_enable=False,
    fe_hybrid_orth_quadruplet_enable=False,
    fe_hinge_enable=False,
    fe_kfold_te_enable=False,
    fe_binned_numeric_agg_enable=False,
)


def test_count_encoding_survives_when_raw_cat_is_selected():
    """count_encoding survives when raw cat is selected."""
    X, y = _categorical(20)
    sel = MRMR(verbose=0, random_seed=0, fe_count_encoding_enable=True, fe_count_encoding_cols=("cat",), **_SCORER_KW)
    sel.fit(X, y)
    assert sel.count_encoding_features_ == ["cat__count"], (
        f"count_encoding should survive the cat-FE floor-drop rescue on this fixture (raw 'cat' is an imperfect "
        f"linear predictor of the shuffled per-category effect, leaving real residual R^2 for the encoding to "
        f"recover); got {sel.count_encoding_features_!r}"
    )


def test_frequency_encoding_survives_alongside_count_encoding():
    """frequency_encoding (an exact monotone rescale of count_encoding: freq = count / n) must survive under the
    SAME fixture/mechanism -- info-equivalent encodings should get the same rescue outcome."""
    X, y = _categorical(20)
    sel = MRMR(verbose=0, random_seed=0, fe_frequency_encoding_enable=True, fe_frequency_encoding_cols=("cat",), **_SCORER_KW)
    sel.fit(X, y)
    msg = f"frequency_encoding should survive alongside count_encoding on an info-equivalent fixture; got {sel.frequency_encoding_features_!r}"
    assert sel.frequency_encoding_features_ == ["cat__freq"], msg
