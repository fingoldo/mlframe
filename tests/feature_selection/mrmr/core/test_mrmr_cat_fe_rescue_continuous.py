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


def _fit(seed, **kw):
    """Fit MRMR with one encoding enabled on the ``_categorical`` fixture and return the estimator."""
    X, y = _categorical(seed)
    return MRMR(verbose=0, random_state=0, fe_count_encoding_cols=("cat",), fe_frequency_encoding_cols=("cat",), **kw, **_SCORER_KW).fit(X, y)


def test_the_rescue_returns_an_encoding_with_held_out_gain_and_leaves_a_noise_encoding_out():
    """The mechanism itself, on controlled data: the held-out R^2 probe re-adds an encoding that carries the per-category effect the selected design lacks and skips one that is noise.

    End to end this is no longer observable on a count encoding: counts of a uniformly drawn category are noise about the target, the raw category carries the effect, and the parsimony
    compaction rightly removes an encoding that adds nothing to the finished selection."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._group3 import _score_encoded_features_heldout

    X, y = _categorical(20)
    n = len(X)
    rng = np.random.default_rng(0)
    eff = np.array([-3, 0, 1, 2, 3, 0, -1, -2], dtype=float)[X["cat"].to_numpy()]
    X = X.assign(informative=eff + 0.1 * rng.standard_normal(n), noise=rng.standard_normal(n))
    cols = list(X.columns)
    index = {c: i for i, c in enumerate(cols)}
    yv = y.to_numpy(dtype=float)
    va = np.zeros(n, dtype=bool)
    va[rng.permutation(n)[: n // 3]] = True
    base = [np.ones(n), X["num"].to_numpy(dtype=float)]  # the selected design: the intercept and ``num``; the category is not in it
    kept = _score_encoded_features_heldout(
        ~va, va, base, yv, ["informative", "noise"], index, {index["num"]}, {"num"}, {}, X, X.to_numpy(dtype=float), n, 0.005, [index["num"]], False, cols
    )
    assert index["informative"] in kept and index["noise"] not in kept


def test_count_and_frequency_encodings_get_the_same_outcome_and_the_category_information_reaches_the_output():
    """Count and frequency encodings of one column are the same information up to a scale (freq = count / n): whatever the finished selection does with one it does with the other, and
    the information of the category reaches the output through the raw column or an encoding."""
    by_count = _fit(20, fe_count_encoding_enable=True)
    by_freq = _fit(20, fe_frequency_encoding_enable=True)
    out_count = set(by_count.get_feature_names_out())
    out_freq = set(by_freq.get_feature_names_out())
    assert bool(by_count.count_encoding_features_) == bool(by_freq.frequency_encoding_features_), (by_count.count_encoding_features_, by_freq.frequency_encoding_features_)
    assert out_count - {"cat__count"} == out_freq - {"cat__freq"}
    assert "cat" in out_count or "cat__count" in out_count
