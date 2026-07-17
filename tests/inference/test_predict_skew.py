"""Regression tests for the predict-entry skew fixes (slug-keys, quantile-ens post-aggregation,
chunked inference, probe-eager model_names filter)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_combine_probs_quantile_post_aggregation_sorts_crossings():
    """Two members both have monotone quantiles per-row; mean averaging would NOT preserve monotonicity in
    general (counterexamples exist where averaging two monotone sequences breaks monotonicity at boundary
    points). The post-aggregation fix_quantile_crossing(sort) ensures the ensembled output is monotone
    regardless of member monotonicity."""
    from mlframe.training.core.predict import _combine_probs

    # Construct a counterexample: member A has rows like [1, 2, 5]; member B has rows like [3, 2, 4].
    # Average is [2, 2, 4.5]; that's monotone but only because of choice. Use a stronger crossing case:
    # A = [1, 5, 6], B = [3, 0, 5] -> mean = [2, 2.5, 5.5] still monotone; pick A=[1,5,2], B=[5,1,3]:
    # this case A itself is NOT monotone -- skip. Use A=[1,2,3], B=[2,1,4] (B not monotone) -> mean
    # [1.5, 1.5, 3.5]; still monotone. So construct directly: pass already-crossing array via member.
    member_a = np.array([[1.0, 5.0, 4.0]])  # crossing at col 1->2
    member_b = np.array([[2.0, 6.0, 5.0]])
    alphas = [0.1, 0.5, 0.9]

    out = _combine_probs([member_a, member_b], "mean", quantile_alphas=alphas)
    # After fix_quantile_crossing(sort): row 0 must be monotone non-decreasing.
    assert np.all(np.diff(out[0]) >= 0), f"post-aggregation output must be monotone; got {out[0]}"


def test_combine_probs_no_quantile_alphas_skips_crossing_fix():
    """When quantile_alphas is None the function returns the raw ensemble (back-compat).

    Member values are kept in [0, 1] so the classification-mode probability
    clip (``ensure_prob_limits=True`` when quantile_alphas is None) doesn't
    collapse them all to 1.0 -- the prior test used values >1 that landed
    in the clip floor and made the "not sorted" assertion impossible to
    satisfy.
    """
    from mlframe.training.core.predict import _combine_probs

    member_a = np.array([[0.1, 0.5, 0.4]])
    member_b = np.array([[0.2, 0.6, 0.5]])
    out_no_alpha = _combine_probs([member_a, member_b], "mean", quantile_alphas=None)
    out_no_alpha_arr = np.asarray(out_no_alpha)
    # Should be exact mean -- raw, NOT sorted. Mean is [0.15, 0.55, 0.45],
    # so [0, 2]=0.45 < [0, 1]=0.55 confirms the crossing is preserved.
    assert out_no_alpha_arr[0, 2] < out_no_alpha_arr[0, 1], f"without quantile_alphas the output should NOT be sorted; got {out_no_alpha_arr[0]}"


def test_resolve_quantile_alphas_metadata_lookup():
    """Metadata-stored quantile alphas resolve correctly."""
    from mlframe.training.core.predict import _resolve_quantile_alphas

    md = {"quantile_alphas": {"quantile_regression": {"my_target": [0.1, 0.5, 0.9]}}}
    out = _resolve_quantile_alphas(md, "quantile_regression", "my_target")
    assert out == [0.1, 0.5, 0.9]


def test_resolve_quantile_alphas_model_introspection():
    """When metadata is absent the helper falls back to model attribute introspection."""
    from mlframe.training.core.predict import _resolve_quantile_alphas

    class _Inner:
        quantile_alpha = [0.05, 0.5, 0.95]

    class _Wrap:
        model = _Inner()

    out = _resolve_quantile_alphas({}, "quantile_regression", "t", _Wrap())
    assert out == [0.05, 0.5, 0.95]


def test_resolve_quantile_alphas_skips_non_quantile_target():
    """Non-quantile target types short-circuit to None even when metadata has stale alphas."""
    from mlframe.training.core.predict import _resolve_quantile_alphas

    md = {"quantile_alphas": [0.1, 0.5, 0.9]}
    assert _resolve_quantile_alphas(md, "binary_classification", "t") is None


def test_slice_frame_polars_and_pandas_match():
    """The slicing helper must produce equivalent rows for polars + pandas."""
    pl = pytest.importorskip("polars")
    from mlframe.training.core.predict import _slice_frame

    pdf = pd.DataFrame({"x": np.arange(100), "y": np.arange(100, 200)})
    pldf = pl.from_pandas(pdf)

    out_pd = _slice_frame(pdf, 10, 20)
    out_pl = _slice_frame(pldf, 10, 20)
    assert len(out_pd) == 20
    assert len(out_pl) == 20
    assert list(out_pd["x"]) == list(out_pl.get_column("x"))


def test_concat_probs_dicts_handles_missing_keys():
    """Concat helper preserves keys that only some batches produced (a model that crashed on batch-2 still
    contributes its batch-1 output)."""
    from mlframe.training.core.predict import _concat_probs_dicts

    batch1 = {"m1": np.zeros(10), "m2": np.ones(10)}
    batch2 = {"m1": np.zeros(5)}  # m2 missing -- m2 only has batch-1 data
    out = _concat_probs_dicts([batch1, batch2])
    assert out["m1"].shape == (15,)
    assert out["m2"].shape == (10,)


def test_combine_probs_quantile_passes_through_when_shape_mismatch():
    """If combined.shape[1] != len(alphas) the helper does NOT touch the output (defensive)."""
    from mlframe.training.core.predict import _combine_probs

    # (N, K=2) but alphas=[0.1, 0.5, 0.9] (3 alphas) -> mismatch, should not invoke fix_quantile_crossing.
    member_a = np.array([[1.0, 5.0]])
    member_b = np.array([[3.0, 2.0]])
    out = _combine_probs([member_a, member_b], "mean", quantile_alphas=[0.1, 0.5, 0.9])
    # Output should be raw mean (no shape coercion): [[2.0, 3.5]]
    np.testing.assert_allclose(out, [[2.0, 3.5]])
