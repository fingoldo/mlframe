"""biz_value test for ``preprocessing.gaussian_power_transform_search``.

Synthetic: a heavily right-skewed (log-normal-like) feature. The unsupervised search should pick a transform
that drives absolute skewness far below the raw feature's skewness -- no target/model involved, purely a
distribution-shape claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import skew

from mlframe.preprocessing.gaussian_power_transform_search import apply_gaussian_power_transform, gaussian_power_transform_search


def test_biz_val_gaussian_power_transform_search_reduces_skew_on_lognormal_feature():
    rng = np.random.default_rng(0)
    raw = np.exp(rng.normal(loc=0.0, scale=1.0, size=5000))  # classic log-normal, heavily right-skewed
    df = pd.DataFrame({"x": raw})

    raw_abs_skew = float(abs(skew(raw)))
    result = gaussian_power_transform_search(df)

    assert "x" in result
    best_abs_skew = result["x"]["best_abs_skew"]
    assert best_abs_skew < raw_abs_skew * 0.15, f"expected the best transform to cut abs-skew by >=85% vs raw ({raw_abs_skew:.4f}), got {best_abs_skew:.4f}"
    assert result["x"]["best_transform"] != "identity", "expected a non-identity transform to win on a heavily skewed feature"

    transformed_df = apply_gaussian_power_transform(df, result)
    transformed_abs_skew = float(abs(skew(transformed_df["x"].to_numpy())))
    assert transformed_abs_skew == pytest.approx(best_abs_skew, rel=1e-6)


def test_gaussian_power_transform_search_identity_wins_on_already_gaussian_feature():
    rng = np.random.default_rng(1)
    raw = rng.normal(loc=0.0, scale=1.0, size=5000)
    df = pd.DataFrame({"x": raw})

    result = gaussian_power_transform_search(df)
    assert result["x"]["best_abs_skew"] < 0.1


def test_gaussian_power_transform_search_skips_columns_with_too_few_finite_values():
    df = pd.DataFrame({"x": [np.nan, np.nan, 1.0]})
    result = gaussian_power_transform_search(df)
    assert "x" not in result


def test_gaussian_power_transform_search_boxcox_skipped_for_non_positive_column():
    df = pd.DataFrame({"x": np.concatenate([np.array([-1.0, 0.0]), np.random.default_rng(2).normal(size=200)])})
    result = gaussian_power_transform_search(df)
    assert "boxcox" not in result["x"]["all_abs_skew"]


def _make_linear_in_raw_scale_dataset(seed: int = 0, n: int = 5000):
    """Lognormal feature ``x`` whose target is linear in the RAW scale (``y = 3*x + small_noise``).

    Skew-only search picks a log/sqrt-family transform (skew fix on a lognormal is dramatic); but that same
    transform badly weakens the linear relationship with ``y``, since the true relationship is linear in ``x``,
    not in ``log(x)``/``sqrt(x)``. This is the exact bug class the target-correlation guard exists to catch --
    Spearman rank correlation can't see it (every candidate transform is monotonic, so rank corr is invariant),
    only Pearson can.
    """
    rng = np.random.default_rng(seed)
    x = np.exp(rng.normal(loc=0.0, scale=1.0, size=n))
    noise = rng.normal(loc=0.0, scale=0.1, size=n)
    y = 3.0 * x + noise
    return pd.DataFrame({"x": x}), y


def test_biz_val_gaussian_power_transform_search_target_guard_rejects_signal_destroying_transform():
    from scipy.stats import pearsonr

    df, y = _make_linear_in_raw_scale_dataset()

    raw_target_corr = float(abs(pearsonr(df["x"].to_numpy(), y)[0]))
    assert raw_target_corr > 0.9, "fixture sanity: raw x must be strongly linearly related to y"

    naive_result = gaussian_power_transform_search(df)
    naive_pick = naive_result["x"]["best_transform"]
    assert naive_pick != "identity", "fixture sanity: skew-only search must pick a non-identity transform"
    naive_pick_corr = float(abs(pearsonr(_apply_transform_for_test(df["x"].to_numpy(), naive_pick), y)[0]))
    assert naive_pick_corr < 0.85 * raw_target_corr, (
        f"fixture sanity: the skew-only pick ({naive_pick}) must materially weaken the target correlation, "
        f"got {naive_pick_corr:.4f} vs raw {raw_target_corr:.4f}"
    )

    guarded_result = gaussian_power_transform_search(df, y=y, require_target_correlation_retention=0.9)
    guarded_pick = guarded_result["x"]["best_transform"]
    assert naive_pick in guarded_result["x"]["target_correlation_rejected"], (
        f"expected the naive skew-only pick ({naive_pick}) to be rejected by the target-correlation guard"
    )
    assert guarded_pick != naive_pick, "expected the guard to steer away from the signal-destroying naive pick"
    guarded_pick_corr = guarded_result["x"]["all_target_corr"][guarded_pick]
    assert guarded_pick_corr >= 0.9 * raw_target_corr, (
        f"expected the guarded pick ({guarded_pick}) to retain >=90% of the raw target correlation "
        f"({raw_target_corr:.4f}), got {guarded_pick_corr:.4f}"
    )


def _apply_transform_for_test(x: np.ndarray, transform_name: str) -> np.ndarray:
    from mlframe.preprocessing.gaussian_power_transform_search import _apply_transform

    out = _apply_transform(x, transform_name)
    assert out is not None
    return out


def test_gaussian_power_transform_search_target_guard_default_off_is_bit_identical():
    """Omitting y/require_target_correlation_retention must reproduce the exact prior unsupervised result."""
    df, y = _make_linear_in_raw_scale_dataset()
    without_guard_args = gaussian_power_transform_search(df)
    without_y_kw = gaussian_power_transform_search(df, y=None, require_target_correlation_retention=None)
    assert without_guard_args == without_y_kw


def test_gaussian_power_transform_search_target_guard_requires_y():
    df, _ = _make_linear_in_raw_scale_dataset()
    with pytest.raises(ValueError):
        gaussian_power_transform_search(df, require_target_correlation_retention=0.9)
