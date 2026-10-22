"""The no-copy QR update of the raw-protection scorer equals a plain least-squares fit on the same rows."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._raw_protect_r2 import heldout_r2_scorer


def _reference(base, extra, y, tr, va):
    """Held-out R^2 of lstsq on [base | extra] trained on tr, scored on va."""
    cols = [base] if extra is None else [base, extra.reshape(len(y), -1)]
    a = np.column_stack(cols)
    coef = np.linalg.lstsq(a[tr], y[tr], rcond=None)[0]
    yv = y[va]
    return 1.0 - float(np.sum((yv - a[va] @ coef) ** 2)) / float(np.sum((yv - yv.mean()) ** 2))


@pytest.mark.parametrize("width", [1, 2, 3])
@pytest.mark.parametrize("p", [1, 4, 9])
def test_extra_columns_score_like_lstsq(p, width):
    """A single candidate column and blocks of columns, against designs of several widths, with an unrelated row split."""
    rng = np.random.default_rng(p * 10 + width)
    n = 4_000
    base = rng.normal(size=(n, p))
    extra = rng.normal(size=(n, width)) + 0.4 * base[:, :1]
    y = base @ rng.normal(size=p) + extra @ rng.normal(size=width) + rng.normal(size=n)
    mask = rng.random(n) < 0.7
    scorer = heldout_r2_scorer(base, y, mask, ~mask)
    got = scorer(extra if width > 1 else extra[:, 0])
    assert abs(got - _reference(base, extra, y, mask, ~mask)) < 1e-9
    assert abs(scorer() - _reference(base, None, y, mask, ~mask)) < 1e-9


def test_a_sequence_of_base_columns_is_accepted():
    """The base may be a list of length-n columns (the full-height design never exists)."""
    rng = np.random.default_rng(3)
    n = 1_500
    cols = [rng.normal(size=n) for _ in range(3)]
    y = cols[0] - 2 * cols[1] + rng.normal(size=n)
    mask = rng.random(n) < 0.6
    extra = rng.normal(size=n)
    got = heldout_r2_scorer(cols, y, mask, ~mask)(extra)
    assert abs(got - _reference(np.column_stack(cols), extra, y, mask, ~mask)) < 1e-9


def test_an_exact_copy_of_a_base_column_falls_back_to_the_minimum_norm_fit():
    """A collinear candidate leaves the QR path and still gets the rank-revealing answer."""
    rng = np.random.default_rng(4)
    n = 2_000
    base = rng.normal(size=(n, 3))
    y = base.sum(axis=1) + rng.normal(size=n)
    mask = rng.random(n) < 0.7
    got = heldout_r2_scorer(base, y, mask, ~mask)(base[:, 1].copy())
    assert abs(got - _reference(base, base[:, 1], y, mask, ~mask)) < 1e-9


def test_paired_gain_equals_the_r2_increment_and_separates_signal_from_noise():
    """The paired gain is the held-out R^2 increment; its standard error makes a real column many errors above zero and a noise column about one."""
    rng = np.random.default_rng(3)
    n = 3_000
    base = rng.normal(size=(n, 2))
    signal = rng.normal(size=n)
    noise = rng.normal(size=n)
    y = base @ np.array([1.0, -0.5]) + 0.6 * signal + rng.normal(size=n)
    mask = rng.random(n) < 0.67
    scorer = heldout_r2_scorer(base, y, mask, ~mask)
    for col in (signal, noise):
        gain, se = scorer.paired_gain(col)  # type: ignore[attr-defined]
        assert gain == pytest.approx(scorer(col) - scorer(None), abs=1e-12)
        assert se > 0.0
    gain_s, se_s = scorer.paired_gain(signal)  # type: ignore[attr-defined]
    gain_n, se_n = scorer.paired_gain(noise)  # type: ignore[attr-defined]
    assert gain_s / se_s > 5.0
    assert abs(gain_n / se_n) < 3.0
