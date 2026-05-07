"""Hypothesis-driven fuzz of multi-target panel composers.

Goal: catch composer crashes that show up only on rare subsets of
panel tokens (e.g. some panel needs ≥ 2 classes; some breaks on
all-zero predictions; PROB_DIST blows up if any class has n=0
samples after subsampling).

For each of (multiclass / multilabel / LTR) composers, we generate:
- a random non-empty subset of ALLOWED_*_PANEL_TOKENS as the template
- a synthetic dataset that's small but valid for that target type
- assert the composer returns a non-empty FigureSpec without raising

Hypothesis ``settings(max_examples=...)`` keeps each test under a
few seconds. ``deadline=None`` because compose work + matplotlib
import time can spike past the default 200ms cap on cold runs.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from mlframe.reporting.charts.ltr import (
    ALLOWED_LTR_PANEL_TOKENS, compose_ltr_figure,
)
from mlframe.reporting.charts.multiclass import (
    ALLOWED_MULTICLASS_PANEL_TOKENS, compose_multiclass_figure,
)
from mlframe.reporting.charts.multilabel import (
    ALLOWED_MULTILABEL_PANEL_TOKENS, compose_multilabel_figure,
)
from mlframe.reporting.spec import FigureSpec


# ----------------------------------------------------------------------------
# Hypothesis strategies for panel templates
# ----------------------------------------------------------------------------


def _token_subset_strategy(allowed: frozenset) -> st.SearchStrategy:
    """Generate a non-empty subset of ``allowed`` as a space-separated str."""
    sorted_tokens = sorted(allowed)
    return st.lists(
        st.sampled_from(sorted_tokens),
        min_size=1, max_size=len(sorted_tokens), unique=True,
    ).map(lambda lst: " ".join(lst))


_HSETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)


# ----------------------------------------------------------------------------
# Fixed inputs (do not regenerate per hypothesis example -- composers
# care about the templates, the fixture stays valid for any subset).
# ----------------------------------------------------------------------------


def _mc_inputs():
    rng = np.random.default_rng(0)
    n, K = 200, 3
    y = rng.integers(0, K, n)
    proba = rng.dirichlet(alpha=[1] * K, size=n)
    for i, t in enumerate(y):
        proba[i, t] += 0.7
        proba[i] /= proba[i].sum()
    return y, proba, ["a", "b", "c"]


def _ml_inputs():
    rng = np.random.default_rng(0)
    n, K = 200, 3
    y = rng.integers(0, 2, (n, K)).astype(np.int8)
    proba = np.clip(rng.uniform(0, 0.5, (n, K)) + y * 0.4, 0.01, 0.99)
    return y, proba, ["x", "y", "z"]


def _ltr_inputs():
    rng = np.random.default_rng(0)
    y, score, gid = [], [], []
    for q in range(40):
        sz = int(rng.integers(4, 9))
        rels = rng.integers(0, 4, sz)
        scores = rels.astype(float) + rng.normal(0, 0.5, sz)
        y.extend(rels.tolist())
        score.extend(scores.tolist())
        gid.extend([q] * sz)
    return np.asarray(y), np.asarray(score, dtype=np.float64), np.asarray(gid)


# ----------------------------------------------------------------------------
# Multiclass fuzz
# ----------------------------------------------------------------------------


class TestMulticlassFuzz:
    @_HSETTINGS
    @given(template=_token_subset_strategy(ALLOWED_MULTICLASS_PANEL_TOKENS))
    def test_random_subset_composes(self, template):
        y, proba, classes = _mc_inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = compose_multiclass_figure(
                y, proba, classes, panels_template=template,
            )
        assert isinstance(spec, FigureSpec)
        # Token count == panel count (after grid pack, padded with None).
        n_tokens = len(template.split())
        n_panels_set = sum(
            1 for row in spec.panels for cell in row if cell is not None
        )
        assert n_panels_set == n_tokens


# ----------------------------------------------------------------------------
# Multilabel fuzz
# ----------------------------------------------------------------------------


class TestMultilabelFuzz:
    @_HSETTINGS
    @given(template=_token_subset_strategy(ALLOWED_MULTILABEL_PANEL_TOKENS))
    def test_random_subset_composes(self, template):
        y, proba, labels = _ml_inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = compose_multilabel_figure(
                y, proba, labels, panels_template=template,
            )
        assert isinstance(spec, FigureSpec)
        n_tokens = len(template.split())
        n_panels_set = sum(
            1 for row in spec.panels for cell in row if cell is not None
        )
        assert n_panels_set == n_tokens


# ----------------------------------------------------------------------------
# LTR fuzz
# ----------------------------------------------------------------------------


class TestLTRFuzz:
    @_HSETTINGS
    @given(template=_token_subset_strategy(ALLOWED_LTR_PANEL_TOKENS))
    def test_random_subset_composes(self, template):
        y, score, gid = _ltr_inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec = compose_ltr_figure(
                y, score, gid, panels_template=template,
            )
        assert isinstance(spec, FigureSpec)
        n_tokens = len(template.split())
        n_panels_set = sum(
            1 for row in spec.panels for cell in row if cell is not None
        )
        assert n_panels_set == n_tokens


# ----------------------------------------------------------------------------
# Edge cases (single-token templates per axis)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("token", sorted(ALLOWED_MULTICLASS_PANEL_TOKENS))
def test_multiclass_each_token_alone(token):
    y, proba, classes = _mc_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = compose_multiclass_figure(y, proba, classes, panels_template=token)
    assert sum(1 for row in spec.panels for c in row if c is not None) == 1


@pytest.mark.parametrize("token", sorted(ALLOWED_MULTILABEL_PANEL_TOKENS))
def test_multilabel_each_token_alone(token):
    y, proba, labels = _ml_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = compose_multilabel_figure(y, proba, labels, panels_template=token)
    assert sum(1 for row in spec.panels for c in row if c is not None) == 1


@pytest.mark.parametrize("token", sorted(ALLOWED_LTR_PANEL_TOKENS))
def test_ltr_each_token_alone(token):
    y, score, gid = _ltr_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = compose_ltr_figure(y, score, gid, panels_template=token)
    assert sum(1 for row in spec.panels for c in row if c is not None) == 1
