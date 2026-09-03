"""Smoke test for mlframe.votenrank.fairness_computation (W5-4)."""

from __future__ import annotations

import pytest


def _fake_scorer(model, tokenizer, sentence):
    """Deterministic stand-in for a naive_*_score function: no GPU/model needed, exercises the
    naive_model_scores + *_pipeline aggregation logic (which is pure numpy over pre-computed scores)."""
    return float(len(sentence))


@pytest.mark.fast
def test_naive_model_scores_applies_scorer_to_each_sentence():
    """naive_model_scores maps the scorer over every sentence and returns a numpy array."""
    import numpy as np

    from mlframe.votenrank.fairness_computation import naive_model_scores

    sentences = ["a", "bb", "ccc"]
    out = naive_model_scores(None, None, sentences, _fake_scorer)
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.fast
def test_crows_pipeline_fraction_good_scored_higher():
    """crows_pipeline: fraction of pairs where the "bad" (stereotyping) sentence scores lower than "good"."""
    from mlframe.votenrank.fairness_computation import crows_pipeline

    # _fake_scorer scores by sentence length; "bad" shorter than "good" -> bad_scores < good_scores -> counted.
    good = ["long sentence one", "long sentence two"]
    bad = ["short", "tiny"]
    frac = crows_pipeline(None, None, good, bad, _fake_scorer)
    assert frac == pytest.approx(1.0)


@pytest.mark.fast
def test_stereo_pipeline_returns_lms_ss_icat():
    """stereo_pipeline returns the lms/ss/icat dict with values in the expected ranges."""
    from mlframe.votenrank.fairness_computation import stereo_pipeline

    good = ["aaaaaaaaaa"]  # len 10, scores lower than unrelated -> "preferred"
    bad = ["aaaaaaaaaaaaaaaaaaaa"]  # len 20, scores higher than unrelated -> "not preferred"
    unrelated = ["aaaaaaaaaaaaaaa"]  # len 15
    out = stereo_pipeline(None, None, _fake_scorer, good, bad, unrelated)
    assert set(out) == {"lms", "ss", "icat"}
    assert 0.0 <= out["lms"] <= 1.0
    assert 0.0 <= out["ss"] <= 1.0
    assert 0.0 <= out["icat"] <= 1.0
    # good (shorter) scores lower than unrelated -> preferred; bad (longer) scores higher -> not preferred.
    assert out["lms"] == pytest.approx(0.5)
    assert out["ss"] == pytest.approx(0.0)  # bad_scores < good_scores never happens (bad is longer -> higher score)


@pytest.mark.fast
def test_winobias_pipeline_per_side_accuracy():
    """winobias_pipeline: per-side (pro/anti) fraction where "good" scores lower than "bad"."""
    from mlframe.votenrank.fairness_computation import winobias_pipeline

    wb_data = {
        "pro": {"good": ["a"], "bad": ["aaaaa"]},  # good shorter -> good < bad -> counted
        "anti": {"good": ["aaaaa"], "bad": ["a"]},  # good longer -> good > bad -> not counted
    }
    out = winobias_pipeline(None, None, wb_data, _fake_scorer)
    assert set(out) == {"pro", "anti"}
    assert out["pro"] == pytest.approx(1.0)
    assert out["anti"] == pytest.approx(0.0)


@pytest.mark.fast
def test_import_fairness_computation_module():
    """Module imports cleanly and exposes its public callables.

    Heavy deps (transformers, torch) gated behind importorskip so CI without GPU/HF
    stack still passes.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from mlframe.votenrank import fairness_computation as fc

    for name in (
        "naive_masking_score",
        "naive_t5_score",
        "naive_gpt2_score",
        "naive_model_scores",
        "crows_pipeline",
        "stereo_pipeline",
        "winobias_pipeline",
    ):
        assert callable(getattr(fc, name)), f"{name} not callable"
