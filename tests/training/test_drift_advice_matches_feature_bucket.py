"""The drift snapshot gave categorical advice about columns that had just been promoted to text features.

A production log, one line apart:

    Promoted 5 high-cardinality column(s) from cat_features to text_features: [_raw_countries, ...]
    Category drift suspect: _raw_countries -- ... XGB/CB may crash when constructing val DMatrix with ref=train.
      c) drop '_raw_countries' entirely ... promote to text_features via use_text_features=True

All three claims were wrong for that column: it is no longer in cat_features, it was already promoted, and a
text feature never travels the categorical DMatrix path the crash warning describes. The scan deliberately covers
text features -- vocabulary drift is worth reporting -- so the advice, not the scan, is what had to change.
"""

from __future__ import annotations

import logging

import polars as pl
import pytest

from mlframe.training.core._phase_drift_snapshot import _log_cardinality_and_drift_snapshot
from mlframe.utils.log_throttle import reset_throttle_counts


def _frames():
    """train/val/test frames where a categorical and a text column both carry values train never saw.

    Cardinalities clear the drift gate (>=5 unseen values, or >=5% of the train vocabulary) and the columns
    are polars Categorical, which is what the snapshot scans.
    """
    def _cat(values):
        """A polars Categorical series, the dtype the drift scan looks at."""
        return pl.Series(values, dtype=pl.Categorical)

    train = pl.DataFrame({
        "region": _cat([f"r{i % 150}" for i in range(3000)]),
        "skills_text": _cat([f"tok{i % 2000}" for i in range(3000)]),
    })
    val = pl.DataFrame({
        "region": _cat([f"z{i % 80}" for i in range(600)]),
        "skills_text": _cat([f"new{i % 400}" for i in range(600)]),
    })
    test = pl.DataFrame({
        "region": _cat([f"r{i % 150}" for i in range(600)]),
        "skills_text": _cat([f"tok{i % 2000}" for i in range(600)]),
    })
    return train, val, test


@pytest.fixture
def messages(caplog):
    """Every line the snapshot logged, kept SEPARATE: a slice of one joined blob would let the categorical
    warning bleed into assertions about the text column and vice versa."""
    # The drift warnings go through log_throttle, which suppresses repeats process-wide -- without a reset the
    # second test in this module would observe an empty log and "pass" for the wrong reason.
    reset_throttle_counts()
    train, val, test = _frames()
    with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_drift_snapshot"):
        _log_cardinality_and_drift_snapshot(
            train_df=train, val_df=val, test_df=test,
            cat_features=["region"], text_features=["skills_text"], embedding_features=[],
        )
    return [r.getMessage() for r in caplog.records]


def _line_about(messages, marker: str) -> str:
    """The single logged line containing ``marker``; fails loudly when the snapshot said nothing about it."""
    hits = [m for m in messages if marker in m]
    assert hits, f"nothing logged about {marker!r}; logged: {messages}"
    return hits[0]


class TestTextFeaturesGetTextAdvice:
    """What a text column is told."""

    def test_it_is_not_told_to_be_promoted_to_text_features(self, messages):
        """It already is one; the advice was self-contradictory."""
        assert "use_text_features=True" not in _line_about(messages, "Text-feature vocabulary drift")

    def test_it_is_not_warned_about_a_dmatrix_crash(self, messages):
        """Text features do not travel the categorical DMatrix path."""
        assert "DMatrix" not in _line_about(messages, "Text-feature vocabulary drift")

    def test_its_drift_is_still_reported(self, messages):
        """Silence would be the wrong fix: unseen tokens are worth knowing about."""
        assert "skills_text" in _line_about(messages, "Text-feature vocabulary drift")

    def test_it_is_never_called_a_category_drift_suspect(self, messages):
        """The categorical warning must not fire for it at all."""
        assert not [m for m in messages if "Category drift suspect: skills_text" in m]


class TestCategoricalFeaturesKeepCategoricalAdvice:
    """The existing behaviour for real categoricals must be untouched."""

    def test_a_categorical_still_gets_its_healing_options(self, messages):
        """The actionable part of the categorical warning must survive the branch."""
        assert "suggested actions" in _line_about(messages, "Category drift suspect: region")

    def test_aligned_categorical_does_not_claim_a_crash(self, messages):
        """With the default train+val category-domain alignment the val DMatrix cannot crash; the line says so."""
        line = _line_about(messages, "Category drift suspect: region")
        assert "DMatrix" not in line
        assert "prevents a crash" in line

    def test_unaligned_categorical_is_still_warned_about_the_crash_path(self, caplog):
        """With alignment off the val-DMatrix crash is a genuine risk and must still be named."""
        reset_throttle_counts()
        train, val, test = _frames()
        with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_drift_snapshot"):
            _log_cardinality_and_drift_snapshot(
                train_df=train, val_df=val, test_df=test,
                cat_features=["region"], text_features=["skills_text"], embedding_features=[], align_categorical_dicts=False,
            )
        assert "DMatrix" in _line_about([r.getMessage() for r in caplog.records], "Category drift suspect: region")


def test_low_cardinality_aligned_drift_is_reported_as_handled(caplog):
    """A production log warned "XGB/CB may crash" and told the user to add an ``__UNSEEN__`` bucket for one unseen
    value of a 4-level column, which the shared train+val Enum domain already handles. That case is INFO now."""
    reset_throttle_counts()
    cat = lambda v: pl.Series(v, dtype=pl.Categorical)  # noqa: E731
    train = pl.DataFrame({"job_post_type": cat([f"t{i % 4}" for i in range(400)])})
    val = pl.DataFrame({"job_post_type": cat([f"t{i % 4}" for i in range(80)] + ["brand_new"] * 20)})
    test = pl.DataFrame({"job_post_type": cat([f"t{i % 4}" for i in range(100)])})
    with caplog.at_level(logging.INFO, logger="mlframe.training.core._phase_drift_snapshot"):
        _log_cardinality_and_drift_snapshot(
            train_df=train, val_df=val, test_df=test, cat_features=["job_post_type"], text_features=[], embedding_features=[],
        )
    records = [r for r in caplog.records if "job_post_type --" in r.getMessage()]
    assert records and all(r.levelno == logging.INFO for r in records)
    assert "Handled automatically" in records[0].getMessage()

    def test_a_categorical_does_not_get_the_text_line(self, messages):
        """Each column gets exactly one kind of advice."""
        assert "region" not in _line_about(messages, "Text-feature vocabulary drift")
