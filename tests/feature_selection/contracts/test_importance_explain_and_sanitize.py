"""Unit + smoke tests for the three previously-untested helpers in
``mlframe.feature_selection.importance``:

  - ``_sanitize_for_filename`` -- the path-traversal / Windows-reserved-char
    guard applied to a model_name before it becomes a ``reports/*.png`` filename.
  - ``explain_top_feature_importances`` -- SHAP beeswarm chart writer (CWD-relative).
  - ``show_shap_beeswarm_plot`` -- covered transitively by the explain-smoke test.

Behavioral assertions only (no ``inspect.getsource`` string checks). The
sanitizer's ALLOWLIST regex deliberately KEEPS ``[ ] @ = . _`` and space (a
real fi_name carries ``@iter=`` and ``[NF]``); these tests therefore pin only
the stripping of true path-traversal + Windows-reserved characters, the length
cap, and the empty/whitespace fallback -- never assert the kept chars are gone.
"""

from __future__ import annotations

import glob
import os

import matplotlib

# Force the Agg backend before pyplot import so the SHAP chart renders headless
# on CI / Windows with no display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.importance import _sanitize_for_filename

pytestmark = pytest.mark.uses_matplotlib


# ----------------------------------------------------------------------------
# _sanitize_for_filename -- pure unit, no deps
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, banned_substrings",
    [
        # Forward-slash path traversal: every "/" must be stripped, so neither a
        # bare "/" nor a "../" traversal sequence can survive.
        ("../../etc/passwd", ("/", "../")),
        # Backslash path traversal (Windows): every "\\" must be stripped.
        ("..\\..\\windows\\system32", ("\\", "..\\")),
        # Each Windows-reserved char must be stripped.
        ('na<m>e:q"t|p?z*x', ("<", ">", ":", '"', "|", "?", "*")),
    ],
)
def test_sanitize_strips_path_and_reserved_chars(raw, banned_substrings):
    out = _sanitize_for_filename(raw)
    for bad in banned_substrings:
        assert bad not in out, f"sanitized {raw!r} -> {out!r} still contains banned substring {bad!r}"


@pytest.mark.parametrize("char", list('<>:"|?*'))
def test_sanitize_removes_each_windows_reserved_char(char):
    """Every Windows-reserved character, isolated, must be removed."""
    raw = f"model{char}name"
    out = _sanitize_for_filename(raw)
    assert char not in out, f"reserved char {char!r} survived sanitization: {out!r}"


def test_sanitize_truncates_to_120_chars():
    """A 200-char input is truncated to the 120-char cap (default max_len)."""
    out = _sanitize_for_filename("a" * 200)
    assert len(out) == 120, f"expected exactly 120 chars after truncation; got {len(out)}"
    assert out == "a" * 120


@pytest.mark.parametrize("raw", ["", "   ", "...", " . . "])
def test_sanitize_empty_or_whitespace_falls_back_to_unnamed(raw):
    """Empty input, or input made only of the chars ``strip(' .')`` removes
    (spaces and dots), collapses to the 'unnamed' fallback once the cleaned
    string is empty. Note: tab/newline are NOT allowlisted, so they become a
    literal ``_`` (a valid filename char) and do NOT trigger the fallback --
    see ``test_sanitize_non_allowlisted_whitespace_becomes_underscore``."""
    out = _sanitize_for_filename(raw)
    assert out == "unnamed", f"empty/whitespace input {raw!r} must fall back to 'unnamed'; got {out!r}"


def test_sanitize_non_allowlisted_whitespace_becomes_underscore():
    """Tab / newline are outside the allowlist, so a run of them maps to a single
    ``_``; since ``strip(' .')`` only strips spaces and dots, the ``_`` survives
    and the result is NOT the 'unnamed' fallback."""
    out = _sanitize_for_filename("\t\n ")
    assert out == "_", f"non-allowlisted whitespace must sanitize to '_'; got {out!r}"


def test_sanitize_strips_trailing_dot_and_space():
    """The ``strip(' .')`` defuses the trailing-dot hidden-file / Windows
    trailing-space-or-dot trick."""
    out = _sanitize_for_filename("model . ")
    assert out == "model", f"trailing ' .' must be stripped; got {out!r}"
    assert not out.endswith(".")
    assert not out.endswith(" ")


def test_sanitize_keeps_allowlisted_fi_name_chars():
    """The allowlist deliberately KEEPS ``[ ] @ = . _`` and space because a real
    fi_name carries ``@iter=`` and ``[NF]``. Pin that they survive verbatim so a
    future tightening of the regex doesn't silently corrupt legitimate names."""
    raw = "MyModel GradientBoostingClassifier @iter=17 [5F]"
    out = _sanitize_for_filename(raw)
    assert out == raw, f"allowlisted fi_name must round-trip unchanged; got {out!r}"


# ----------------------------------------------------------------------------
# explain_top_feature_importances (+ show_shap_beeswarm_plot transitively)
# ----------------------------------------------------------------------------


class _ModelStub:
    """Minimal carrier matching the ``.model / .metrics / .columns`` contract
    that ``explain_top_feature_importances`` reads:

      - ``model.model``              -> the fitted estimator handed to TreeExplainer
      - ``model.metrics.get('best_iter','')`` -> stamped into the chart title
      - ``len(model.columns)``       -> the ``[NF]`` feature-count token
    """

    def __init__(self, estimator, columns, metrics):
        self.model = estimator
        self.columns = columns
        self.metrics = metrics


def _build_tree_classifier_stub(n: int = 120, p: int = 5, seed: int = 0):
    """A tiny GradientBoostingClassifier (TreeExplainer-supported) on a 2-signal
    binary target, wrapped in the stub contract."""
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + X[:, 1] * 0.5 > 0).astype(int)
    cols = [f"f{i}" for i in range(p)]
    X_df = pd.DataFrame(X, columns=cols)
    clf = GradientBoostingClassifier(n_estimators=20, max_depth=3, random_state=seed).fit(X_df, y)
    return _ModelStub(clf, cols, {"best_iter": 17}), X_df


@pytest.mark.slow
def test_explain_top_feature_importances_writes_one_sanitized_png(monkeypatch, tmp_path):
    """Smoke: ``save_chart=True`` writes exactly one ``reports/*_shap_beeswarm.png``
    (>1KB) into a CWD-relative ``reports/`` dir, and the filename is sanitized
    (no path-traversal / reserved chars leak from the model_name into the path).

    Floor 1KB; measured ~224KB on the reference build -- the floor only catches a
    silently-empty/corrupt figure, not normal rendering variation.
    """
    pytest.importorskip("shap")
    from mlframe.feature_selection.importance import explain_top_feature_importances

    stub, X_df = _build_tree_classifier_stub()
    # A hostile model_name that, unsanitized, would escape ``reports/`` upward.
    hostile_name = "../../evil:name|with?bad*chars"
    monkeypatch.chdir(tmp_path)
    try:
        explain_top_feature_importances(stub, hostile_name, X_df, save_chart=True, figsize=(8, 6))

        reports_dir = tmp_path / "reports"
        assert reports_dir.is_dir(), "explain_top_feature_importances must create a CWD-relative reports/ dir"

        pngs = glob.glob(str(reports_dir / "*_shap_beeswarm.png"))
        assert len(pngs) == 1, f"expected exactly one *_shap_beeswarm.png; got {pngs}"

        png = pngs[0]
        assert os.path.getsize(png) > 1024, f"rendered beeswarm PNG must be >1KB; got {os.path.getsize(png)} bytes"

        # The written file must live INSIDE tmp_path/reports -- the hostile name's
        # traversal sequences must not have escaped the directory.
        png_real = os.path.realpath(png)
        reports_real = os.path.realpath(str(reports_dir))
        assert png_real.startswith(reports_real + os.sep), f"sanitized output must stay inside reports/: {png_real} not under {reports_real}"

        # The leaf filename must not carry any path-separator or Windows-reserved char.
        leaf = os.path.basename(png)
        for bad in ("/", "\\", ":", "|", "?", "*", "<", ">", '"'):
            # ":" can legitimately appear in a Windows drive prefix of the FULL path,
            # but never in the leaf filename we constructed.
            assert bad not in leaf, f"sanitized leaf filename {leaf!r} must not contain {bad!r}"
    finally:
        plt.close("all")


@pytest.mark.slow
def test_explain_top_feature_importances_save_chart_false_writes_nothing(monkeypatch, tmp_path):
    """``save_chart=False`` renders the figure but writes no file to disk."""
    pytest.importorskip("shap")
    from mlframe.feature_selection.importance import explain_top_feature_importances

    stub, X_df = _build_tree_classifier_stub()
    monkeypatch.chdir(tmp_path)
    try:
        explain_top_feature_importances(stub, "NoSaveModel", X_df, save_chart=False, figsize=(8, 6))
        assert not (tmp_path / "reports").exists(), "save_chart=False must not create reports/ nor write any PNG"
    finally:
        plt.close("all")


def test_explain_top_feature_importances_fast_representative(monkeypatch, tmp_path):
    """Fast-mode representative of the SHAP-writer smoke (kept unmarked so a
    ``MLFRAME_FAST=1`` run still exercises the explain+sanitize write path on a
    minimal estimator). Asserts the single sanitized PNG lands in reports/."""
    pytest.importorskip("shap")
    from mlframe.feature_selection.importance import explain_top_feature_importances

    stub, X_df = _build_tree_classifier_stub(n=60, p=3)
    monkeypatch.chdir(tmp_path)
    try:
        explain_top_feature_importances(stub, "FastModel", X_df, save_chart=True, figsize=(6, 4))
        pngs = glob.glob(str(tmp_path / "reports" / "*_shap_beeswarm.png"))
        assert len(pngs) == 1, f"expected exactly one beeswarm PNG; got {pngs}"
        assert os.path.getsize(pngs[0]) > 1024, f"PNG must be >1KB; got {os.path.getsize(pngs[0])} bytes"
    finally:
        plt.close("all")
