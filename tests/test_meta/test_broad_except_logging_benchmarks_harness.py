"""Baseline-debt wave 13: closes the remaining genuine broad_except_swallow sites in the dev-only
``_benchmarks`` harness scripts (training/_benchmarks/, feature_selection/_benchmarks/), the last
category of non-MRMR broad_except_swallow debt outside the already-documented false-positive class
(model_card.py / _cb_pool.py / _repr.py / bench_pysr_fe.py / bench_bur_lambda_qual22.py's first
site / bench_mi_correction_miller_madow.py's first site, all of which already surface the
exception via their own return value).

Behavioural, not source-text, for every target: the actual failure path is triggered (psutil
unavailable, an unsupported binning method, a degenerate holdout split, ...) and the real return
value / log record is asserted, per this repo's own "test behaviour, not source text" rule. For the
handful of scripts confirmed unsafe or impossible to import as a package module, `_load_function()`
parses the file off disk, takes just the target function's definition and executes it in a
namespace that supplies its globals, so the real function body runs without importing the script:
``bench_mrmr.py`` triggers real GPU/kernel-tuning-cache work as a MODULE-LEVEL side effect (not
inside the target functions themselves); ``profile_wellbore_mrmr_only_100k.py`` reads a
machine-local, hardcoded data path at module level; ``round4_su_seeded_interactions_bench.py`` /
``round4_synergy_combine_bench.py`` import bare sibling-directory modules (``synth``,
``hybrid_selector``, ...) that only resolve when run as a script from that directory, not as a
package import; ``mrmr_largeN_campaign.py`` is a heavyweight campaign driver.
"""

from __future__ import annotations

import ast
import logging
import sys
import types

import numpy as np
import pytest

import mlframe

_MLFRAME_SRC_DIR = mlframe.__path__[0]


def _read(relpath: str) -> str:
    """Read a source file under `mlframe/` directly off disk, bypassing module import."""
    import os

    with open(os.path.join(_MLFRAME_SRC_DIR, relpath), encoding="utf-8") as fh:
        return fh.read()


def _load_function(relpath: str, name: str, namespace: dict):
    """Execute the definition of function ``name`` found in the parsed source file ``relpath`` inside ``namespace`` and return it."""
    tree = ast.parse(_read(relpath))
    matches = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, f"expected exactly one definition of {name} in {relpath}, found {len(matches)}"
    scope = dict(namespace)
    exec(compile(ast.Module(body=[matches[0]], type_ignores=[]), relpath, "exec"), scope)  # nosec B102 - runs this repo's own benchmark function
    return scope[name]


def _no_psutil(monkeypatch):
    """Make `import psutil` raise ImportError for the duration of the test, restored automatically."""
    import sys

    monkeypatch.setitem(sys.modules, "psutil", None)


def test_bench_arch_d_free_ram_bytes_logs_on_failure(monkeypatch, caplog):
    """`_free_ram_bytes` must log and fall back to 12 GB on a psutil probe failure."""
    from mlframe.training._benchmarks.bench_arch_d import _free_ram_bytes

    _no_psutil(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="mlframe.training._benchmarks.bench_arch_d"):
        out = _free_ram_bytes()
    assert out == 12 * 1024 * 1024 * 1024
    assert any("_free_ram_bytes: psutil probe failed" in rec.message for rec in caplog.records)


def test_bench_content_fingerprint_rss_mb_logs_on_failure(monkeypatch, caplog):
    """`_rss_mb` delegates to the shared ``mlframe._bench_rmse_shared.rss_mb``; exercised for real
    (not just a source-text delegation check) so a future re-implementation that silently drops the
    psutil-failure handling would be caught."""
    import math

    from mlframe._bench_rmse_shared import rss_mb
    from mlframe.training._benchmarks.bench_content_fingerprint import _rss_mb

    assert _rss_mb.func is rss_mb
    _no_psutil(monkeypatch)
    # The record's logger name is whichever logger THIS module's own `partial(rss_mb, logger)`
    # baked in (its own `logging.getLogger(__name__)`), not `_bench_rmse_shared`'s -- `rss_mb`
    # only ever logs through the logger its caller handed it.
    with caplog.at_level(logging.DEBUG, logger="mlframe.training._benchmarks.bench_content_fingerprint"):
        out = _rss_mb()
    assert math.isnan(out)
    assert any("rss_mb: psutil probe failed" in rec.message for rec in caplog.records)


def test_bench_drift_value_counts_microbench_logs_on_failure(caplog):
    """`_is_object_array_col` must log and return False on a dtype-probe failure (a column that
    does not exist raises KeyError inside the probe)."""
    import pandas as pd

    from mlframe.training._benchmarks.bench_drift_value_counts_microbench import _is_object_array_col

    df = pd.DataFrame({"a": [1, 2, 3]})
    with caplog.at_level(logging.DEBUG, logger="mlframe.training._benchmarks.bench_drift_value_counts_microbench"):
        out = _is_object_array_col(df, "missing_column")
    assert out is False
    assert any("_is_object_array_col: probe failed" in rec.message for rec in caplog.records)


def test_bench_lgb_dataset_polars_bridge_rss_mb_logs_on_failure(monkeypatch, caplog):
    """Same delegation as `test_bench_content_fingerprint_rss_mb_logs_on_failure`, this module's copy."""
    import math

    from mlframe._bench_rmse_shared import rss_mb
    from mlframe.training._benchmarks.bench_lgb_dataset_polars_bridge import _rss_mb

    assert _rss_mb.func is rss_mb
    _no_psutil(monkeypatch)
    # Same logger-identity note as `test_bench_content_fingerprint_rss_mb_logs_on_failure` above.
    with caplog.at_level(logging.DEBUG, logger="mlframe.training._benchmarks.bench_lgb_dataset_polars_bridge"):
        out = _rss_mb()
    assert math.isnan(out)
    assert any("rss_mb: psutil probe failed" in rec.message for rec in caplog.records)


def test_bench_rmse_shared_rss_mb_logs_on_failure(monkeypatch, caplog):
    """The shared ``rss_mb`` helper (used by both bench_content_fingerprint.py and
    bench_lgb_dataset_polars_bridge.py above) must log and return NaN on a psutil probe failure."""
    import math

    from mlframe._bench_rmse_shared import rss_mb

    _no_psutil(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="mlframe._bench_rmse_shared"):
        out = rss_mb(logging.getLogger("mlframe._bench_rmse_shared"))
    assert math.isnan(out)
    assert any("rss_mb: psutil probe failed" in rec.message for rec in caplog.records)


def test_bench_adaptive_nbins_ab_logs_on_failure(caplog):
    """The per-method edge computation must log and return None when `per_feature_edges` rejects
    an unsupported method name."""
    import numpy as np

    from mlframe.feature_selection._benchmarks.bench_adaptive_nbins_ab import _run_fold_ab

    rng = np.random.default_rng(0)
    x = rng.random(64)
    y = (x > 0.5).astype(int)
    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection._benchmarks.bench_adaptive_nbins_ab"):
        out = _run_fold_ab("not_a_real_binning_method", "baseline", x, y, x, y, "uniform", "linear", 64, 0)
    assert out is None
    assert any("bench_adaptive_nbins_ab: edge computation failed for method=" in rec.message for rec in caplog.records)


def test_bench_boruta_auto_dispatch_logs_on_failure(caplog):
    """`_honest_holdout_auc` must log and fall back to chance-level 0.5 when the refit/score raises
    -- a single-class ``y`` makes `train_test_split(..., stratify=y)` raise before any model fits."""
    import numpy as np
    import pandas as pd

    from mlframe.feature_selection._benchmarks.bench_boruta_auto_dispatch import _honest_holdout_auc

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((20, 3)), columns=["f0", "f1", "f2"])
    y = pd.Series([1] * 20)  # single class -> stratified split raises
    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection._benchmarks.bench_boruta_auto_dispatch"):
        out = _honest_holdout_auc(X, y, selected=["f0", "f1"], seed=0)
    assert out == 0.5
    assert any("_honest_holdout_auc: refit/score failed" in rec.message for rec in caplog.records)


def test_bench_bur_lambda_qual22_logs_on_failure(caplog):
    """`_downstream` must log and return NaN when scoring raises. The `try` block only wraps
    `roc_auc_score(...)`/`.predict_proba(...)`, NOT the `.fit()` call above it -- a single-class
    ``ytr`` or ``yte`` either raises OUTSIDE the try (fit) or is merely warned-and-NaN'd on the
    SUCCESS path by modern sklearn's `roc_auc_score` (`UndefinedMetricWarning`), neither of which
    reaches the except branch this test exists to pin. A held-out `Xte` with a different column
    count than what the model was fit on does: `.predict_proba` raises a feature-count
    `ValueError` squarely inside the try."""
    import numpy as np

    from mlframe.feature_selection._benchmarks.bench_bur_lambda_qual22 import _downstream

    rng = np.random.default_rng(0)
    Xtr = rng.random((20, 2))
    Xte = rng.random((10, 3))  # wrong feature count for the fitted model -> predict_proba raises
    ytr = (rng.random(20) > 0.5).astype(int)
    yte = (rng.random(10) > 0.5).astype(int)
    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection._benchmarks.bench_bur_lambda_qual22"):
        out = _downstream(Xtr, Xte, ytr, yte, "classification")
    import math

    assert math.isnan(out)
    assert any("_downstream: AUC scoring failed" in rec.message for rec in caplog.records)


def test_bench_fs_levers_dflip_logs_on_failure(caplog):
    """`_honest_metric` (fits MRMR with the given lever kwargs, then scores a Ridge on the
    selection) must log and return NaN when the fit/transform raises -- an unrecognised MRMR
    constructor kwarg raises `TypeError` immediately, the cheapest way to exercise the except
    branch without paying for a real MRMR fit."""
    import numpy as np

    from mlframe.feature_selection._benchmarks.bench_fs_levers_dflip import _honest_metric

    rng = np.random.default_rng(0)
    X = rng.random((20, 4))
    y = rng.random(20)
    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection._benchmarks.bench_fs_levers_dflip"):
        out = _honest_metric(X, y, {"not_a_real_mrmr_kwarg": 1}, seed=0)
    import math

    assert math.isnan(out)
    assert any("bench_fs_levers_dflip: config failed, scoring as nan" in rec.message for rec in caplog.records)


def test_bench_mi_correction_miller_madow_logs_on_failure(caplog):
    """Same `_downstream` shape and the same feature-count-mismatch trigger as
    `test_bench_bur_lambda_qual22_logs_on_failure` above, this module's copy."""
    import math

    import numpy as np

    from mlframe.feature_selection._benchmarks.bench_mi_correction_miller_madow import _downstream

    rng = np.random.default_rng(0)
    Xtr = rng.random((20, 2))
    Xte = rng.random((10, 3))  # wrong feature count for the fitted model -> predict_proba raises
    ytr = (rng.random(20) > 0.5).astype(int)
    yte = (rng.random(10) > 0.5).astype(int)
    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection._benchmarks.bench_mi_correction_miller_madow"):
        out = _downstream(Xtr, Xte, ytr, yte, "classification")
    assert math.isnan(out)
    assert any("_downstream: AUC scoring failed" in rec.message for rec in caplog.records)


def test_bench_mrmr_git_sha_and_gpu_model_log_on_failure(monkeypatch, caplog):
    """`_git_sha` and `_gpu_model` must both log on their respective probe failures and return their sentinel, and report the probe on success.

    The function bodies are executed without importing `bench_mrmr.py`: importing it runs real GPU
    kernel-tuning-cache lookups as a MODULE-LEVEL side effect, unrelated to and far heavier than these
    two probe functions.
    """
    import pathlib
    import subprocess

    relpath = "feature_selection/_benchmarks/bench_mrmr.py"
    logger = logging.getLogger("bench_mrmr_under_test")
    git_sha = _load_function(relpath, "_git_sha", {"subprocess": subprocess, "Path": pathlib.Path, "logger": logger, "__file__": __file__})
    gpu_model = _load_function(relpath, "_gpu_model", {"logger": logger})

    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: " abc123\n")
    assert git_sha() == "abc123"

    def _git_missing(*args, **kwargs):
        """Stand in for a missing git executable."""
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "check_output", _git_missing)
    fake_cupy = types.SimpleNamespace(
        cuda=types.SimpleNamespace(Device=lambda index: types.SimpleNamespace(id=index), runtime=types.SimpleNamespace(getDeviceProperties=lambda i: {"name": b"Fake GPU"}))
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    assert gpu_model() == "Fake GPU"
    monkeypatch.setitem(sys.modules, "cupy", None)
    with caplog.at_level(logging.DEBUG, logger="bench_mrmr_under_test"):
        assert git_sha() == "nogit"
        assert gpu_model() == "no-gpu"
    messages = [rec.getMessage() for rec in caplog.records]
    assert len(messages) == 2
    assert messages[0].startswith("_git_sha: git rev-parse failed")
    assert messages[1].startswith("_gpu_model: cupy device probe failed")


def test_bench_mrmr_threading_vs_loky_logs_on_failure(monkeypatch, caplog):
    """`_peak_rss_mb` must log and return NaN on a psutil probe failure."""
    from mlframe.feature_selection._benchmarks.bench_mrmr_threading_vs_loky import _peak_rss_mb

    _no_psutil(monkeypatch)
    with caplog.at_level(logging.DEBUG, logger="mlframe.feature_selection._benchmarks.bench_mrmr_threading_vs_loky"):
        out = _peak_rss_mb()
    import math

    assert math.isnan(out)
    assert any("_peak_rss_mb: psutil probe failed" in rec.message for rec in caplog.records)


def test_profile_wellbore_mrmr_only_100k_prints_on_failure(capsys):
    """The dump-audit hook must print (its own logging convention, gated behind
    ``WELLBORE_DUMP_AUDIT=1``) on introspection failure rather than silently swallowing it, and still call the wrapped reducer.

    The hook's definition is executed without importing the script: it reads a hardcoded, machine-local data
    path at MODULE level, so it cannot be imported as a package module anywhere but the machine it was written on.
    """
    import traceback

    reduced = []
    audited = _load_function(
        "feature_selection/_benchmarks/profile_wellbore_mrmr_only_100k.py",
        "_audited",
        {"traceback": traceback, "logger": logging.getLogger("wellbore_under_test"), "_orig_reduce": lambda self, a: reduced.append(a) or "reduced"},
    )

    class _BrokenArray(np.ndarray):
        """An ndarray whose size probe raises, as a failing introspection would."""

        @property
        def nbytes(self):
            """Fail the way a broken array introspection does."""
            raise RuntimeError("introspection exploded")

    broken = np.zeros(4).view(_BrokenArray)
    assert audited(None, broken) == "reduced"
    failed = capsys.readouterr().out
    assert failed.startswith("[dump-audit] audit hook failed: RuntimeError(")

    big = np.zeros(200_000)
    assert audited(None, big) == "reduced"
    ok = capsys.readouterr().out
    assert ok.startswith("[dump-audit] shape=(200000,)") and "audit hook failed" not in ok
    assert len(reduced) == 2


def test_round4_su_seeded_interactions_bench_logs_on_failure(caplog):
    """`_qbin` must log on a quantile-binning failure and return zeros, and bin a healthy column into quantile classes.

    The function is executed without importing the script: it does ``from synth import
    make_dataset`` / ``import fs_selectors as S`` -- bare sibling-directory module names that only
    resolve when run as a script from ``fs_hybrid/`` with that directory on ``sys.path``, not as a
    package import (importing it as
    ``mlframe.feature_selection._benchmarks.fs_hybrid.round4_su_seeded_interactions_bench`` raises
    ``ModuleNotFoundError: No module named 'synth'``).
    """
    checkpoints: list[str] = []
    qbin = _load_function(
        "feature_selection/_benchmarks/fs_hybrid/round4_su_seeded_interactions_bench.py",
        "_qbin",
        {"np": np, "logger": logging.getLogger("round4_su_under_test"), "ck": checkpoints.append},
    )
    healthy = qbin(np.arange(100), 4)
    assert sorted(set(healthy.tolist())) == [0, 1, 2, 3]
    assert checkpoints == []
    with caplog.at_level(logging.DEBUG, logger="round4_su_under_test"):
        failed = qbin(np.arange(10), "not-a-bin-count")
    assert failed.tolist() == [0] * 10
    assert len(checkpoints) == 1 and checkpoints[0].startswith("_qbin: quantile binning failed, returning zeros: TypeError(")
    assert caplog.messages[0].startswith("quantile binning failed: TypeError(")


def test_round4_synergy_combine_bench_logs_on_failure(caplog):
    """`combine_referee` must log on a held-out-split failure and fall back to keeping every voted representative.

    Same sibling-directory-relative-import shape as `test_round4_su_seeded_interactions_bench_logs_on_failure`
    above (``from round3_realdata_bench import ...``, ``from hybrid_selector import HybridSelector``)
    -- not importable as a package module, so the method body is executed on a stand-in object.
    """
    import pandas as pd

    combine_referee = _load_function(
        "feature_selection/_benchmarks/fs_hybrid/round4_synergy_combine_bench.py",
        "combine_referee",
        {"logger": logging.getLogger("round4_synergy_under_test")},
    )

    class _Stub:
        """Just the attributes ``combine_referee`` reads before its held-out split."""

        random_state = 0
        _Xaug_ = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0)[::-1]})
        _y_ = np.array([0] * 19 + [1])  # a class with one member makes the stratified split raise

        def _cluster_votes(self, member_sel):
            """Two clusters: one with two votes (consensus), one with a single vote (contested)."""
            return {"r1": [1, 2], "r2": [3]}

        def _emit(self, reps, cols):
            """Emit the representatives unchanged."""
            return list(reps)

    with caplog.at_level(logging.DEBUG, logger="round4_synergy_under_test"):
        kept = combine_referee(_Stub(), {}, ["a", "b"])
    assert kept == ["r1", "r2"]
    assert caplog.messages[0].startswith("combine_referee: held-out split failed, keeping all voted:")


def test_mrmr_largeN_campaign_logs_on_failure_and_reraises_when_requested(monkeypatch, caplog):
    """The per-cell downstream scorer must log and score None on failure, and re-raise when `MRMR_CAMPAIGN_RAISE` is set.

    The scorer is executed without importing the campaign driver; a selected-column index outside the feature matrix makes the
    downstream fit raise inside the guarded block, which is cheap and deterministic.
    """
    import os

    score = _load_function(
        "feature_selection/_benchmarks/fs_quality/mrmr_largeN_campaign.py",
        "_downstream_score",
        {"np": np, "os": os, "logger": logging.getLogger("campaign_under_test")},
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = (X[:, 0] + 0.1 * rng.normal(size=80) > 0).astype(int)

    monkeypatch.delenv("MRMR_CAMPAIGN_RAISE", raising=False)
    healthy = score(X[:50], y[:50], X[50:], y[50:], [0], "classification")
    assert healthy is not None and 0.5 < healthy <= 1.0
    with caplog.at_level(logging.DEBUG, logger="campaign_under_test"):
        assert score(X[:50], y[:50], X[50:], y[50:], [7], "classification") is None
    assert caplog.messages[0].startswith("campaign cell failed, scoring as None:")
    monkeypatch.setenv("MRMR_CAMPAIGN_RAISE", "1")
    with pytest.raises(IndexError):
        score(X[:50], y[:50], X[50:], y[50:], [7], "classification")
