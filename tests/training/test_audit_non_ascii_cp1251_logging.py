"""Wave 71 (2026-05-21): non-ASCII chars reaching print() / logger crash on
Windows cp866 (Russian CMD default) console with UnicodeEncodeError.

Per memory rule feedback_windows_encoding, non-ASCII Python print output on
Windows crashes. Audit found 2 em-dash (U+2014) findings:

  1. P0: feature_selection/_benchmarks/bench_mrmr_threading_vs_loky.py:114
     print(...) with em-dash, fires unconditionally on every invocation;
     also was using `{n}` without f-string prefix (separate cosmetic).
     Replaced em-dash with ASCII `--`, added missing f-prefix.

  2. P1: training/pipeline.py:535
     logger.warning(...) with em-dash inside fallback branch; logger
     handler uses console encoding by default, so emits to a cp866-active
     CMD would crash. Replaced em-dash with `--`.

Combined check: zero non-ASCII chars now appear inside print() / logger
arguments in production paths.
"""

from __future__ import annotations

from pathlib import Path

MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def test_bench_mrmr_no_em_dash_in_print(monkeypatch, capsys) -> None:
    """The MRMR threading-vs-loky benchmark prints only characters a cp866 / cp1251 console encodes, and its run-3 header interpolates n_jobs."""
    import sys

    import pandas as pd

    from mlframe.feature_selection._benchmarks import bench_mrmr_threading_vs_loky as bench

    monkeypatch.setattr(bench, "_build_frame", lambda n_rows, seed: (pd.DataFrame({"a": [0.0]}), pd.Series([0])))
    monkeypatch.setattr(
        bench,
        "_run_one",
        lambda df, y, backend, n_jobs, verbose: {"backend": backend, "wall_s": 1.0, "rss_before_mb": 1.0, "rss_after_mb": 2.0, "rss_delta_mb": 1.0, "n_selected": 1},
    )
    monkeypatch.setattr(sys, "argv", ["bench", "--n-rows", "10", "--n-jobs", "3"])
    bench.main()
    out = capsys.readouterr().out
    assert out
    assert out.encode("cp866")
    assert out.encode("cp1251")
    run3 = [line for line in out.splitlines() if line.startswith("--- run 3")]
    assert run3 == ["--- run 3: backend=loky n_jobs=3 (legacy default -- may break on env w/ pickle issues) ---"]


def test_pipeline_to_pandas_fallback_no_em_dash(monkeypatch, caplog) -> None:
    """The wide-frame to_pandas() fallback warning is emitted in plain ASCII with the failure reason and the polars version."""
    import logging

    import polars as pl

    from mlframe.training.configs import PreprocessingExtensionsConfig
    from mlframe.training.pipeline import apply_preprocessing_extensions

    real_to_pandas = pl.DataFrame.to_pandas

    def old_polars_to_pandas(self, *args, **kwargs):
        """Reject the split-blocks keywords the way a polars older than 0.20.4 does."""
        if "split_blocks" in kwargs:
            raise TypeError("to_pandas() got an unexpected keyword argument 'split_blocks'")
        return real_to_pandas(self, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "to_pandas", old_polars_to_pandas)
    frame = pl.DataFrame({"x0": [float(i) for i in range(30)], "x1": [float(i % 5) for i in range(30)]})
    with caplog.at_level(logging.WARNING):
        apply_preprocessing_extensions(frame, None, None, PreprocessingExtensionsConfig(polynomial_degree=2), verbose=0)
    messages = [r.getMessage() for r in caplog.records if "falling back to bare" in r.getMessage()]
    assert len(messages) == 1
    assert messages[0].isascii()
    assert "unexpected keyword argument 'split_blocks'" in messages[0]
    assert "falling back to bare .to_pandas() -- wide-frame conversion will be ~30x slower" in messages[0]
    assert messages[0].endswith(f"polars version={pl.__version__}")


def test_no_non_ascii_in_print_or_logger_arguments() -> None:
    """Forensic-style: enumerate every U+2013 / U+2014 / U+2192 / U+00B1
    character in src/mlframe/ and assert none lands inside a print() or
    logger.* argument context (heuristic: check the enclosing line)."""
    import re

    forbidden_in_io = ["\u2014", "\u2013", "\u2192", "\u2190", "\u2713", "\u2717", "\u00b1", "\u2265", "\u2264", "\u2026"]
    io_pattern = re.compile(r"\b(?:print|logger\.\w+|logging\.\w+|sys\.stdout\.write|sys\.stderr\.write|tqdmu)\s*\(")
    canary = 'logger.warning("a \u2014 b")'
    assert io_pattern.search(canary) and any(ch in canary for ch in forbidden_in_io)

    py_files = sorted(MLFRAME_ROOT.rglob("*.py"))
    assert py_files
    leaks: list = []
    for py in py_files:
        text = py.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            if not any(ch in line for ch in forbidden_in_io):
                continue
            if io_pattern.search(line):
                leaks.append(f"{py.relative_to(MLFRAME_ROOT)}:{line_no}: {line.strip()[:120]}")

    assert not leaks, (
        "Wave 71: non-ASCII char(s) found inside print()/logger argument(s); "
        "Windows cp866 console will crash on UnicodeEncodeError. Sites:\n  " + "\n  ".join(leaks)
    )
