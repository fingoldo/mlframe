"""Wave 72 (2026-05-21): polars cast(strict=False) silent OOV-nulling.

Audit found 6 P2 sites where `pl.col(...).cast(enum_dt, strict=False)` silently
nulls cast-failed values without logging the count. mlframe NOT use cast(strict=False)
for numeric narrowing -- only for Enum OOV handling on val/test splits.

Strongest concern: _phase_helpers.py:826 built val Enum from train-ONLY domain,
biasing ES away from val-rare-cat-sensitive splits. Wave 72 fix extends domain
to train+val UNION (test stays unseen), making val strict=True (any cast failure
is now a logic bug, not silent), and adds null-count delta logging on test.

Sites fixed (5 actionable):

  1. training/core/_phase_helpers.py:826 (Enum cast for str cols)
     Domain extended from train-only -> train+val. val cast is now strict=True.
     test cast keeps strict=False + delta-log via _enum_cast(split_name="test").

  2. training/core/_phase_polars_fixes.py:222 (dict-alignment test cast)
     Added null-pre/null-post delta + logger.info on nonzero OOV.

  3. training/strategies.py:693 (XGB category_map cast)
     Wrapped cast with null-count tracking; logs OOV cols when nonzero.

  4. training/strategies.py:889 (HGB low-card category_map cast)
     Tracked via _strict_false_cols list + post-cast delta-log.

  5. training/strategies.py:886 (HGB high-card to_physical UInt32 cast)
     Same _strict_false_cols tracking; the to_physical().cast(UInt32) chain
     keeps the null-bit so post-cast UInt32 has null where Enum had OOV-null.

All fixes follow the uniform "null_count() pre + null_count() post + logger.info
delta when nonzero" pattern so OOV-cast-failure is no longer invisible.
"""
from __future__ import annotations

from pathlib import Path


MLFRAME_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "mlframe"


def _read(rel: str) -> str:
    return (MLFRAME_ROOT / rel).read_text(encoding="utf-8")


def _read_phase_helpers_combined() -> str:
    """Read the union of _phase_helpers.py + _phase_helpers_fit_split.py.
    The enum-cast block moved into the sibling during the 2026-05-21
    monolith split (re-exported via the bottom-of-file import at
    _phase_helpers.py:714). Both locations are valid for the wave-72 sensor."""
    return _read("training/core/_phase_helpers.py") + "\n" + _read(
        "training/core/_phase_helpers_fit_split.py"
    )


def test_phase_helpers_enum_cast_uses_train_plus_val_domain() -> None:
    src = _read_phase_helpers_combined()
    # Domain is now train+val union (sorted by str).
    assert "sorted(set(_u_train) | set(_u_val), key=str)" in src
    # val cast strict=True now (domain includes val so no silent OOV).
    assert "val_df = _enum_cast(val_df, strict=True)" in src
    # test cast keeps strict=False with delta logging.
    assert "_enum_cast(test_df, strict=False, split_name=\"test\")" in src


def test_phase_helpers_enum_cast_logs_oov_delta_on_test() -> None:
    src = _read_phase_helpers_combined()
    # The _enum_cast helper now computes null_pre/null_post deltas.
    assert "_null_pre = {c: int(df[c].null_count()) for c in _affected_cols}" in src
    assert "[enum-cast] %s split: %d col(s) had OOV nulls cast-failed" in src


def test_phase_polars_fixes_test_cast_logs_oov_delta() -> None:
    src = _read("training/core/_phase_polars_fixes.py")
    assert "[cat-alignment] test col=%s: %d row(s) cast-failed to null" in src
    # The null-count baseline is now batched into one collect across all eligible cols (S44)
    # instead of a per-col sync call inside the alignment loop. The diagnostic semantics are
    # preserved: a pre-cast null_count is captured per col and compared against the post-cast
    # null_count to emit the OOV log line.
    assert "_test_nulls_pre" in src
    assert "null_count()" in src


def test_strategies_xgb_cat_cast_logs_oov_delta() -> None:
    """The xgb cat-cast logging moved into the sibling
    _strategies_xgboost.py during the strategies monolith split; check
    both locations."""
    facade = _read("training/strategies.py")
    sibling = _read("training/_strategies_xgboost.py")
    needle = "[xgb cat-cast] %d col(s) had OOV nulls cast-failed"
    assert needle in facade or needle in sibling


def test_strategies_hgb_cat_cast_logs_oov_delta() -> None:
    src = _read("training/strategies.py")
    assert "[hgb cat-cast] %d col(s) had OOV nulls cast-failed" in src
    # _strict_false_cols list-tracking is the wave-72 wiring.
    assert "_strict_false_cols: list[str] = []" in src


# ---------------------------------------------------------------------------
# Behavioural sensors
# ---------------------------------------------------------------------------


def test_train_plus_val_enum_domain_eliminates_val_oov() -> None:
    """val-only categorical values must NOT cast to null when the Enum domain
    is built from train+val (vs train-only)."""
    pl = __import__("polars")

    # train has "a", "b"; val has "a", "c" -- val-only "c" must survive.
    train = pl.DataFrame({"cat": ["a", "b"]})
    val = pl.DataFrame({"cat": ["a", "c"]})

    # The wave-72 fix: domain = sorted(set(train) | set(val), key=str).
    domain = sorted(set(train["cat"].drop_nulls().unique().to_list())
                    | set(val["cat"].drop_nulls().unique().to_list()), key=str)
    assert domain == ["a", "b", "c"]

    # val cast strict=True succeeds (no OOV).
    val_cast = val.with_columns(pl.col("cat").cast(pl.Enum(domain), strict=True))
    assert val_cast["cat"].null_count() == 0
