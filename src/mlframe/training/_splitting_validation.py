"""Stages of ``make_train_test_split``."""

from __future__ import annotations

# --- imports (managed) ---
# --- end imports ---


def _validate_split_arguments(test_size, val_size, trainset_aging_limit, val_placement):
    """Stage of ``make_train_test_split``."""
    if not (0.0 <= test_size <= 1.0):
        raise ValueError(f"test_size must be in [0, 1], got {test_size}")
    if not (0.0 <= val_size <= 1.0):
        raise ValueError(f"val_size must be in [0, 1], got {val_size}")
    # trainset_aging_limit: None = no-aging; otherwise must be strictly in (0, 1).
    # 0 is rejected explicitly because it would mean "keep zero of training
    # data", which is unusable and used to silently fall through the
    # ``if trainset_aging_limit:`` falsy short-circuit.
    if trainset_aging_limit is not None and not (0 < trainset_aging_limit < 1.0):
        raise ValueError(f"trainset_aging_limit must be in (0, 1), got {trainset_aging_limit}")
    if val_placement not in ("forward", "backward"):
        raise ValueError(f"val_placement must be 'forward' or 'backward', got {val_placement!r}")


def _assert_nonempty_splits(test_size, train_idx, df, wholeday_splitting, timestamps, val_size, trainset_aging_limit, val_idx, val_sequential_fraction, test_idx, test_sequential_fraction):
    """Stage of ``_make_train_test_sp_wholeday_splitting_timestamps_none``."""
    # Empty-split guard: the user requested a non-zero val/test fraction but the
    # split wound up with 0 rows. The splitter is the SOURCE of this defect --
    # the sequential / aging path floors ``int(pool * frac)`` to 0 at low n, and
    # the wholeday path can collapse to a single date -- so raise an actionable
    # error here naming the offending config rather than handing a silent 0-row
    # split downstream (where it surfaces far from the cause as e.g. CatBoost
    # "Input data must have at least one feature" or an empty-eval-set crash).
    # The fallback paths above already redirect the recoverable wholeday-collapse
    # case to row-based; a still-empty split here is a genuine misconfiguration.
    # test_size=1.0 (val_size=0.0) is a legitimate, explicitly-requested "evaluate-only, no
    # training" configuration (see test_splitting_edges.py's
    # NaT-strftime-on-empty-train sensor) and must NOT raise here, mirroring how the val/test
    # guards below only fire when a POSITIVE requested size produced nothing. test_size < 1.0
    # means the user did not claim the whole pool via test alone, so an empty train here is a
    # genuine floor-to-zero / aging-limit surprise, not the user's explicit intent.
    if test_size < 1.0 and len(train_idx) == 0:
        raise ValueError(
            f"Split produced 0 train rows on n={len(df)} rows (mode={'wholeday' if wholeday_splitting and timestamps is not None else 'sequential/row'}, "
            f"val_size={val_size}, test_size={test_size}, trainset_aging_limit={trainset_aging_limit}). "
            f"The same floor-to-zero / wholeday-collapse mechanisms that can empty val_idx/test_idx "
            f"(guarded above) can also empty train_idx, most commonly a large trainset_aging_limit on "
            f"a small n. Reduce val_size/test_size/trainset_aging_limit, or increase n."
        )
    if val_size > 0 and len(val_idx) == 0:
        raise ValueError(
            f"Split produced 0 validation rows from val_size={val_size} on "
            f"n={len(df)} rows (mode={'wholeday' if wholeday_splitting and timestamps is not None else 'sequential/row'}, "
            f"val_sequential_fraction={val_sequential_fraction}, "
            f"trainset_aging_limit={trainset_aging_limit}). int(pool*val_size) "
            f"floored to 0, or wholeday_splitting collapsed to a single date. "
            f"Increase val_size, increase n, or disable wholeday_splitting."
        )
    if test_size > 0 and len(test_idx) == 0:
        raise ValueError(
            f"Split produced 0 test rows from test_size={test_size} on "
            f"n={len(df)} rows (mode={'wholeday' if wholeday_splitting and timestamps is not None else 'sequential/row'}, "
            f"test_sequential_fraction={test_sequential_fraction}). "
            f"int(pool*test_size) floored to 0, or wholeday_splitting collapsed "
            f"to a single date. Increase test_size, increase n, or disable "
            f"wholeday_splitting."
        )
