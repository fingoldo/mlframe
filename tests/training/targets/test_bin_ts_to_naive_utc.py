"""Bin edges given as stdlib datetimes become naive UTC whatever the host zone is (tracker 2026-11-02 wave 9, TIME9-1)."""

from datetime import datetime, timedelta, timezone

from mlframe.training.targets._target_temporal_types import bin_ts_to_naive_utc


def test_utc_datetime_keeps_its_wall_time():
    """A UTC midnight stays midnight; astimezone(None) shifted it by the host's offset on any non-UTC machine."""
    assert bin_ts_to_naive_utc(datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)) == datetime(2024, 1, 1, 0, 0)


def test_offset_datetime_is_converted_to_utc():
    """03:00 at UTC+3 is 00:00 UTC, on every host."""
    plus3 = timezone(timedelta(hours=3))
    assert bin_ts_to_naive_utc(datetime(2024, 1, 1, 3, 0, tzinfo=plus3)) == datetime(2024, 1, 1, 0, 0)
