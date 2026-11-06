"""With no environment override the decompression cap is half of physical RAM, never above 64 GiB.

The 64 GiB default was more than most hosts have, so a bomb under that size still exhausted memory before the cap refused it.
"""

from __future__ import annotations

import types

import psutil

from mlframe.training import _bounded_zstd as bz


def test_default_cap_is_half_of_ram_on_a_small_host(monkeypatch):
    """On a host with 2 GiB of RAM the default cap is 1 GiB."""
    monkeypatch.delenv(bz.MAX_DECOMPRESSED_ENV, raising=False)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(total=2 * 1024**3))
    assert bz.max_decompressed_bytes() == 1024**3


def test_default_cap_never_exceeds_64_gib(monkeypatch):
    """On a very large host the default stays at 64 GiB."""
    monkeypatch.delenv(bz.MAX_DECOMPRESSED_ENV, raising=False)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: types.SimpleNamespace(total=512 * 1024**3))
    assert bz.max_decompressed_bytes() == 64 * 1024**3
