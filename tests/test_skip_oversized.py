"""Tests for the always-on oversized-file exclusion in cache.is_oversized.

A file is skipped only when ALL THREE hold: size > 3GB, duration > 20min, and
resolution >= 4K. These tests drive the predicate directly with a monkeypatched
metadata probe, so they need no real multi-GB 4K video.
"""
from __future__ import annotations

from pathlib import Path

from vidsim import cache
from vidsim.config import (
    SKIP_MIN_DURATION_S,
    SKIP_MIN_SIZE_BYTES,
)

BIG = SKIP_MIN_SIZE_BYTES + 1
SMALL = SKIP_MIN_SIZE_BYTES - 1
LONG = SKIP_MIN_DURATION_S + 1
SHORT = SKIP_MIN_DURATION_S - 1
UHD = (3840, 2160)
HD = (1920, 1080)


def _patch_probe(monkeypatch, w: int, h: int, dur: float) -> None:
    monkeypatch.setattr(cache, "_probe_dims_duration", lambda p: (w, h, dur))


def test_skips_4k_long_big(monkeypatch) -> None:
    _patch_probe(monkeypatch, *UHD, LONG)
    assert cache.is_oversized(Path("x.mov"), BIG) is True


def test_small_file_never_probed_never_skipped(monkeypatch) -> None:
    # Size gate fails -> returns False WITHOUT probing (probe would raise).
    def boom(_p):
        raise AssertionError("probe must not run when size gate fails")
    monkeypatch.setattr(cache, "_probe_dims_duration", boom)
    assert cache.is_oversized(Path("x.mov"), SMALL) is False


def test_big_but_short_not_skipped(monkeypatch) -> None:
    _patch_probe(monkeypatch, *UHD, SHORT)
    assert cache.is_oversized(Path("x.mov"), BIG) is False


def test_big_long_but_not_4k_not_skipped(monkeypatch) -> None:
    _patch_probe(monkeypatch, *HD, LONG)
    assert cache.is_oversized(Path("x.mov"), BIG) is False


def test_duration_and_resolution_from_same_probe(monkeypatch) -> None:
    # Single probe supplies both duration and resolution (self-consistent).
    # Long + 4K -> skipped.
    _patch_probe(monkeypatch, *UHD, LONG)
    assert cache.is_oversized(Path("x.mov"), BIG) is True
    # Long but HD -> not skipped (resolution gate fails).
    _patch_probe(monkeypatch, *HD, LONG)
    assert cache.is_oversized(Path("x.mov"), BIG) is False


def test_unreadable_file_not_skipped(monkeypatch) -> None:
    # Probe returns (0,0,0.0) on error -> duration gate fails -> not skipped.
    monkeypatch.setattr(cache, "_probe_dims_duration", lambda p: (0, 0, 0.0))
    assert cache.is_oversized(Path("x.mov"), BIG) is False


def test_height_only_4k_counts(monkeypatch) -> None:
    # Portrait 4K (height >= 2160 via width) still counts as 4K.
    _patch_probe(monkeypatch, 2160, 3840, LONG)
    assert cache.is_oversized(Path("x.mov"), BIG) is True
