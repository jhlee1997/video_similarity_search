"""Tests for the pairs ignore list (pairs_ignore.py)."""
from __future__ import annotations

from vidsim import pairs_ignore as pi


def test_canonical_key_both_orders_same() -> None:
    ig: set = set()
    pi.add(ig, "B", "A")
    # ignoring B~A hides both directions
    assert pi.is_ignored(ig, "A", "B")
    assert pi.is_ignored(ig, "B", "A")
    assert len(ig) == 1  # stored once, canonicalised


def test_remove() -> None:
    ig: set = set()
    pi.add(ig, "A", "B")
    pi.remove(ig, "B", "A")   # order-independent
    assert not pi.is_ignored(ig, "A", "B")


def test_prune_drops_entries_with_missing_video() -> None:
    ig = set()
    pi.add(ig, "A", "B")
    pi.add(ig, "A", "C")
    pruned = pi.prune(ig, present_ids={"A", "B"})  # C is gone
    assert pi.is_ignored(pruned, "A", "B")
    assert not pi.is_ignored(pruned, "A", "C")


def test_save_load_round_trip(tmp_path) -> None:
    ig = set()
    pi.add(ig, "A", "B")
    pi.add(ig, "C", "D")
    pi.save(tmp_path, ig)
    loaded = pi.load(tmp_path)
    assert loaded == {("A", "B"), ("C", "D")}


def test_load_missing_is_empty(tmp_path) -> None:
    assert pi.load(tmp_path) == set()


def test_load_corrupt_is_empty(tmp_path) -> None:
    from vidsim.config import vidsim_dir
    p = vidsim_dir(tmp_path) / "pairs_ignore.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ broken")
    assert pi.load(tmp_path) == set()
