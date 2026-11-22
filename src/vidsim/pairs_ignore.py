"""Persistent 'ignore' list for pairs: video pairs the user has reviewed and
wants hidden from future `pairs` output.

Keyed by an UNORDERED pair of content ids -- ignoring A~B hides both the
"A drawn from B" and "B drawn from A" rows. Stored as
`<root>/.vidsim/pairs_ignore.json`: {"ignored": [[id_a, id_b], ...]} with each
pair canonicalised (sorted) so (A,B) and (B,A) collapse to one entry. Survives
renames/moves (content ids are stable). Entries whose videos are gone get pruned.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import vidsim_dir


def _ignore_path(root: str | Path) -> Path:
    return vidsim_dir(root) / "pairs_ignore.json"


def _key(a: str, b: str) -> tuple[str, str]:
    """Canonical unordered pair key, so (A,B) and (B,A) are the same entry."""
    return (a, b) if a <= b else (b, a)


def load(root: str | Path) -> set[tuple[str, str]]:
    """Set of canonical (id, id) pairs to ignore. Empty on missing/corrupt."""
    p = _ignore_path(root)
    if not p.exists():
        return set()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {_key(str(x[0]), str(x[1])) for x in data["ignored"] if len(x) == 2}
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError, IndexError):
        return set()


def save(root: str | Path, pairs: set[tuple[str, str]]) -> None:
    """Atomically write the ignore set (`.tmp` -> replace)."""
    p = _ignore_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {"ignored": [list(k) for k in sorted(pairs)]}
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(p)


def is_ignored(ignored: set[tuple[str, str]], a: str, b: str) -> bool:
    return _key(a, b) in ignored


def add(ignored: set[tuple[str, str]], a: str, b: str) -> None:
    ignored.add(_key(a, b))


def remove(ignored: set[tuple[str, str]], a: str, b: str) -> None:
    ignored.discard(_key(a, b))


def prune(ignored: set[tuple[str, str]], present_ids: set[str]) -> set[tuple[str, str]]:
    """Drop entries whose either video is no longer present in the collection."""
    return {k for k in ignored if k[0] in present_ids and k[1] in present_ids}
