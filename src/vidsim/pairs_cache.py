"""Incremental cache of `pairs` results for a collection.

The all-pairs scan is expensive, but a collection changes only by adding or
removing whole videos (existing videos are never edited). So we cache the full
pair list and, on a rerun, update only what changed:

  * videos ADDED since last time  -> compute their pairs (both directions) via
    aggregate.compare_bidirectional; new-vs-new falls out because the index
    already holds every current frame. Deduped by (query_id, target_id).
  * videos REMOVED                -> drop every cached pair touching them.
  * everything else               -> kept from cache, not recomputed.

The cache stores EVERY pair the aggregator emits (no coverage floor), so the
`--min-coverage` display threshold never forces a recompute -- it just filters
what's shown. There is intentionally NO code/param versioning: if the matching
rules change (`--frame-radius`/`--min-shared-frames`) or the aggregator code
changes, delete `.vidsim/pairs.json` and rerun (documented in the README).

Cache file (collection root `.vidsim/pairs.json`):
    {"file_ids": [sorted fids covered], "results": [PairResult.to_dict(), ...]}
"""
from __future__ import annotations

import json
from pathlib import Path

from .aggregate import PairResult, compare_bidirectional
from .cache import VideoRecord
from .config import pairs_cache_path
from .index import FrameIndex
from .config import MatchParams


def load(root: str | Path) -> tuple[set[str], list[PairResult]] | None:
    """Load (covered file_ids, results) from the cache, or None if absent/corrupt."""
    p = pairs_cache_path(root)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        fids = set(str(x) for x in data["file_ids"])
        results = [PairResult.from_dict(d) for d in data["results"]]
        return fids, results
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError):
        return None


def save(root: str | Path, file_ids: set[str], results: list[PairResult]) -> None:
    """Atomically write the cache (`.tmp` -> replace, like the other writers)."""
    p = pairs_cache_path(root)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "file_ids": sorted(file_ids),
        "results": [r.to_dict() for r in results],
    }
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(p)


def clear(root: str | Path) -> None:
    """Remove the cache file (e.g. on --refresh or when the collection is gone)."""
    pairs_cache_path(root).unlink(missing_ok=True)


def full_scan(records: dict[str, VideoRecord], idx: FrameIndex,
              mp: MatchParams) -> list[PairResult]:
    """Compute every pair from scratch (used for the first run and --refresh).

    Uses the plain directional `compare` per video -- equivalent to the old
    pairs loop, but stores ALL pairs (no coverage filter) so the cache is a
    superset the display threshold can filter freely.
    """
    from .aggregate import compare

    out: list[PairResult] = []
    for rec in records.values():
        out.extend(compare(rec, idx, mp, records))
    return out


def incremental_update(
    cached_fids: set[str],
    cached_results: list[PairResult],
    records: dict[str, VideoRecord],
    idx: FrameIndex,
    mp: MatchParams,
) -> tuple[list[PairResult], set[str], set[str]]:
    """Return (merged_results, added, removed) updating only what changed.

    - Drop cached pairs touching a removed fid.
    - Add both-direction pairs for each added fid (deduped by (query,target),
      so new-vs-new pairs produced from both endpoints collapse to one).
    - Keep all other cached pairs untouched.
    """
    current = set(records)
    added = current - cached_fids
    removed = cached_fids - current

    # Keep cached pairs whose BOTH endpoints are still present.
    kept = [r for r in cached_results
            if r.query_id in current and r.target_id in current]

    # Compute new pairs for the added videos (both directions). Dedup by the
    # ordered (query, target) key: a pair between two added videos is emitted by
    # each endpoint's compare_bidirectional, so collapse duplicates.
    new_by_key: dict[tuple[str, str], PairResult] = {}
    for fid in added:
        rec = records.get(fid)
        if rec is None:
            continue
        for r in compare_bidirectional(rec, idx, mp, records):
            new_by_key[(r.query_id, r.target_id)] = r

    merged = kept + list(new_by_key.values())
    merged.sort(key=lambda r: r.coverage, reverse=True)
    return merged, added, removed
