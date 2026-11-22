"""Tests for the incremental pairs cache.

We build a small synthetic FrameIndex + records (no real video needed) so the
aggregator runs on controlled frame hashes, then exercise add / batch-add /
remove and check the incremental result against a full scan. Also guards the
cache file against being deleted by cache-build's record cleanup.
"""
from __future__ import annotations

import numpy as np

from vidsim import pairs_cache
from vidsim.aggregate import PairResult, Segment
from vidsim.cache import VideoRecord
from vidsim.config import DEFAULT_MATCH_PARAMS, HASH_BYTES
from vidsim.hashing import FrameFeature
from vidsim.index import FrameIndex

MP = DEFAULT_MATCH_PARAMS


# ---- round-trip ----------------------------------------------------------

def test_segment_round_trip() -> None:
    s = Segment(q_start=1.0, q_end=9.0, t_start=49.0, t_end=57.0,
                offset=48.0, frames=9)
    assert Segment.from_dict(s.to_dict()) == s


def test_pairresult_round_trip() -> None:
    r = PairResult(
        query_id="b", target_id="a", coverage=0.5,
        query_duration=10.0, target_duration=120.0,
        segments=[Segment(0.0, 3.0, 48.0, 51.0, 48.0, 4)],
    )
    assert PairResult.from_dict(r.to_dict()) == r


# ---- synthetic index helpers --------------------------------------------

def _hash_for(seed: int) -> np.ndarray:
    """Deterministic distinct 32-byte hash for a given integer."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=HASH_BYTES, dtype=np.uint8)


def _record(fid: str, frame_hashes: list[np.ndarray], duration: float) -> VideoRecord:
    feats = [
        FrameFeature(frame_number=i, timestamp=float(i), quality=100,
                     hex=h.tobytes().hex())
        for i, h in enumerate(frame_hashes)
    ]
    return VideoRecord(
        file_id=fid, paths=[f"/x/{fid}.mp4"], duration=duration,
        frame_count=len(feats), params_fingerprint="t", vpdq_version="t",
        schema_version=1, features=feats,
        stat_by_path={f"/x/{fid}.mp4": [1000.0, 1.0]},
    )


def _make_collection() -> dict[str, VideoRecord]:
    """Three videos: A (long) contains a chunk reused by B and by C; B and C
    share nothing with each other beyond A's chunk."""
    shared = [_hash_for(1000 + i) for i in range(8)]   # the reused chunk
    a_only = [_hash_for(i) for i in range(20)]
    # A: 0..19 unique, then the shared chunk at 20..27
    A = _record("A", a_only + shared, duration=28.0)
    # B: the shared chunk (same frames) + some unique tail
    B = _record("B", shared + [_hash_for(500 + i) for i in range(4)], duration=12.0)
    # C: unique head + the shared chunk
    C = _record("C", [_hash_for(800 + i) for i in range(3)] + shared, duration=11.0)
    return {"A": A, "B": B, "C": C}


def _index(records: dict[str, VideoRecord]) -> FrameIndex:
    return FrameIndex.build(records, kind="flat")


def _keyset(results: list[PairResult]) -> dict[tuple[str, str], float]:
    return {(r.query_id, r.target_id): round(r.coverage, 6) for r in results}


# ---- equivalence guard ---------------------------------------------------

def test_incremental_add_one_at_a_time_equals_full() -> None:
    recs = _make_collection()
    fids = list(recs)
    idx = _index(recs)   # index over ALL frames (membership is via records arg)
    full = _keyset(pairs_cache.full_scan(recs, idx, MP))

    cached_fids: set[str] = set()
    cached: list[PairResult] = []
    for i, _ in enumerate(fids):
        cur = {k: recs[k] for k in fids[: i + 1]}
        cached, _a, _r = pairs_cache.incremental_update(
            cached_fids, cached, cur, idx, MP)
        cached_fids = set(cur)
    assert _keyset(cached) == full


def test_incremental_batch_add_equals_full() -> None:
    recs = _make_collection()
    fids = list(recs)
    idx = _index(recs)
    full = _keyset(pairs_cache.full_scan(recs, idx, MP))

    start = {k: recs[k] for k in fids[:1]}
    cached = pairs_cache.full_scan(start, idx, MP)
    cached, added, removed = pairs_cache.incremental_update(
        set(start), cached, recs, idx, MP)   # batch-add the rest at once
    assert added == set(fids[1:]) and removed == set()
    assert _keyset(cached) == full


def test_incremental_remove_equals_full() -> None:
    recs = _make_collection()
    fids = list(recs)
    idx_all = _index(recs)
    cached = pairs_cache.full_scan(recs, idx_all, MP)

    keep = {k: recs[k] for k in fids[:-1]}   # drop the last video
    # In the real flow, removing a video rebuilds the index WITHOUT it before
    # `pairs` runs, so the index passed to incremental_update matches `keep`.
    idx_keep = _index(keep)
    cached, added, removed = pairs_cache.incremental_update(
        set(fids), cached, keep, idx_keep, MP)
    assert removed == {fids[-1]} and added == set()
    full_after = _keyset(pairs_cache.full_scan(keep, idx_keep, MP))
    assert _keyset(cached) == full_after
    # no surviving pair references the removed video
    assert all(fids[-1] not in (r.query_id, r.target_id) for r in cached)


# ---- cache file load/save ------------------------------------------------

def test_save_load_round_trip(tmp_path) -> None:
    recs = _make_collection()
    idx = _index(recs)
    results = pairs_cache.full_scan(recs, idx, MP)
    # tmp_path acts as the collection root; save writes <root>/.vidsim/pairs.json
    pairs_cache.save(tmp_path, set(recs), results)
    loaded = pairs_cache.load(tmp_path)
    assert loaded is not None
    fids, loaded_results = loaded
    assert fids == set(recs)
    assert _keyset(loaded_results) == _keyset(results)


def test_load_missing_returns_none(tmp_path) -> None:
    assert pairs_cache.load(tmp_path) is None


def test_load_corrupt_returns_none(tmp_path) -> None:
    from vidsim.config import pairs_cache_path
    p = pairs_cache_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{ not valid json")
    assert pairs_cache.load(tmp_path) is None


# ---- regression: cache-build must not delete the pairs cache --------------

def test_reserved_json_not_treated_as_record(tmp_path) -> None:
    """collection.json / pairs.json / pairs_ignore.json in a .vidsim/ dir must
    NOT be picked up by _scan_record_files as per-video records -- otherwise
    sync's cleanup deletes them as 'stale records', which silently wiped the
    pairs cache on every cache-build (so `pairs` always fell back to full scan)."""
    from vidsim.cache import _scan_record_files
    from vidsim.config import VIDSIM_DIRNAME

    vd = tmp_path / VIDSIM_DIRNAME
    vd.mkdir(parents=True)
    # a genuine per-video record (64-hex-ish stem) + the reserved root files
    (vd / "abc123.json").write_text("{}")
    (vd / "collection.json").write_text("{}")
    (vd / "pairs.json").write_text("{}")
    (vd / "pairs_ignore.json").write_text("{}")

    found = _scan_record_files(tmp_path)
    assert "abc123" in found                 # real record IS scanned
    assert "pairs" not in found              # reserved files are NOT
    assert "collection" not in found
    assert "pairs_ignore" not in found
