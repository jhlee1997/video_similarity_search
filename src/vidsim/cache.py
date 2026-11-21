"""Per-video hash cache: one JSON record per video, keyed by content file_id.

A record stores the frame features plus enough provenance (hash params, vpdq
version, schema version) to detect when it is stale and must be recomputed.

Keeping one file per video (rather than one big DB) makes incremental sync
cheap and crash-safe: adding or dropping a video touches exactly one file.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from pathlib import Path

from tqdm import tqdm

from .config import (
    SCHEMA_VERSION,
    VIDEO_EXTS,
    VIDSIM_DIRNAME,
    HashParams,
    manifest_path,
    record_path,
    vidsim_dir,
)
from .fileid import file_id
from .hashing import FrameFeature, hash_video

VPDQ_VERSION = importlib.metadata.version("vpdq")


@dataclass
class VideoRecord:
    """Cached hashes + provenance for one video (identified by content)."""

    file_id: str
    paths: list[str]  # last known path(s) with this content
    duration: float
    frame_count: int
    params_fingerprint: str
    vpdq_version: str
    schema_version: int
    features: list[FrameFeature] = field(default_factory=list)
    # Fast-path: resolved path -> [size, mtime]. Lets a re-run skip the content
    # read when a known path's stat is unchanged. Empty for pre-fast-path records
    # (they simply miss the fast-path and fall back to a content hash).
    stat_by_path: dict[str, list[float]] = field(default_factory=dict)

    # ---- serialisation ----

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "paths": self.paths,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "params_fingerprint": self.params_fingerprint,
            "vpdq_version": self.vpdq_version,
            "schema_version": self.schema_version,
            "features": [f.to_dict() for f in self.features],
            "stat_by_path": self.stat_by_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VideoRecord":
        return cls(
            file_id=d["file_id"],
            paths=list(d.get("paths", [])),
            duration=float(d.get("duration", 0.0)),
            frame_count=int(d.get("frame_count", 0)),
            params_fingerprint=d.get("params_fingerprint", ""),
            vpdq_version=d.get("vpdq_version", ""),
            schema_version=int(d.get("schema_version", 0)),
            features=[FrameFeature.from_dict(x) for x in d.get("features", [])],
            stat_by_path={k: list(v) for k, v in d.get("stat_by_path", {}).items()},
        )

    def is_stale(self, params: HashParams) -> bool:
        return (
            self.schema_version != SCHEMA_VERSION
            or self.vpdq_version != VPDQ_VERSION
            or self.params_fingerprint != params.fingerprint()
        )


def load_record(folder: str | Path, fid: str) -> VideoRecord | None:
    """Load the record for *fid* from *folder*'s `.vidsim/`, if present."""
    p = record_path(folder, fid)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return VideoRecord.from_dict(json.load(f))
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def save_record(rec: VideoRecord, folder: str | Path) -> None:
    """Write *rec* into *folder*'s `.vidsim/` (atomic)."""
    vidsim_dir(folder).mkdir(parents=True, exist_ok=True)
    p = record_path(folder, rec.file_id)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(rec.to_dict(), f)
    tmp.replace(p)  # atomic on POSIX


def _scan_record_files(root: Path) -> dict[str, list[Path]]:
    """Map file_id -> all `.vidsim/<fid>.json` paths found under *root*.

    A given content may (through partial earlier runs or migration) have a
    record file in more than one folder; we collect them all so the caller can
    keep one and delete the rest. Only `<fid>.json` files are considered: the
    `*.json` glob already excludes `index.faiss`/`rowmap.npz`, and we skip
    `collection.json` explicitly.
    """
    found: dict[str, list[Path]] = {}
    for vd in root.rglob(VIDSIM_DIRNAME):
        if not vd.is_dir():
            continue
        for p in vd.glob("*.json"):
            if p.name == "collection.json":
                continue
            fid = p.stem
            found.setdefault(fid, []).append(p)
    return found


def load_all_records(root: str | Path) -> dict[str, tuple[VideoRecord, Path]]:
    """Load every record under *root*, keyed by file_id -> (record, folder).

    `folder` is the directory whose `.vidsim/` held the record. If the same
    file_id has record files in several folders, one is chosen deterministically
    (shallowest path, then lexical) and the duplicates are left for `sync` to
    clean up. Unreadable records are skipped.
    """
    out: dict[str, tuple[VideoRecord, Path]] = {}
    root = Path(root)
    for fid, paths in _scan_record_files(root).items():
        for p in sorted(paths, key=lambda q: (len(q.parts), str(q))):
            try:
                with p.open("r", encoding="utf-8") as f:
                    rec = VideoRecord.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
            out[rec.file_id] = (rec, p.parent.parent)  # folder = parent of .vidsim/
            break
    return out


def load_records(root: str | Path) -> dict[str, VideoRecord]:
    """Convenience: records under *root* keyed by file_id (folder stripped)."""
    return {fid: rec for fid, (rec, _folder) in load_all_records(root).items()}


def iter_video_files(root: str | Path):
    """Yield video file paths under *root*, recursively, skipping `.vidsim/`."""
    root = Path(root)
    for p in root.rglob("*"):
        if VIDSIM_DIRNAME in p.parts:
            continue  # never descend into cache dirs
        if p.is_file() and p.suffix.lower().lstrip(".") in VIDEO_EXTS:
            yield p


def find_collection_root(start: str | Path) -> Path | None:
    """Nearest ancestor of *start* (incl. itself) holding a collection index.

    Returns None if none is found. Used by read commands (find/pairs/index) so
    pointing them at a subfolder still locates the real collection.
    """
    start = Path(start).resolve()
    for folder in (start, *start.parents):
        if (vidsim_dir(folder) / "index.faiss").exists():
            return folder
    return None


def resolve_collection_root(start: str | Path) -> Path:
    """Resolve the collection root for a `cache-build` on *start*.

    If an ancestor (including *start*) already holds a collection index, use that
    ancestor so a sub-run attaches to the existing collection instead of
    fragmenting it. Otherwise *start* itself is the root.
    """
    return find_collection_root(start) or Path(start).resolve()


def _stat_pair(path: Path) -> list[float]:
    """[size, mtime] for *path*, used by the fast-path."""
    st = path.stat()
    return [float(st.st_size), float(st.st_mtime)]


def assemble_record(path: Path, fid: str, features: list[FrameFeature],
                    params: HashParams,
                    stat_pair: list[float] | None = None) -> VideoRecord:
    """Build a VideoRecord from already-computed hashes (cheap; no decoding)."""
    duration = max((f.timestamp for f in features), default=0.0)
    rp = str(path.resolve())
    return VideoRecord(
        file_id=fid,
        paths=[rp],
        duration=duration,
        frame_count=len(features),
        params_fingerprint=params.fingerprint(),
        vpdq_version=VPDQ_VERSION,
        schema_version=SCHEMA_VERSION,
        features=features,
        stat_by_path={rp: stat_pair if stat_pair is not None else _stat_pair(path)},
    )


def build_record(path: Path, fid: str, params: HashParams) -> VideoRecord:
    """Hash a video and assemble its cache record (single-process path)."""
    features = hash_video(path, params)
    return assemble_record(path, fid, features, params)


def _hash_job(args: tuple[str, str, HashParams]) -> tuple[str, Path, list[FrameFeature]]:
    """Process-pool worker: hash one video. Returns (fid, path, features).

    Pins thread_count=1 so N worker processes don't each spawn N ffmpeg threads.
    Must be module-level so ProcessPoolExecutor can pickle it.
    """
    path_str, fid, params = args
    features = hash_video(path_str, params, thread_count=1)
    return fid, Path(path_str), features


@dataclass
class SyncStats:
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    removed: int = 0
    failed: int = 0
    duplicate_copies: int = 0  # byte-identical files sharing one record


def _write_manifest(root: Path, params: HashParams,
                    location: dict[str, Path]) -> None:
    """Write collection.json: provenance + file_id -> folder-holding-its-record."""
    rootr = root.resolve()
    folder_map: dict[str, str] = {}
    for fid, folder in location.items():
        try:
            rel = str(folder.resolve().relative_to(rootr))
        except ValueError:
            rel = str(folder.resolve())
        folder_map[fid] = rel or "."
    data = {
        "schema_version": SCHEMA_VERSION,
        "vpdq_version": VPDQ_VERSION,
        "params_fingerprint": params.fingerprint(),
        "records": folder_map,
    }
    vidsim_dir(root).mkdir(parents=True, exist_ok=True)
    p = manifest_path(root)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(p)


def _prune_empty_vidsim(root: Path) -> None:
    """Remove `.vidsim/` dirs holding nothing but leftover `.tmp` garbage.

    A crashed save can leave `<fid>.json.tmp`; we delete those first so the dir
    can be recognised as empty and removed. The collection root's own `.vidsim/`
    (which holds the index) is never removed.
    """
    rootr = root.resolve()
    for vd in root.rglob(VIDSIM_DIRNAME):
        if not vd.is_dir():
            continue
        for tmp in vd.glob("*.tmp"):
            tmp.unlink(missing_ok=True)
        if vd.parent.resolve() == rootr:
            continue
        if not any(vd.iterdir()):
            vd.rmdir()


def default_workers() -> int:
    """Leave ~a quarter of cores free so the workstation stays usable during a
    build, but never drop below 2 when more than one core exists (so the pool is
    actually used on small machines; `-3` would disable parallelism on a 4-core
    box)."""
    cpu = os.cpu_count() or 4
    if cpu <= 1:
        return 1
    return max(2, cpu - max(1, cpu // 4))


def sync(root: str | Path, params: HashParams,
         workers: int | None = None) -> SyncStats:
    """Reconcile the per-folder cache with the videos currently under *root*.

    - new content          -> hash, write record in the video's own folder
    - stale record         -> recompute, written to the video's current folder
    - unchanged            -> keep hashes; record follows the video if it moved
    - byte-identical copy  -> share one record (first folder seen); add its path
    - content gone          -> drop its record

    Identifying a file uses a fast-path: if a known path's size+mtime match the
    cached record, its file_id is reused without re-reading the file. Hashing of
    new/stale videos is fanned out over a process pool (vpdq holds the GIL, so
    threads don't help); all cache I/O stays in this process. Writes
    collection.json and prunes emptied `.vidsim/` dirs. Does NOT build the index.
    """
    root = Path(root)
    if workers is None:
        workers = default_workers()
    stats = SyncStats()
    existing = load_all_records(root)             # fid -> (rec, old home folder)
    all_record_files = _scan_record_files(root)   # fid -> every record file path

    # Fast-path lookup: resolved path -> (fid, [size, mtime]) from prior records.
    stat_lookup: dict[str, tuple[str, list[float]]] = {}
    for fid, (rec, _folder) in existing.items():
        for p, sp in rec.stat_by_path.items():
            stat_lookup[p] = (fid, sp)

    location: dict[str, Path] = {}   # fid -> folder its record now lives in

    # Per-fid aggregation decided in Phase 1 and consumed in Phase 2 / finalise.
    # Each fid maps to a Plan: where its record lives, every current path with its
    # stat, what to do (reuse cached hashes, or hash), and the old home to clean.
    @dataclass
    class _Plan:
        folder: Path                       # home folder for the record (first sighting)
        kind: str                          # "cached" | "add" | "update"
        paths: list[str] = field(default_factory=list)        # all current paths
        stats_by_path: dict[str, list[float]] = field(default_factory=dict)
        cached_rec: VideoRecord | None = None   # reused record for "cached"
        hash_path: Path | None = None           # which file to hash for add/update
        old_home: Path | None = None            # prior record folder, if it moved

    plans: dict[str, _Plan] = {}

    # ---- Phase 1: classify every file (cheap; no frame decoding) ----
    files = list(iter_video_files(root))
    for path in tqdm(files, desc="scanning", unit="vid"):
        folder = path.parent
        rp = str(path.resolve())
        try:
            sp = _stat_pair(path)
        except OSError:
            stats.failed += 1
            continue

        # Fast-path: known path with unchanged size+mtime -> reuse cached fid,
        # skip the content read entirely.
        hit = stat_lookup.get(rp)
        if hit is not None and hit[1] == sp:
            fid = hit[0]
        else:
            try:
                fid = file_id(path)
            except OSError:
                stats.failed += 1
                continue

        plan = plans.get(fid)
        if plan is not None:
            # A byte-identical copy already seen this run: just record the extra
            # path on the existing plan (first sighting owns the home folder).
            if rp not in plan.paths:
                plan.paths.append(rp)
            plan.stats_by_path[rp] = sp
            stats.duplicate_copies += 1
            continue

        prior = existing.get(fid)
        if prior is None:
            kind = "add"
            cached = None
            old_home = None
        else:
            old_rec, old_home_folder = prior
            old_home = old_home_folder
            if old_rec.is_stale(params):
                kind, cached = "update", None
            else:
                kind, cached = "cached", old_rec
        plans[fid] = _Plan(
            folder=folder, kind=kind, paths=[rp], stats_by_path={rp: sp},
            cached_rec=cached, hash_path=(None if kind == "cached" else path),
            old_home=old_home,
        )

    # ---- Phase 2: hash new/stale videos in parallel, then write every record ----
    to_hash = [(p.hash_path, fid) for fid, p in plans.items() if p.kind != "cached"]
    features_by_fid: dict[str, list[FrameFeature]] = {}
    failed_fids: set[str] = set()

    if to_hash:
        bar = tqdm(total=len(to_hash), desc="hashing", unit="vid")
        if workers == 1:
            for path, fid in to_hash:
                bar.set_postfix_str(path.name, refresh=True)
                try:
                    features_by_fid[fid] = hash_video(path, params, thread_count=1)
                except Exception:  # noqa: BLE001 - bad file, skip it
                    failed_fids.add(fid)
                bar.update(1)
        else:
            # With N workers, N files hash concurrently, so "the file being
            # hashed" isn't single-valued; the bar's count IS the progress signal
            # and the postfix shows the most recently completed file as context.
            jobs = [(str(p), fid, params) for p, fid in to_hash]
            try:
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(_hash_job, j): j[1] for j in jobs}
                    for fut in as_completed(futures):
                        job_fid = futures[fut]
                        try:
                            fid, path, features = fut.result()
                            features_by_fid[fid] = features
                            bar.set_postfix_str(path.name, refresh=False)
                        except BrokenProcessPool:
                            # A worker died natively (segfault/OOM): the pool is
                            # dead and EVERY remaining future will raise this too.
                            # Don't mark this fid failed here -- propagate to the
                            # fallback, which re-hashes all un-done jobs. (If we
                            # swallowed it as a per-file failure, one native crash
                            # would mark the whole remaining batch failed.)
                            raise
                        except Exception:  # noqa: BLE001 - ordinary bad file, skip
                            failed_fids.add(job_fid)
                        bar.update(1)
            except KeyboardInterrupt:
                pool.shutdown(wait=False, cancel_futures=True)
                bar.close()
                raise
            except BrokenProcessPool:
                # Finish the jobs that never produced a result, single-process, so
                # one bad file can't fail the whole batch. The file that actually
                # crashed the worker will raise an ordinary exception on retry and
                # be marked failed; the rest succeed.
                done = set(features_by_fid) | failed_fids
                leftover = [(Path(j[0]), j[1]) for j in jobs if j[1] not in done]
                for path, fid in leftover:
                    bar.set_postfix_str(path.name, refresh=True)
                    try:
                        features_by_fid[fid] = hash_video(path, params, thread_count=1)
                    except Exception:  # noqa: BLE001 - this one is the bad apple
                        failed_fids.add(fid)
                    bar.update(1)
        bar.close()

    # ---- Finalise: write one record per fid (main process only) ----
    for fid, plan in plans.items():
        if fid in failed_fids:
            stats.failed += 1
            continue
        try:
            if plan.kind == "cached":
                # Copy, don't mutate the object held by `existing` (aliasing).
                rec = replace(plan.cached_rec,
                              paths=list(plan.paths),
                              stat_by_path=dict(plan.stats_by_path))
                stats.unchanged += 1
            else:
                feats = features_by_fid.get(fid)
                if feats is None:            # shouldn't happen, but never crash
                    stats.failed += 1
                    continue
                rec = assemble_record(plan.hash_path, fid, feats, params,
                                      plan.stats_by_path[str(plan.hash_path.resolve())])
                rec.paths = list(plan.paths)
                rec.stat_by_path = dict(plan.stats_by_path)
                stats.added += 1 if plan.kind == "add" else 0
                stats.updated += 1 if plan.kind == "update" else 0

            save_record(rec, plan.folder)
            location[fid] = plan.folder
            # If the record used to live elsewhere, drop the old file.
            if plan.old_home is not None and plan.old_home.resolve() != plan.folder.resolve():
                record_path(plan.old_home, fid).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            stats.failed += 1

    # Remove record files for content no longer present, and collapse any
    # duplicate record files for seen content down to the chosen home folder.
    # A fid is "present this run" iff it has a plan; location[fid] is set only
    # when its record was successfully (re)written in finalise.
    for fid, paths in all_record_files.items():
        home = location.get(fid)
        if fid not in plans:
            for p in paths:
                p.unlink(missing_ok=True)
            stats.removed += 1
        elif home is not None:
            keep = record_path(home, fid).resolve()
            for p in paths:
                if p.resolve() != keep:
                    p.unlink(missing_ok=True)

    _write_manifest(root, params, location)
    _prune_empty_vidsim(root)
    return stats
