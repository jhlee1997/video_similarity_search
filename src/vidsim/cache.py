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
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    as_completed,
    wait,
)
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field, replace
from pathlib import Path

from tqdm import tqdm

from . import ui
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


def _shutdown_now(pool: ProcessPoolExecutor) -> None:
    """Stop a pool promptly, making a best effort to not orphan workers.

    Plain `shutdown(wait=False)` returns immediately and leaves workers (and
    their native ffmpeg work) running -- the orphan bug. We instead SIGKILL the
    live worker processes directly (their native hashing isn't cancellable any
    other way) and join each with a timeout, then shut the executor down. In the
    normal case every worker is reaped before we return; the join timeout is a
    safety valve so a worker wedged in uninterruptible IO can't hang the whole
    program -- in that rare case one process may briefly outlive this call, but
    it is already SIGKILLed and the OS reaps it once the IO completes.
    """
    # Snapshot the worker processes BEFORE shutdown() (which may clear the pool's
    # bookkeeping). `_processes` is a private {pid: Process} dict, possibly None
    # before the pool finishes spinning up -- guard it. SIGKILL each live worker
    # directly (ffmpeg mid-decode won't honour SIGTERM promptly); kill() then
    # join() guarantees the OS has reaped them, so none are orphaned.
    procs = list((getattr(pool, "_processes", None) or {}).values())
    for p in procs:
        try:
            if p.is_alive():
                p.kill()
        except Exception:  # noqa: BLE001
            pass
    for p in procs:
        try:
            p.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
    # Now tell the executor to tear down its queues/threads; workers are gone.
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:  # noqa: BLE001
        pass


def _hash_parallel(to_hash, params, workers, weight_by_fid, bars,
                   write_hashed, done_fids, failed_fids, refresh_manifest) -> None:
    """Hash `to_hash` across a process pool with a BOUNDED-submission loop.

    Keeps exactly `workers` futures in flight, each pinned to a display slot, so
    the parent knows which file occupies which per-worker line WITHOUT any worker
    cooperation (workers run the plain `_hash_job`). Records are written as each
    future completes (resumable). Ctrl+C hard-kills workers; a native worker crash
    (BrokenProcessPool) falls back to finishing the rest single-process.
    """
    queue = list(to_hash)           # (path, fid) still to submit
    qi = 0
    free_slots = list(range(min(workers, len(queue))))
    slot_of: dict = {}              # future -> display slot
    fid_of: dict = {}               # future -> fid
    advanced_fids: set = set()      # fids already counted on the bar (no double-count)
    pending: set = set()

    pool = ProcessPoolExecutor(max_workers=workers)

    def _submit_next() -> None:
        nonlocal qi
        if qi >= len(queue) or not free_slots:
            return
        path, fid = queue[qi]; qi += 1
        slot = free_slots.pop()
        fut = pool.submit(_hash_job, (str(path), fid, params))
        slot_of[fut] = slot; fid_of[fut] = fid
        pending.add(fut)
        bars.set_slot(slot, path.name)

    try:
        for _ in range(min(workers, len(queue))):
            _submit_next()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                fid = fid_of[fut]; slot = slot_of[fut]
                try:
                    r_fid, _path, features = fut.result()
                except BrokenProcessPool:
                    raise
                except Exception:  # noqa: BLE001 - ordinary bad file, skip
                    failed_fids.add(fid)
                else:
                    try:
                        write_hashed(r_fid, features)
                        done_fids.add(r_fid)
                    except Exception:  # noqa: BLE001 - write failed (disk/perms)
                        failed_fids.add(r_fid)
                bars.clear_slot(slot)
                bars.advance(weight_by_fid[fid]); advanced_fids.add(fid)
                free_slots.append(slot)
                _submit_next()      # refill this freed slot
        pool.shutdown(wait=True)
        bars.finish(); bars.close()
    except KeyboardInterrupt:
        _shutdown_now(pool)
        bars.close()
        refresh_manifest()
        raise
    except BrokenProcessPool:
        _shutdown_now(pool)
        # Finish everything not yet done, single-process, so one native crash
        # can't fail the whole batch; the crasher raises on retry -> marked failed.
        remaining = [(p, f) for p, f in to_hash
                     if f not in done_fids and f not in failed_fids]
        for path, fid in remaining:
            bars.set_slot(0, path.name)
            try:
                feats = hash_video(path, params, thread_count=1)
                write_hashed(fid, feats); done_fids.add(fid)
            except KeyboardInterrupt:
                bars.close(); refresh_manifest(); raise
            except Exception:  # noqa: BLE001
                failed_fids.add(fid)
            bars.clear_slot(0)
            if fid not in advanced_fids:   # don't double-count a file already advanced
                bars.advance(weight_by_fid[fid]); advanced_fids.add(fid)
        bars.finish(); bars.close()


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

    failed_fids: set[str] = set()

    # ---- Write unchanged records first (free -- no hashing) ----
    # Doing this before hashing means a Ctrl+C during hashing still leaves every
    # cached record's path/stat refreshed on disk.
    for fid, plan in plans.items():
        if plan.kind != "cached":
            continue
        try:
            rec = replace(plan.cached_rec,          # copy; don't mutate `existing`
                          paths=list(plan.paths),
                          stat_by_path=dict(plan.stats_by_path))
            save_record(rec, plan.folder)
            location[fid] = plan.folder
            if plan.old_home is not None and plan.old_home.resolve() != plan.folder.resolve():
                record_path(plan.old_home, fid).unlink(missing_ok=True)
            stats.unchanged += 1
        except Exception:  # noqa: BLE001
            stats.failed += 1

    # ---- Hash new/stale videos, writing each record AS IT COMPLETES ----
    # Incremental writes make the build resumable: a Ctrl+C (or crash) keeps
    # every video hashed so far; a re-run hashes only what's left.
    def _write_hashed(fid: str, features: list[FrameFeature]) -> None:
        plan = plans[fid]
        rec = assemble_record(plan.hash_path, fid, features, params,
                              plan.stats_by_path[str(plan.hash_path.resolve())])
        rec.paths = list(plan.paths)
        rec.stat_by_path = dict(plan.stats_by_path)
        save_record(rec, plan.folder)
        location[fid] = plan.folder
        if plan.old_home is not None and plan.old_home.resolve() != plan.folder.resolve():
            record_path(plan.old_home, fid).unlink(missing_ok=True)
        stats.added += 1 if plan.kind == "add" else 0
        stats.updated += 1 if plan.kind == "update" else 0

    to_hash = [(p.hash_path, fid) for fid, p in plans.items() if p.kind != "cached"]
    done_fids: set[str] = set()

    # Collection-level status (distinct from the per-run hashing bar below): how
    # much of the whole library is already cached. `plans` holds every video seen
    # this run; to_hash is the subset still needing work.
    if plans:
        ui.collection_status(cached=len(plans) - len(to_hash), total=len(plans))

    if to_hash:
        # --- Progress weight per file (DURATION), computed parent-side, no worker
        # involvement. Cached files use their stored duration (free). New files
        # get a bounded, timed-out parallel metadata read (can't hang); any that
        # don't answer fall back to file size from the Phase-1 stat. ---
        weight_by_fid: dict[str, float] = {}
        # Scan any file WITHOUT a usable cached duration -- brand-new OR a prior
        # record that carries no duration. (A file with a good cached duration
        # reuses it for free.) Show a bar: the scan can take a bit on a large new
        # collection, and a silent gap here would look like a hang.
        def _has_cached_dur(fid: str) -> bool:
            p = existing.get(fid)
            return p is not None and p[0].duration > 0

        scan_paths = [path for path, fid in to_hash if not _has_cached_dur(fid)]
        scanned: dict[str, float] = {}
        if scan_paths:
            from .vid_length import durations_for
            ui.header("Reading durations", detail=f"{len(scan_paths)} video(s)")
            scanned = durations_for(scan_paths, progress=True)  # {resolved-path: s}
        for path, fid in to_hash:
            if _has_cached_dur(fid):
                w = existing[fid][0].duration                # cached duration
            else:
                w = scanned.get(str(path.resolve()), 0.0)    # fresh scan
                if w <= 0:                                   # unreadable -> size
                    sp = plans[fid].stats_by_path.get(str(path.resolve()))
                    w = sp[0] if sp else 0.0
            weight_by_fid[fid] = max(0.01, w)
        total_units = sum(weight_by_fid.values())

        bars = ui.WorkerBars(total_units, len(to_hash),
                             workers if workers > 1 else 0)

        if workers == 1:
            for path, fid in to_hash:
                bars.set_slot(0, path.name)
                try:
                    feats = hash_video(path, params, thread_count=1)
                    _write_hashed(fid, feats); done_fids.add(fid)
                except KeyboardInterrupt:
                    bars.close(); _write_manifest(root, params, location); raise
                except Exception:  # noqa: BLE001 - bad file, skip it
                    failed_fids.add(fid)
                bars.clear_slot(0)
                bars.advance(weight_by_fid[fid])
            bars.finish(); bars.close()
        else:
            _hash_parallel(to_hash, params, workers, weight_by_fid, bars,
                           _write_hashed, done_fids, failed_fids,
                           lambda: _write_manifest(root, params, location))

    # Count bad files (hash or record-write failures) as failed. This runs only
    # on a normal / BrokenProcessPool completion -- a KeyboardInterrupt re-raises
    # above and never reaches here (so stats isn't returned on interrupt at all).
    for fid in failed_fids:
        stats.failed += 1

    # Remove record files for content no longer present, and collapse any
    # duplicate record files for seen content down to the chosen home folder.
    # A fid is "present this run" iff it has a plan AND got a record written
    # (location[fid] is set only on a successful write; a not-yet-hashed fid
    # under an interrupt has a plan but no location, so we leave its files alone).
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
