"""Per-video hash cache: one JSON record per video, keyed by content file_id.

A record stores the frame features plus enough provenance (hash params, vpdq
version, schema version) to detect when it is stale and must be recomputed.

Keeping one file per video (rather than one big DB) makes incremental sync
cheap and crash-safe: adding or dropping a video touches exactly one file.
"""

from __future__ import annotations

import importlib.metadata
import json
import multiprocessing
import os
import queue as queue_mod
from concurrent.futures import ProcessPoolExecutor, as_completed
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


# Per-worker queues are handed to each worker process once, at pool start, via
# the executor initializer -- NOT via a multiprocessing.Manager (its server
# process can deadlock the parent handshake on macOS spawn: scan done, hashing
# never starts, CPU idle). They live as module globals in the worker process.
_SLOT_Q = None
_STATUS_Q = None


def _init_worker(slot_q, status_q) -> None:
    """ProcessPoolExecutor initializer: stash the queues as worker globals."""
    global _SLOT_Q, _STATUS_Q
    _SLOT_Q, _STATUS_Q = slot_q, status_q


def _hash_job_reporting(args) -> tuple[str, Path, list[FrameFeature]]:
    """Like _hash_job, but reports start/done over the worker-global queues for
    live per-worker progress bars. Claims a free bar-slot, announces the file,
    releases the slot in a finally so a bad file never permanently occupies a
    worker line. Workers only SEND events; the parent owns all drawing. If the
    queues aren't set (shouldn't happen), degrade to plain hashing.
    """
    path_str, fid, params = args
    if _SLOT_Q is None or _STATUS_Q is None:
        return fid, Path(path_str), hash_video(path_str, params, thread_count=1)
    slot = _SLOT_Q.get()  # blocks until a slot is free (always is: one per worker)
    try:
        _STATUS_Q.put(("start", slot, fid, Path(path_str).name))
        features = hash_video(path_str, params, thread_count=1)
        return fid, Path(path_str), features
    finally:
        _STATUS_Q.put(("done", slot, fid, ""))
        _SLOT_Q.put(slot)


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

    if to_hash:
        # Weight the overall bar by video DURATION so a long video owns a
        # proportional share (not "1 of N"). For files we've hashed before, the
        # exact duration is already stored in their prior record -- free. Only
        # genuinely new files need a metadata read, via vid_length (the same path
        # `vidsim duration` uses; reads container metadata, does not decode, and
        # is proven not to hang). Unreadable -> tiny epsilon so it still advances.
        from .vid_length import duration as _video_duration

        dur_by_fid: dict[str, float] = {}
        for path, fid in to_hash:
            prior = existing.get(fid)
            d = prior[0].duration if (prior and prior[0].duration > 0) else _video_duration(path)
            dur_by_fid[fid] = d if d > 0 else 0.01
        total_units = sum(dur_by_fid.values())

        bars = ui.WorkerBars(total_units, len(to_hash),
                             workers if workers > 1 else 0)

        if workers == 1:
            for path, fid in to_hash:
                bars.on_start(0, path.name)
                try:
                    feats = hash_video(path, params, thread_count=1)
                    _write_hashed(fid, feats); done_fids.add(fid)
                except KeyboardInterrupt:
                    bars.close(); _write_manifest(root, params, location); raise
                except Exception:  # noqa: BLE001 - bad file, skip it
                    failed_fids.add(fid)
                bars.on_done(0)
                bars.advance(dur_by_fid[fid])
        else:
            jobs = [(str(p), fid, params) for p, fid in to_hash]
            # Plain spawn-context queues handed to workers via the pool
            # initializer -- NO multiprocessing.Manager (its server process can
            # deadlock the parent on macOS spawn: scan done, hashing never starts).
            ctx = multiprocessing.get_context("spawn")
            slot_q = ctx.Queue()
            status_q = ctx.Queue()
            for k in range(workers):
                slot_q.put(k)

            # Background drain: workers emit start/done events continuously, but
            # as_completed only wakes the main thread when a FILE FINISHES. If all
            # workers are on long videos, nothing completes for minutes -- so a
            # dedicated thread drains the queue so the per-worker lines stay live
            # the whole time. It blocks on get(timeout) (no busy-wait) and exits
            # when signalled. All bar writes are lock-guarded in WorkerBars.
            import threading
            stop_drain = threading.Event()

            def _drain_loop() -> None:
                while not stop_drain.is_set():
                    try:
                        kind, slot, _fid, name = status_q.get(timeout=0.2)
                    except queue_mod.Empty:
                        continue
                    except (EOFError, OSError, BrokenPipeError, ValueError):
                        return  # queue closed / torn down
                    if kind == "start":
                        bars.on_start(slot, name)
                    else:
                        bars.on_done(slot)

            drain = threading.Thread(target=_drain_loop, daemon=True)
            drain.start()

            def _stop_drain() -> None:
                stop_drain.set()
                drain.join(timeout=2)

            def _close_queues(graceful: bool) -> None:
                # Plain spawn Queues have a background feeder thread that must be
                # released, or it can delay interpreter exit (the Manager used to
                # do this for us). On a hard teardown (workers SIGKILLed mid-put),
                # a feeder may hold the write lock -- cancel_join_thread() drops
                # the buffered bytes instead of blocking on it.
                for q in (slot_q, status_q):
                    try:
                        if not graceful:
                            q.cancel_join_thread()
                        q.close()
                        if graceful:
                            q.join_thread()
                    except Exception:  # noqa: BLE001
                        pass

            pool = ProcessPoolExecutor(
                max_workers=workers, mp_context=ctx,
                initializer=_init_worker, initargs=(slot_q, status_q),
            )
            try:
                futures = {
                    pool.submit(_hash_job_reporting, (path_str, fid, job_params)): fid
                    for path_str, fid, job_params in jobs
                }
                for fut in as_completed(futures):
                    job_fid = futures[fut]
                    try:
                        fid, path, features = fut.result()
                    except BrokenProcessPool:
                        raise
                    except Exception:  # noqa: BLE001 - ordinary bad file, skip
                        failed_fids.add(job_fid)
                        bars.advance(dur_by_fid[job_fid])
                    else:
                        try:
                            _write_hashed(fid, features)
                            done_fids.add(fid)
                        except Exception:  # noqa: BLE001 - write failed, skip
                            failed_fids.add(fid)
                        bars.advance(dur_by_fid[fid])
                pool.shutdown(wait=True)
                _stop_drain()
                _close_queues(graceful=True)
            except KeyboardInterrupt:
                # Hard-kill in-flight workers so Ctrl+C returns promptly; records
                # written so far are already on disk. Stop the drain thread so it
                # isn't orphaned, release the queues, refresh manifest.
                _shutdown_now(pool)
                _stop_drain()
                _close_queues(graceful=False)
                bars.close()
                _write_manifest(root, params, location)
                raise
            except BrokenProcessPool:
                _shutdown_now(pool)
                _stop_drain()                 # no more queue events will come
                _close_queues(graceful=False)
                bars.idle_all()               # killed workers never sent "done"
                # Finish un-done jobs single-process so one bad file can't fail the
                # whole batch; the crasher raises on retry and is marked failed.
                leftover = [(Path(j[0]), j[1]) for j in jobs
                            if j[1] not in done_fids and j[1] not in failed_fids]
                for path, fid in leftover:
                    bars.on_start(0, path.name)
                    try:
                        feats = hash_video(path, params, thread_count=1)
                        _write_hashed(fid, feats); done_fids.add(fid)
                    except KeyboardInterrupt:
                        bars.close(); _write_manifest(root, params, location); raise
                    except Exception:  # noqa: BLE001 - this one is the bad apple
                        failed_fids.add(fid)
                    bars.on_done(0)
                    bars.advance(dur_by_fid[fid])
        bars.finish()
        bars.close()

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
