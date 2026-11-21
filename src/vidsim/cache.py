"""Per-video hash cache: one JSON record per video, keyed by content file_id.

A record stores the frame features plus enough provenance (hash params, vpdq
version, schema version) to detect when it is stale and must be recomputed.

Keeping one file per video (rather than one big DB) makes incremental sync
cheap and crash-safe: adding or dropping a video touches exactly one file.
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass, field
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


def build_record(path: Path, fid: str, params: HashParams) -> VideoRecord:
    """Hash a video and assemble its cache record."""
    features = hash_video(path, params)
    duration = max((f.timestamp for f in features), default=0.0)
    return VideoRecord(
        file_id=fid,
        paths=[str(path.resolve())],
        duration=duration,
        frame_count=len(features),
        params_fingerprint=params.fingerprint(),
        vpdq_version=VPDQ_VERSION,
        schema_version=SCHEMA_VERSION,
        features=features,
    )


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


def sync(root: str | Path, params: HashParams) -> SyncStats:
    """Reconcile the per-folder cache with the videos currently under *root*.

    - new content          -> hash, write record in the video's own folder
    - stale record         -> recompute, written to the video's current folder
    - unchanged            -> keep hashes; record follows the video if it moved
    - byte-identical copy  -> share one record (first folder seen); add its path
    - content gone          -> drop its record
    Duplicate record files for the same content are collapsed to one. Writes
    collection.json and prunes emptied `.vidsim/` dirs. Does NOT build the index
    (the caller does that) so this stays pure cache reconciliation.
    """
    root = Path(root)
    stats = SyncStats()
    existing = load_all_records(root)             # fid -> (rec, old home folder)
    all_record_files = _scan_record_files(root)   # fid -> every record file path
    seen_ids: set[str] = set()
    home_rec: dict[str, VideoRecord] = {}          # fid -> the live record (in memory)
    location: dict[str, Path] = {}                 # fid -> folder its record lives in

    files = list(iter_video_files(root))
    bar = tqdm(files, desc="scanning", unit="vid")
    for path in bar:
        try:
            fid = file_id(path)
        except OSError:
            stats.failed += 1
            continue
        folder = path.parent
        rp = str(path.resolve())
        try:
            if fid in seen_ids:
                # Byte-identical copy already handled this run: add its path to
                # the one shared record (kept in memory, so no fragile re-read).
                rec = home_rec[fid]
                if rp not in rec.paths:
                    rec.paths.append(rp)
                    save_record(rec, location[fid])
                stats.duplicate_copies += 1
                continue

            seen_ids.add(fid)
            prior = existing.get(fid)
            if prior is None:
                # New content: this is where the time goes (ffmpeg decode + hash).
                # Surface the file so a long pause reads as "hashing the big one",
                # not a hang.
                bar.set_postfix_str(f"hashing {path.name}", refresh=True)
                rec = build_record(path, fid, params)
                save_record(rec, folder)
                stats.added += 1
            else:
                old_rec, old_home = prior
                if old_rec.is_stale(params):
                    bar.set_postfix_str(f"rehashing {path.name}", refresh=True)
                    rec = build_record(path, fid, params)  # paths reset to [rp]
                    stats.updated += 1
                else:
                    rec = old_rec
                    rec.paths = [rp]  # drop stale paths; copies re-append above
                    stats.unchanged += 1
                # The record follows the video to its current folder. If the
                # video moved, write to the new folder and drop the old file.
                save_record(rec, folder)
                if old_home.resolve() != folder.resolve():
                    record_path(old_home, fid).unlink(missing_ok=True)
            home_rec[fid] = rec
            location[fid] = folder
        except Exception:  # noqa: BLE001 - a bad file shouldn't abort the run
            stats.failed += 1

    # Remove record files for content no longer present, and collapse any
    # duplicate record files for seen content down to the chosen home folder.
    for fid, paths in all_record_files.items():
        home = location.get(fid)
        if fid not in seen_ids:
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
