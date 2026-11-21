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

from .config import SCHEMA_VERSION, VIDEO_EXTS, HashParams, features_dir
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


def _record_path(fid: str) -> Path:
    return features_dir() / f"{fid}.json"


def load_record(fid: str) -> VideoRecord | None:
    p = _record_path(fid)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return VideoRecord.from_dict(json.load(f))
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def save_record(rec: VideoRecord) -> None:
    features_dir().mkdir(parents=True, exist_ok=True)
    p = _record_path(rec.file_id)
    tmp = p.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(rec.to_dict(), f)
    tmp.replace(p)  # atomic on POSIX


def load_all_records() -> dict[str, VideoRecord]:
    """Load every cached record, keyed by file_id."""
    out: dict[str, VideoRecord] = {}
    d = features_dir()
    if not d.exists():
        return out
    for p in d.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                rec = VideoRecord.from_dict(json.load(f))
            out[rec.file_id] = rec
        except (json.JSONDecodeError, KeyError, OSError):
            continue
    return out


def iter_video_files(root: str | Path):
    """Yield video file paths under *root*, recursively."""
    root = Path(root)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in VIDEO_EXTS:
            yield p


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


def sync(root: str | Path, params: HashParams) -> SyncStats:
    """Reconcile the cache with the videos currently under *root*.

    - new content -> hash and add
    - stale record (schema/vpdq/params changed) -> recompute
    - unchanged -> leave, just refresh known path
    - records whose content is no longer present anywhere under root -> drop
    """
    stats = SyncStats()
    existing = load_all_records()
    seen_ids: set[str] = set()

    files = list(iter_video_files(root))
    for path in tqdm(files, desc="scanning", unit="vid"):
        try:
            fid = file_id(path)
        except OSError:
            stats.failed += 1
            continue
        seen_ids.add(fid)
        rec = existing.get(fid)
        try:
            if rec is None:
                save_record(build_record(path, fid, params))
                stats.added += 1
            elif rec.is_stale(params):
                save_record(build_record(path, fid, params))
                stats.updated += 1
            else:
                # Keep hashes; refresh path if it moved.
                rp = str(path.resolve())
                if rp not in rec.paths:
                    rec.paths = [rp] + [p for p in rec.paths if p != rp]
                    save_record(rec)
                stats.unchanged += 1
        except Exception:  # noqa: BLE001 - hashing a bad file shouldn't abort the run
            stats.failed += 1

    # Drop records whose content was not seen anywhere under root.
    for fid in existing:
        if fid not in seen_ids:
            _record_path(fid).unlink(missing_ok=True)
            stats.removed += 1

    return stats
