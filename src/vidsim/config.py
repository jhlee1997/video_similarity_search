"""Central configuration and tunable defaults for vidsim.

Everything that affects *what gets stored* (sample rate, quality floor, schema)
is versioned so the cache can tell when it is stale. Tunables that only affect
*matching* (Hamming radius, thresholds) can change freely without invalidating
the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Schema / versioning -------------------------------------------------

# Bump when the on-disk cache record layout changes in a breaking way.
SCHEMA_VERSION = 1

# PDQ hashes are 256 bits = 32 bytes.
HASH_BITS = 256
HASH_BYTES = HASH_BITS // 8

# Recognised video extensions (lowercase, no dot).
VIDEO_EXTS = {
    "mp4", "mkv", "webm", "mov", "avi", "m4v", "flv", "wmv",
    "mpg", "mpeg", "ts", "m2ts", "3gp", "ogv", "gif",
}


# --- Oversized-file exclusion (always on) --------------------------------
#
# Some videos are prohibitively expensive to hash at ~1 frame/sec: 4K, long,
# high-fps files decode every frame regardless of sample rate, so a single
# 4K60 two-hour file can occupy a worker for hours (measured). Such a file is
# SKIPPED by cache-build -- never hashed, no cache record, absent from the
# index and therefore from `find`/`pairs`. The video file itself is never
# touched. A file is skipped only when ALL THREE conditions hold, so an
# ordinary big-but-short or long-but-low-res video is still hashed normally:
#
#   resolution >= 4K   (width >= SKIP_MIN_WIDTH OR height >= SKIP_MIN_HEIGHT)
#   duration   >  SKIP_MIN_DURATION_S
#   file size  >  SKIP_MIN_SIZE_BYTES
#
# The resolution check needs a metadata probe, so it runs LAST -- only on files
# that already pass the free size gate and the (usually cached) duration gate.
SKIP_MIN_WIDTH = 3840
SKIP_MIN_HEIGHT = 2160
SKIP_MIN_DURATION_S = 20 * 60      # 20 minutes
SKIP_MIN_SIZE_BYTES = 3 * 1000**3  # 3 GB (decimal, matches `du`/df reporting)


# --- Hashing parameters (part of cache identity) -------------------------


@dataclass(frozen=True)
class HashParams:
    """Parameters that change the stored hashes. Part of cache staleness key."""

    seconds_per_hash: float = 1.0  # vpdq sampling interval; 1.0 == ~1 fps
    quality_floor: int = 50        # drop frames below this PDQ quality (0-100)

    def fingerprint(self) -> str:
        """Stable string identifying this parameter set."""
        return f"sph={self.seconds_per_hash:g};qf={self.quality_floor}"


# --- Matching parameters (do NOT affect cache identity) ------------------


@dataclass(frozen=True)
class MatchParams:
    """Parameters that only affect comparison, not stored hashes."""

    # A per-frame PDQ match: Hamming distance <= this (out of 256 bits).
    # 31 is the ThreatExchange/PDQ conventional strong-match threshold.
    frame_hamming_radius: int = 31

    # Candidate generation: nearest neighbours fetched per query frame.
    knn: int = 16

    # A (X, Y) pair must share at least this many candidate frame matches
    # before we bother running full temporal aggregation (prunes O(N^2)).
    min_shared_frames: int = 2

    # Temporal Hough voting: offsets are binned at this width (seconds), and two
    # matches within this many seconds count as the same alignment. Absorbs
    # re-encode / sampling jitter.
    offset_tolerance_s: float = 1.5

    # A histogram peak must gather at least this many distinct aligned query
    # frames to count as a real shared segment -- prunes incidental collisions
    # (black frames, logos, coincidental look-alikes). ~3s at 1 fps.
    min_segment_frames: int = 3

    # FLOOR for the run-splitting gap. Within one offset bin, aligned frames
    # more than the effective gap apart in B-time belong to DIFFERENT shared
    # segments (the reuse stopped and later resumed at the same offset). This
    # absorbs a dropped/low-quality frame or two without bridging genuine gaps,
    # and stops scattered same-offset frames being fabricated into one segment.
    # The EFFECTIVE gap used by _vote_segments is max(this, 2.5 x the sampling
    # interval), so coarse --seconds-per-hash values don't split every run into
    # single frames (which would destroy all matching). See _vote_segments.
    max_frame_gap_s: float = 3.0

    # Only report a pair whose directional coverage reaches this. A pair is
    # reported if EITHER direction's coverage >= this (so subsets surface even
    # when the reverse coverage is tiny). This is a DISPLAY filter only: the
    # pairs cache stores every pair the aggregator emits, and this threshold
    # just filters what's shown, so changing it never forces a recompute.
    report_coverage: float = 0.05


# --- Paths ---------------------------------------------------------------
#
# The cache lives WITH the videos. Each folder that directly contains videos
# gets a hidden `.vidsim/` holding one `<file_id>.json` record per video in it.
# The collection root (the folder you run `cache-build` on) additionally holds
# the global index, its row-map, and the manifest in its own `.vidsim/`:
#
#   A/.vidsim/{index.faiss, rowmap.npz, collection.json}
#   A/B/.vidsim/<file_id>.json   (records for videos directly in B)
#
# Record filenames are `<file_id>.json`; the collection-level files have fixed
# names, so they never collide even when a video sits directly in the root.

VIDSIM_DIRNAME = ".vidsim"


def vidsim_dir(folder: str | Path) -> Path:
    """The `.vidsim/` directory inside *folder*."""
    return Path(folder) / VIDSIM_DIRNAME


def record_path(folder: str | Path, fid: str) -> Path:
    """Path to the hash record for *fid* stored in *folder*'s `.vidsim/`."""
    return vidsim_dir(folder) / f"{fid}.json"


def index_path(root: str | Path) -> Path:
    """Serialised FAISS binary index for the collection rooted at *root*."""
    return vidsim_dir(root) / "index.faiss"


def rowmap_path(root: str | Path) -> Path:
    """Row-map sidecar: index row -> (file_id, frame_number, timestamp)."""
    return vidsim_dir(root) / "rowmap.npz"


def manifest_path(root: str | Path) -> Path:
    """Collection manifest: schema/params fingerprint + file_id -> folder map."""
    return vidsim_dir(root) / "collection.json"


def pairs_cache_path(root: str | Path) -> Path:
    """Cached `pairs` results for the collection rooted at *root*."""
    return vidsim_dir(root) / "pairs.json"


DEFAULT_HASH_PARAMS = HashParams()
DEFAULT_MATCH_PARAMS = MatchParams()
