"""Central configuration and tunable defaults for vidsim.

Everything that affects *what gets stored* (sample rate, quality floor, schema)
is versioned so the cache can tell when it is stale. Tunables that only affect
*matching* (Hamming radius, thresholds) can change freely without invalidating
the cache.
"""

from __future__ import annotations

import os
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
    min_shared_frames: int = 4

    # Temporal Hough voting: offsets are binned at this width (seconds), and two
    # matches within this many seconds count as the same alignment. Absorbs
    # re-encode / sampling jitter.
    offset_tolerance_s: float = 1.5

    # A histogram peak must gather at least this many distinct aligned query
    # frames to count as a real shared segment -- prunes incidental collisions
    # (black frames, logos, coincidental look-alikes). ~3s at 1 fps.
    min_segment_frames: int = 3

    # Only report a pair whose directional coverage reaches this. A pair is
    # reported if EITHER direction's coverage >= this (so subsets surface even
    # when the reverse coverage is tiny).
    report_coverage: float = 0.15


# --- Paths ---------------------------------------------------------------


def cache_dir() -> Path:
    """Root cache directory. Override with VIDSIM_CACHE_DIR."""
    env = os.environ.get("VIDSIM_CACHE_DIR")
    base = Path(env) if env else Path.home() / ".cache" / "vidsim"
    return base


def features_dir() -> Path:
    """Per-video hash records live here (one file per video)."""
    return cache_dir() / "features"


def index_path() -> Path:
    """Serialised FAISS binary index."""
    return cache_dir() / "index.faiss"


def rowmap_path() -> Path:
    """Row-map sidecar: maps each index row -> (file_id, frame_number, timestamp)."""
    return cache_dir() / "rowmap.npz"


DEFAULT_HASH_PARAMS = HashParams()
DEFAULT_MATCH_PARAMS = MatchParams()
