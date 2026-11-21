"""Wrap vpdq.computeHash into our own frame-feature representation.

We keep a plain, serialisable per-frame record (frame_number, timestamp,
quality, hex) and can turn a list of those into a contiguous (N, 32) uint8
matrix for FAISS binary indexes (256 bits = 32 bytes per hash).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vpdq

from .config import HASH_BYTES, HashParams


@dataclass
class FrameFeature:
    """One sampled frame's PDQ hash and metadata."""

    frame_number: int
    timestamp: float
    quality: int
    hex: str  # 64-char hex string == 256-bit PDQ hash

    def to_dict(self) -> dict:
        return {
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "quality": self.quality,
            "hex": self.hex,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FrameFeature":
        return cls(
            frame_number=int(d["frame_number"]),
            timestamp=float(d["timestamp"]),
            quality=int(d["quality"]),
            hex=str(d["hex"]),
        )


def hash_video(path: str | Path, params: HashParams) -> list[FrameFeature]:
    """Compute PDQ frame hashes for *path*, filtered by the quality floor."""
    features = vpdq.computeHash(
        str(path),
        seconds_per_hash=params.seconds_per_hash,
    )
    out: list[FrameFeature] = []
    for f in features:
        if f.quality < params.quality_floor:
            continue
        out.append(
            FrameFeature(
                frame_number=int(f.frame_number),
                timestamp=float(f.timestamp),
                quality=int(f.quality),
                hex=str(f.hex),
            )
        )
    return out


def hex_to_bytes(hex_str: str) -> np.ndarray:
    """One 64-char hex hash -> (32,) uint8."""
    return np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)


def features_to_matrix(features: list[FrameFeature]) -> np.ndarray:
    """List of features -> (N, 32) uint8 matrix for FAISS binary indexes."""
    if not features:
        return np.empty((0, HASH_BYTES), dtype=np.uint8)
    mat = np.empty((len(features), HASH_BYTES), dtype=np.uint8)
    for i, f in enumerate(features):
        mat[i] = np.frombuffer(bytes.fromhex(f.hex), dtype=np.uint8)
    return mat
