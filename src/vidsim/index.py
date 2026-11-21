"""FAISS binary index over all per-frame PDQ hashes.

The index is a flat pool of every frame hash from every cached video. A parallel
row-map records, for each index row, which (file_id, frame_number, timestamp) it
came from, so a neighbour hit can be traced back to a video and a moment.

Index type is swappable via `kind`:
  - "flat": IndexBinaryFlat -- exact Hamming search. Correct, simple, O(N) scan.
  - "hnsw": IndexBinaryHNSW -- approximate, much faster at tens of millions of
            rows. Switch to this when flat lookups get slow.

The stored hashes are untouched by this choice, so you can rebuild with a
different kind without recomputing any video.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from .config import HASH_BITS, index_path, rowmap_path
from .cache import VideoRecord
from .hashing import features_to_matrix

HNSW_M = 32  # neighbours per node for IndexBinaryHNSW


def _new_index(kind: str) -> faiss.IndexBinary:
    if kind == "flat":
        return faiss.IndexBinaryFlat(HASH_BITS)
    if kind == "hnsw":
        return faiss.IndexBinaryHNSW(HASH_BITS, HNSW_M)
    raise ValueError(f"unknown index kind: {kind!r}")


@dataclass
class FrameIndex:
    """A built FAISS binary index plus its row-map."""

    index: faiss.IndexBinary
    file_ids: np.ndarray       # (N,) object array of file_id per row
    frame_numbers: np.ndarray  # (N,) int32
    timestamps: np.ndarray     # (N,) float32

    def __len__(self) -> int:
        return self.index.ntotal

    # ---- build ----

    @classmethod
    def build(cls, records: dict[str, VideoRecord], kind: str = "flat") -> "FrameIndex":
        mats: list[np.ndarray] = []
        fids: list[str] = []
        fnums: list[int] = []
        tss: list[float] = []

        for fid, rec in records.items():
            if not rec.features:
                continue
            mats.append(features_to_matrix(rec.features))
            for feat in rec.features:
                fids.append(fid)
                fnums.append(feat.frame_number)
                tss.append(feat.timestamp)

        index = _new_index(kind)
        if mats:
            data = np.vstack(mats)
            index.add(data)

        return cls(
            index=index,
            file_ids=np.array(fids, dtype=object),
            frame_numbers=np.array(fnums, dtype=np.int32),
            timestamps=np.array(tss, dtype=np.float32),
        )

    # ---- persistence ----

    def save(self) -> None:
        index_path().parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index_binary(self.index, str(index_path()))
        np.savez(
            rowmap_path(),
            file_ids=self.file_ids,
            frame_numbers=self.frame_numbers,
            timestamps=self.timestamps,
        )

    @classmethod
    def load(cls) -> "FrameIndex | None":
        if not index_path().exists() or not rowmap_path().exists():
            return None
        index = faiss.read_index_binary(str(index_path()))
        data = np.load(rowmap_path(), allow_pickle=True)
        return cls(
            index=index,
            file_ids=data["file_ids"],
            frame_numbers=data["frame_numbers"],
            timestamps=data["timestamps"],
        )

    # ---- query ----

    def search(self, queries: np.ndarray, knn: int) -> tuple[np.ndarray, np.ndarray]:
        """k-NN search. Returns (distances, row-indices), each (Q, knn)."""
        if self.index.ntotal == 0 or queries.shape[0] == 0:
            q = queries.shape[0]
            return (
                np.empty((q, 0), dtype=np.int32),
                np.empty((q, 0), dtype=np.int64),
            )
        k = min(knn, self.index.ntotal)
        return self.index.search(queries, k)
