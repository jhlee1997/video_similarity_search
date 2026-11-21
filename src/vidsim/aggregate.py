"""CORE: turn raw frame-level matches into directional "derived-from" scores.

Model (no categories): every ordered pair (A, B) gets a number --
    cov(B|A) = fraction of B's duration that is drawn from A
-- together with the list of time-aligned segments that prove it. Everything
else is read off the two directional numbers:

    cov(B|A) ~ 1 and cov(A|B) ~ 1   -> the same whole video ("duplicate")
    cov(B|A) ~ 1 and cov(A|B) small -> B is a clip/subset of A
    cov(B|A) high, many segments    -> B is an edit/highlights built from A
    both small                      -> incidental shared snippet

Method: temporal Hough voting (standard video-copy-localization technique).
A PDQ hash describes one frame as an image; it carries no time or identity. So
FAISS gives us only "frame at B-time b looks like frame at A-time a". If B
genuinely reuses a chunk of A, every true match in that chunk shares a constant
time offset delta = a - b. We therefore:

  1. turn each frame match into a vote for its offset delta,
  2. down-weight bursts (a B-frame matching many near-identical A-frames votes
     1/burst for each, so near-static scenes can't stuff one bin),
  3. accumulate votes into offset bins; every TALL bin is a real shared segment
     (a reordered highlight reel produces several clean peaks -- order in the
     sequence is irrelevant to the offset, so jumbling doesn't hurt),
  4. recover each segment's span, merge overlapping spans in B, and sum their
     length -> directional coverage.

Bursts scatter (one frame -> many offsets) so noise spreads thin across bins;
only genuine temporal structure stacks into a peak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MatchParams
from .cache import VideoRecord
from .hashing import features_to_matrix
from .index import FrameIndex


@dataclass
class Segment:
    """A time-aligned shared segment between query (B) and target (A)."""

    q_start: float  # span in the query video (B)
    q_end: float
    t_start: float  # span in the target video (A)
    t_end: float
    offset: float   # a - b for this segment
    frames: int     # aligned frame votes supporting it

    @property
    def q_duration(self) -> float:
        return max(0.0, self.q_end - self.q_start)


@dataclass
class PairResult:
    """Directional 'B is drawn from A' result for one (query=B, target=A) pair."""

    query_id: str     # B
    target_id: str    # A
    # Fraction of the QUERY (B) duration that is drawn from the target (A).
    coverage: float
    query_duration: float
    target_duration: float
    segments: list[Segment]

    @property
    def total_aligned(self) -> float:
        """Seconds of B covered by merged segments."""
        return self.coverage * self.query_duration


def _collect_hits(
    query: VideoRecord,
    frame_index: FrameIndex,
    mp: MatchParams,
) -> dict[str, list[tuple[float, float]]]:
    """Return, per target id, a list of (b_time, a_time) frame matches.

    Burst-aware: a single query frame that matches several target frames of the
    SAME target contributes a fractional vote 1/burst to each, applied later in
    voting. Here we just record the raw (b, a) pairs plus the burst size.
    """
    queries = features_to_matrix(query.features)
    dists, rows = frame_index.search(queries, mp.knn)
    q_ts = np.array([f.timestamp for f in query.features])

    hits: dict[str, list[tuple[float, float]]] = {}
    for qi in range(rows.shape[0]):
        # Gather this query frame's accepted matches, grouped by target.
        per_target: dict[str, list[float]] = {}
        for k in range(rows.shape[1]):
            row = int(rows[qi, k])
            if row < 0 or int(dists[qi, k]) > mp.frame_hamming_radius:
                continue
            tgt = str(frame_index.file_ids[row])
            if tgt == query.file_id:
                continue
            per_target.setdefault(tgt, []).append(float(frame_index.timestamps[row]))
        for tgt, a_times in per_target.items():
            dst = hits.setdefault(tgt, [])
            for a in a_times:
                dst.append((float(q_ts[qi]), a))
    return hits


def _vote_segments(
    hits: list[tuple[float, float]], mp: MatchParams
) -> list[Segment]:
    """Temporal Hough voting -> list of aligned segments for one target.

    hits: list of (b_time, a_time). We bin delta = a - b, find peaks, and for
    each peak recover the covered span. Burst weighting: votes for a given
    b_time are shared across the target frames it matched.
    """
    if not hits:
        return []

    tol = mp.offset_tolerance_s
    # Burst weight: how many targets did each distinct b_time match?
    from collections import Counter

    burst = Counter(b for b, _ in hits)

    # Weighted histogram over delta, binned at the offset tolerance.
    bin_w = max(tol, 1e-6)
    bin_votes: dict[int, float] = {}
    bin_members: dict[int, list[tuple[float, float, float]]] = {}
    for b, a in hits:
        delta = a - b
        key = int(round(delta / bin_w))
        w = 1.0 / burst[b]
        bin_votes[key] = bin_votes.get(key, 0.0) + w
        bin_members.setdefault(key, []).append((b, a, delta))

    segments: list[Segment] = []
    min_frames = mp.min_segment_frames

    # Each sufficiently-voted bin (and its tolerance-neighbours) is a segment.
    # Process strongest first; consume neighbouring bins so one physical segment
    # isn't split across two adjacent keys.
    consumed: set[int] = set()
    for key in sorted(bin_votes, key=lambda k: bin_votes[k], reverse=True):
        if key in consumed:
            continue
        members: list[tuple[float, float, float]] = []
        for nb in (key - 1, key, key + 1):
            if nb in bin_members and nb not in consumed:
                members.extend(bin_members[nb])
                consumed.add(nb)
        # Distinct aligned query frames in this segment.
        b_times = sorted({round(b, 3) for b, _, _ in members})
        if len(b_times) < min_frames:
            continue
        a_vals = [a for _, a, _ in members]
        deltas = [d for _, _, d in members]
        segments.append(
            Segment(
                q_start=min(b for b, _, _ in members),
                q_end=max(b for b, _, _ in members),
                t_start=min(a_vals),
                t_end=max(a_vals),
                offset=float(np.median(deltas)),
                frames=len(b_times),
            )
        )
    return segments


def _merge_spans(spans: list[tuple[float, float]], gap: float) -> list[tuple[float, float]]:
    """Merge overlapping/adjacent (start, end) spans so coverage isn't double-counted."""
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def compare(
    query: VideoRecord,
    frame_index: FrameIndex,
    mp: MatchParams,
    targets: dict[str, VideoRecord],
) -> list[PairResult]:
    """Compare query video B against the index; one PairResult per target A.

    coverage = (merged length of B's aligned segments) / B's duration.
    """
    if not query.features:
        return []

    q_duration = query.duration or max(
        (f.timestamp for f in query.features), default=0.0
    )
    hits_by_target = _collect_hits(query, frame_index, mp)

    results: list[PairResult] = []
    for tgt, hits in hits_by_target.items():
        if len(hits) < mp.min_shared_frames:
            continue
        segments = _vote_segments(hits, mp)
        if not segments:
            continue
        # Coverage of B = merged length of segment spans in B / B duration.
        spans = [(s.q_start, s.q_end) for s in segments]
        merged = _merge_spans(spans, gap=mp.offset_tolerance_s)
        covered = sum(e - s for s, e in merged)
        # Each aligned frame represents ~seconds_per_hash of content; add one
        # sampling interval so a single-frame-wide span isn't counted as zero.
        covered += len(merged) * 0  # spans already span first..last matched frame
        coverage = min(1.0, covered / q_duration) if q_duration > 0 else 0.0

        tgt_rec = targets.get(tgt)
        t_duration = (tgt_rec.duration if tgt_rec else 0.0)

        results.append(
            PairResult(
                query_id=query.file_id,
                target_id=tgt,
                coverage=coverage,
                query_duration=q_duration,
                target_duration=t_duration,
                segments=sorted(segments, key=lambda s: s.q_start),
            )
        )

    results.sort(key=lambda r: r.coverage, reverse=True)
    return results
