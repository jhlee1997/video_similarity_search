"""Unit tests for the temporal Hough voting core (`aggregate._vote_segments`).

These run on raw (b_time, a_time) hit lists, so they need no video, no FAISS,
and no vpdq -- they pin the segment-recovery logic directly.

The bug these guard against: frames that merely share a time-offset but are
scattered across B used to be fabricated into one contiguous segment (e.g. 3
frames over 11s reported as an 11-second shared segment). A real segment must be
a run of temporally-consecutive aligned frames.
"""

from __future__ import annotations

from vidsim.aggregate import _vote_segments
from vidsim.config import MatchParams

MP = MatchParams()  # defaults: min_segment_frames=3, max_frame_gap_s=3.0


def test_scattered_frames_do_not_become_one_segment() -> None:
    """3 same-offset matches 5-6s apart are noise, not an 11s segment."""
    hits = [(1.0, 15.0), (6.0, 20.0), (12.0, 26.0)]  # all offset +14, far apart
    segs = _vote_segments(hits, MP)
    # No fabricated wide segment: nothing should claim a multi-second span from
    # frames that are never within max_frame_gap_s of each other.
    assert all(s.q_duration <= MP.max_frame_gap_s for s in segs)
    # In fact each frame is isolated, so no run reaches min_segment_frames.
    assert segs == []


def test_contiguous_clip_is_one_segment() -> None:
    """A clean run of consecutive aligned frames -> exactly one segment."""
    hits = [(float(b), float(b + 48)) for b in range(10)]  # b=0..9, offset +48
    segs = _vote_segments(hits, MP)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.frames == 10
    assert seg.q_start == 0.0 and seg.q_end == 9.0
    assert seg.t_start == 48.0 and seg.t_end == 57.0
    assert abs(seg.offset - 48.0) < 1e-6


def test_gap_splits_into_two_segments() -> None:
    """Two contiguous runs at the same offset, separated by a big B-time gap,
    are two distinct shared segments -- not one bridged span."""
    run1 = [(float(b), float(b + 48)) for b in range(4)]       # 0..3
    run2 = [(float(b), float(b + 48)) for b in range(40, 44)]  # 40..43
    segs = _vote_segments(run1 + run2, MP)
    assert len(segs) == 2
    spans = sorted((s.q_start, s.q_end) for s in segs)
    assert spans == [(0.0, 3.0), (40.0, 43.0)]


def test_run_below_min_frames_is_dropped() -> None:
    """A contiguous run shorter than min_segment_frames is not a segment."""
    hits = [(0.0, 48.0), (1.0, 49.0)]  # only 2 frames, min is 3
    assert _vote_segments(hits, MP) == []


def test_no_hits() -> None:
    assert _vote_segments([], MP) == []


def test_coarse_sampling_interval_does_not_destroy_runs() -> None:
    """Regression: at a coarse sampling interval (e.g. 5s), consecutive aligned
    frames are 5s apart. With a fixed 3s gap they'd each split into size-1 runs
    and no segment would form. Passing the interval must keep them as one run."""
    # 5 consecutive aligned frames, 5s apart, same offset +100.
    hits = [(float(b), float(b + 100)) for b in range(0, 25, 5)]  # 0,5,10,15,20
    # Without the interval (fallback to 3s floor) -> broken: every frame alone.
    assert _vote_segments(hits, MP) == []
    # With the real 5s interval -> one contiguous segment of 5 frames.
    segs = _vote_segments(hits, MP, sampling_interval=5.0)
    assert len(segs) == 1
    assert segs[0].frames == 5
    assert segs[0].q_start == 0.0 and segs[0].q_end == 20.0
