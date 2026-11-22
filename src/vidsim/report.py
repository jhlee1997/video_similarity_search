"""Format comparison results. Never mutates any video file.

Each reported pair is directional: "N% of B's duration is drawn from A, across
K segments", with every segment timestamped in both videos.
"""

from __future__ import annotations

import json

from .aggregate import PairResult
from .cache import VideoRecord


def _hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _short_path(rec: VideoRecord | None, fid: str) -> str:
    if rec and rec.paths:
        return rec.paths[0]
    return f"<{fid[:12]}>"


def _size_bytes(rec: VideoRecord | None) -> int:
    """Best-known file size (bytes) for a record, from its stat_by_path.

    Uses the stat for the record's primary path when available, else any stat;
    0 when unknown (pre-stat records or missing data)."""
    if not rec or not rec.stat_by_path:
        return 0
    primary = rec.paths[0] if rec.paths else None
    sp = rec.stat_by_path.get(primary) if primary else None
    if sp is None:
        sp = next(iter(rec.stat_by_path.values()), None)
    try:
        return int(sp[0]) if sp else 0
    except (TypeError, ValueError, IndexError):
        return 0


def _human_size(n: int) -> str:
    """Human-readable file size, e.g. '8.9 GB', '540 MB'."""
    if n <= 0:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    i = 0
    while f >= 1000 and i < len(units) - 1:
        f /= 1000.0
        i += 1
    # Pick the display precision, then roll up one more unit if rounding would
    # render "1000" (e.g. 999.6 MB -> "1.0 GB", not "1000 MB").
    decimals = 0 if (i == 0 or f >= 100) else 1
    if round(f, decimals) >= 1000 and i < len(units) - 1:
        f /= 1000.0
        i += 1
        decimals = 1
    return f"{f:.{decimals}f} {units[i]}"


def format_result_line(r: PairResult, records: dict[str, VideoRecord]) -> str:
    b_rec = records.get(r.query_id)
    a_rec = records.get(r.target_id)
    b = _short_path(b_rec, r.query_id)
    a = _short_path(a_rec, r.target_id)

    header = (
        f"{int(round(r.coverage * 100))}% of  {b}\n"
        f"    is drawn from  {a}\n"
        f"    ({len(r.segments)} segment(s); "
        f"B={_hms(r.query_duration)} {_human_size(_size_bytes(b_rec))}  "
        f"A={_hms(r.target_duration)} {_human_size(_size_bytes(a_rec))})"
    )
    lines = [header]
    for seg in r.segments:
        lines.append(
            f"      B {_hms(seg.q_start)}–{_hms(seg.q_end)}"
            f"  ⇐  A {_hms(seg.t_start)}–{_hms(seg.t_end)}"
            f"  ({seg.frames}f)"
        )
    return "\n".join(lines)


def result_to_dict(r: PairResult, records: dict[str, VideoRecord]) -> dict:
    return {
        "query_id": r.query_id,
        "target_id": r.target_id,
        "query_path": _short_path(records.get(r.query_id), r.query_id),
        "target_path": _short_path(records.get(r.target_id), r.target_id),
        "coverage": round(r.coverage, 4),
        "query_duration": round(r.query_duration, 2),
        "target_duration": round(r.target_duration, 2),
        "query_size_bytes": _size_bytes(records.get(r.query_id)),
        "target_size_bytes": _size_bytes(records.get(r.target_id)),
        "segments": [
            {
                "query_start": round(s.q_start, 2),
                "query_end": round(s.q_end, 2),
                "target_start": round(s.t_start, 2),
                "target_end": round(s.t_end, 2),
                "offset": round(s.offset, 2),
                "frames": s.frames,
            }
            for s in r.segments
        ],
    }


def dump_json(results: list[PairResult], records: dict[str, VideoRecord]) -> str:
    return json.dumps([result_to_dict(r, records) for r in results], indent=2)
