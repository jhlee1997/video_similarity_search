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


def format_result_line(r: PairResult, records: dict[str, VideoRecord]) -> str:
    b = _short_path(records.get(r.query_id), r.query_id)
    a = _short_path(records.get(r.target_id), r.target_id)

    header = (
        f"{int(round(r.coverage * 100))}% of  {b}\n"
        f"    is drawn from  {a}\n"
        f"    ({len(r.segments)} segment(s); "
        f"B={_hms(r.query_duration)} A={_hms(r.target_duration)})"
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
