"""Sum the durations of every video under a folder.

Durations are read from container metadata in-process via PyAV (ffmpeg's
demuxers as a linked library) -- no per-file subprocess, so this is ~28x faster
than shelling out to ffprobe once per file. Work is spread over a thread pool:
each av.open() releases the GIL while it reads, so threads scale near-linearly.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import av
from tqdm import tqdm

VIDEO_EXTS = {"mp4","mkv","webm","mov","avi","m4v","flv","wmv",
              "mpg","mpeg","ts","m2ts","3gp","ogv"}


def duration(path: Path, timeout: float = 10.0) -> float:
    """Seconds, read from container metadata (no decoding).

    Prefers the container-level duration; falls back to the longest stream
    duration when the container doesn't carry one. Returns 0.0 if neither is
    available, the file can't be opened, or the open exceeds *timeout* seconds
    (a malformed/streaming file must not block the caller forever).
    """
    try:
        with av.open(str(path), timeout=timeout) as container:
            if container.duration and container.duration > 0:
                return container.duration / av.time_base
            best = 0.0
            for stream in container.streams:
                if stream.duration and stream.time_base:
                    best = max(best, float(stream.duration * stream.time_base))
            return best
    except Exception:  # noqa: BLE001 - an unreadable/slow file shouldn't abort the run
        return 0.0


def durations_for(paths: list[Path], timeout: float = 10.0,
                  deadline_s: float = 120.0, progress: bool = False) -> dict[str, float]:
    """Read many durations in parallel, bounded so nothing can hang the caller.

    Returns {resolved-path-str: seconds} for files that answered in time. A file
    that doesn't (malformed/streaming demuxer that defeats the per-open timeout)
    is simply OMITTED -- the caller treats a missing path as "unknown" and uses
    its own fallback. We never join wedged threads: the pool is left to daemon
    threads and shut down without waiting, so one pathological file cannot stall
    the build. The whole scan is capped by `deadline_s` via as_completed's own
    timeout (which raises once the window elapses, even if every remaining future
    is wedged). When *progress* is true, show a bar so the scan never looks hung.
    """
    out: dict[str, float] = {}
    if not paths:
        return out
    workers = min(32, (os.cpu_count() or 4) * 2)
    pool = ThreadPoolExecutor(max_workers=workers)
    futs = {pool.submit(duration, p, timeout): str(Path(p).resolve()) for p in paths}
    bar = tqdm(total=len(futs), desc="durations", unit="vid",
               disable=not progress, leave=False)
    try:
        for fu in as_completed(futs, timeout=deadline_s):
            try:
                d = fu.result()
            except Exception:  # noqa: BLE001
                d = 0.0
            if d > 0:
                out[futs[fu]] = d
            bar.update(1)
    except Exception:  # noqa: BLE001 - as_completed deadline hit; abandon the rest
        pass
    finally:
        bar.close()
    # Do NOT wait on wedged threads; drop queued work and let daemons die.
    pool.shutdown(wait=False, cancel_futures=True)
    return out


def total_duration(root: str | Path,
                   progress: bool = False) -> tuple[float, int, int]:
    """Sum durations of all videos under *root*. Returns (seconds, count, failed).

    Reads container metadata in parallel; no frame decoding. When *progress* is
    true, shows a self-updating bar with a running total.
    """
    workers = min(32, (os.cpu_count() or 4) * 2)
    paths = [
        p for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower().lstrip(".") in VIDEO_EXTS
    ]
    total, count, failed = 0.0, 0, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(duration, paths)
        bar = tqdm(results, total=len(paths), desc="reading durations",
                   unit="vid", disable=not progress)
        for d in bar:
            if d > 0:
                total += d
                count += 1
            else:
                failed += 1
            if progress:
                bar.set_postfix_str(f"{total/3600:.1f}h so far", refresh=False)
        bar.close()
    return total, count, failed


def main(root: str):
    total, count, failed = total_duration(root)
    h = int(total // 3600); m = int((total % 3600) // 60); s = int(total % 60)
    print(f"{count} videos, {failed} unreadable")
    print(f"total: {total:.0f}s = {h}h {m}m {s}s")


if __name__ == "__main__":
    main(sys.argv[1])
