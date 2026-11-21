import subprocess, sys
from pathlib import Path

VIDEO_EXTS = {"mp4","mkv","webm","mov","avi","m4v","flv","wmv",
              "mpg","mpeg","ts","m2ts","3gp","ogv"}

def duration(path: Path) -> float:
    """Seconds, read from container metadata (no decoding)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0  # unreadable / no duration metadata

def main(root: str):
    total, count, failed = 0.0, 0, 0
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower().lstrip(".") in VIDEO_EXTS:
            d = duration(p)
            if d > 0:
                total += d; count += 1
            else:
                failed += 1
    h = int(total // 3600); m = int((total % 3600) // 60); s = int(total % 60)
    print(f"{count} videos, {failed} unreadable")
    print(f"total: {total:.0f}s = {h}h {m}m {s}s")

if __name__ == "__main__":
    main(sys.argv[1])
