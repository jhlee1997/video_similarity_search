"""Small terminal-narration helpers so every command reads the same way:

    a header saying what's starting (and the scale), then a self-updating
    progress bar during the work, then a summary of what completed / was
    skipped / failed.

Progress bars themselves come from tqdm (written to stderr, so they never
corrupt stdout / --json output). These helpers just standardise the framing
text around them. Everything writes to stderr so stdout stays reserved for
actual results (reports, JSON).
"""

from __future__ import annotations

import sys

import click


def _err(msg: str = "") -> None:
    click.echo(msg, err=True)


def hms(seconds: float) -> str:
    """Human h/m/s, e.g. 50h 40m 0s."""
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s"


def elapsed(seconds: float) -> str:
    """Compact elapsed time: '3.4s', '4m 12s', or '1h 4m 12s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    secs = int(round(seconds))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def header(title: str, *, step: tuple[int, int] | None = None,
           detail: str = "") -> None:
    """Announce a phase. Optionally '[1/2]' step framing and a detail line."""
    prefix = f"[{step[0]}/{step[1]}] " if step else ""
    _err(f"{prefix}{title}")
    if detail:
        _err(f"      {detail}")


def summary(pairs: list[tuple[str, object]]) -> None:
    """Print an aligned key/value summary block to stderr."""
    if not pairs:
        return
    width = max(len(k) for k, _ in pairs)
    _err("")
    for k, v in pairs:
        _err(f"  {k.rjust(width)} : {v}")


def note(msg: str) -> None:
    """A single indented status line."""
    _err(f"      {msg}")


def is_tty() -> bool:
    return sys.stderr.isatty()
