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


def collection_status(cached: int, total: int) -> None:
    """One-shot collection-level 'how much of the library is cached' line, drawn
    once to stderr before hashing starts. During hashing the live library count
    is carried in the hashing bar's postfix (see WorkerBars), which ticks up as
    each file completes; this line is the starting snapshot. Pure display."""
    try:
        total = max(0, int(total))
        cached = max(0, min(int(cached), total))
        if total == 0:
            return
        pct = cached / total
        width = 24
        filled = int(round(pct * width))
        bar = "█" * filled + "░" * (width - filled)
        to_do = total - cached
        _err(f"      library: |{bar}| {pct*100:3.0f}%  "
             f"{cached}/{total} cached, {to_do} to hash this run")
    except Exception:  # noqa: BLE001 - display must never break the build
        pass


def is_tty() -> bool:
    return sys.stderr.isatty()


_SLOT_CAP = 16  # max per-worker lines drawn, so a huge --workers can't overflow


class WorkerBars:
    """Parent-side progress for parallel hashing: a duration-weighted overall bar
    plus, on a TTY, one status line per worker slot showing the file it's hashing.

    IMPORTANT: this is a pure *observer*. It is driven entirely by the parent
    process from its own bookkeeping; workers never touch it. Every public method
    swallows its own errors so a drawing glitch can NEVER propagate into the build
    (functionality always wins). Only the parent thread calls these -- no locking
    needed.
    """

    def __init__(self, total_units: float, n_files: int, workers: int,
                 enabled: bool | None = None,
                 library_cached: int = 0, library_total: int = 0):
        from tqdm import tqdm

        self.n_files = n_files
        self.done_files = 0
        # Live library progress: how many of the WHOLE collection are cached.
        # Baseline = already-cached before this run; each finished file adds one.
        # Shown in the hashing bar's postfix so it ticks up live without any new
        # bar or any worker-side machinery (parent bookkeeping only).
        self.library_base = max(0, int(library_cached))
        self.library_total = max(0, int(library_total))
        self.enabled = is_tty() if enabled is None else enabled
        self._total = max(1.0, float(total_units))
        self._progress = 0.0
        self.overall = None
        self.slots: list = []
        try:
            self.overall = tqdm(
                total=self._total, position=0, desc="hashing",
                unit_scale=False, leave=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}",
            )
            self._set_postfix()
            if self.enabled:
                for k in range(min(workers, _SLOT_CAP)):
                    s = tqdm(total=0, position=k + 1, bar_format="{desc}",
                             leave=False)
                    s.set_description_str(f"  worker {k}: idle")
                    self.slots.append(s)
        except Exception:  # noqa: BLE001 - display must never break the build
            pass

    def _set_postfix(self) -> None:
        if self.overall is not None:
            msg = f"files {self.done_files}/{self.n_files}"
            if self.library_total > 0:
                lib = min(self.library_base + self.done_files, self.library_total)
                lib_pct = lib / self.library_total * 100
                msg += f" | library {lib}/{self.library_total} ({lib_pct:3.0f}%)"
            self.overall.set_postfix_str(msg, refresh=True)

    def set_slot(self, slot: int, name: str) -> None:
        try:
            if 0 <= slot < len(self.slots):
                self.slots[slot].set_description_str(f"  worker {slot}: {name}")
        except Exception:  # noqa: BLE001
            pass

    def clear_slot(self, slot: int) -> None:
        try:
            if 0 <= slot < len(self.slots):
                self.slots[slot].set_description_str(f"  worker {slot}: idle")
        except Exception:  # noqa: BLE001
            pass

    def advance(self, weight: float) -> None:
        """One file finished: advance the overall bar by its weight, clamped so
        float summation can't exceed 100%."""
        try:
            self.done_files += 1
            step = max(0.0, float(weight))
            if self._progress + step > self._total:
                step = max(0.0, self._total - self._progress)
            self._progress += step
            if self.overall is not None:
                self.overall.update(step)
            self._set_postfix()
        except Exception:  # noqa: BLE001
            pass

    def finish(self) -> None:
        try:
            if self.overall is not None and self._progress < self._total:
                self.overall.update(self._total - self._progress)
                self._progress = self._total
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        for s in self.slots:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            if self.overall is not None:
                self.overall.close()
        except Exception:  # noqa: BLE001
            pass
