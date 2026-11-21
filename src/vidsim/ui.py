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


class WorkerBars:
    """Live progress for parallel hashing: a duration-weighted overall bar plus,
    on a TTY, one status line per worker slot showing the file it's hashing now.

    The overall bar is weighted by video DURATION (not file count), so a long
    video owns a proportional share of the bar -- finishing many short clips does
    not jump the bar near 100% while a 2h file is still hashing.

    All drawing happens in the parent process (the caller). Workers must only
    send events; never draw from a worker. On a non-TTY (pipe/redirect/--json)
    the per-slot lines are suppressed and only the overall bar is shown.
    """

    def __init__(self, total_units: float, n_files: int, workers: int,
                 enabled: bool | None = None, max_slots: int = 12):
        import threading

        from tqdm import tqdm  # local import keeps ui import cheap

        self._tqdm = tqdm
        self.n_files = n_files
        self.done_files = 0
        self.enabled = is_tty() if enabled is None else enabled
        # The overall bar is advanced from the main thread while per-slot lines
        # are updated from a background drain thread; serialise all bar writes.
        self._lock = threading.Lock()
        # Cap visible per-worker lines so a high --workers on a short terminal
        # can't scroll/garble the display (slots beyond the cap just aren't drawn).
        self._n_slots = min(workers, max_slots)

        # Overall bar: unit is seconds of video (duration-weighted). Guard against
        # a zero total (e.g. all-unreadable durations) so tqdm still renders.
        self._total = max(1.0, float(total_units))
        self._progress = 0.0
        self.overall = tqdm(
            total=self._total, position=0, desc="hashing", unit="s",
            unit_scale=False, leave=True,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}",
        )
        self._set_overall_postfix()

        # Per-worker status lines (TTY only), capped at _n_slots lines.
        self.slots: list = []
        if self.enabled:
            for k in range(self._n_slots):
                self.slots.append(tqdm(
                    total=0, position=k + 1, bar_format="{desc}", leave=False,
                ))
                self.slots[-1].set_description_str(f"  worker {k}: idle")

    def _set_overall_postfix(self) -> None:
        self.overall.set_postfix_str(f"files {self.done_files}/{self.n_files}",
                                     refresh=True)

    def on_start(self, slot: int, name: str) -> None:
        # Called from the drain thread. Slots beyond the display cap are ignored.
        with self._lock:
            if self.enabled and 0 <= slot < len(self.slots):
                self.slots[slot].set_description_str(f"  worker {slot}: {name}")

    def on_done(self, slot: int) -> None:
        with self._lock:
            if self.enabled and 0 <= slot < len(self.slots):
                self.slots[slot].set_description_str(f"  worker {slot}: idle")

    def advance(self, dur: float) -> None:
        """One video finished: advance the overall bar by its duration, clamped so
        floating-point summation can't push it past 100%. Called from the main
        thread; the lock guards against the drain thread touching tqdm at once."""
        with self._lock:
            self.done_files += 1
            step = max(0.0, float(dur))
            if self._progress + step > self._total:
                step = max(0.0, self._total - self._progress)
            self._progress += step
            self.overall.update(step)
            self._set_overall_postfix()

    def finish(self) -> None:
        """Snap the overall bar to exactly 100% at the end (per-step clamping can
        leave a sub-unit shortfall from float summation)."""
        with self._lock:
            if self._progress < self._total:
                self.overall.update(self._total - self._progress)
                self._progress = self._total

    def idle_all(self) -> None:
        """Reset every slot line to idle (used when the pool is torn down and
        workers were SIGKILLed, so their 'done' events never arrive)."""
        with self._lock:
            for k, s in enumerate(self.slots):
                s.set_description_str(f"  worker {k}: idle")

    def close(self) -> None:
        with self._lock:
            for s in self.slots:
                try:
                    s.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.overall.close()
            except Exception:  # noqa: BLE001
                pass
