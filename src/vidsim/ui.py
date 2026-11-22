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
    """One-shot collection-level 'how much of the collection is cached' line.

    Used only when there is nothing to hash this run (no live WorkerBars gets
    created); when there IS work, WorkerBars draws a live collection bar that
    climbs instead. Pure display."""
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
        _err(f"      collection: |{bar}| {pct*100:3.0f}%  "
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
                 collection_cached: int = 0, collection_total: int = 0):
        from tqdm import tqdm

        self.n_files = n_files
        self.done_files = 0
        # Live collection progress: how many of the WHOLE collection are cached.
        # Baseline = already-cached before this run; each finished file adds one.
        # Drawn as a live bar at the top and redrawn from the parent's own
        # done_files count -- no worker involvement.
        self.collection_base = max(0, int(collection_cached))
        self.collection_total = max(0, int(collection_total))
        self.enabled = is_tty() if enabled is None else enabled
        self._total = max(1.0, float(total_units))
        self._progress = 0.0
        self.collection = None   # live collection-level bar (position 0)
        self.overall = None      # per-run hashing bar (position 1)
        self.slots: list = []
        try:
            # Collection-level line as a LIVE bar at the top: it redraws (climbs)
            # as each file finishes, via _update_collection() from the parent's
            # own done_files count -- no worker involvement. The per-run hashing
            # bar sits below it, then the per-worker slot lines.
            if self.collection_total > 0:
                self.collection = tqdm(total=self.collection_total, position=0,
                                       bar_format="{desc}", leave=True)
                self._update_collection()
            self.overall = tqdm(
                total=self._total, position=1, desc="hashing",
                unit_scale=False, leave=True,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}",
            )
            self._set_postfix()
            if self.enabled:
                base = 2  # positions 0=collection, 1=hashing, workers start at 2
                for k in range(min(workers, _SLOT_CAP)):
                    s = tqdm(total=0, position=k + base, bar_format="{desc}",
                             leave=False)
                    s.set_description_str(f"  worker {k}: idle")
                    self.slots.append(s)
        except Exception:  # noqa: BLE001 - display must never break the build
            pass

    def _update_collection(self) -> None:
        """Redraw the live collection-level line from done_files."""
        if self.collection is None or self.collection_total <= 0:
            return
        try:
            done = min(self.collection_base + self.done_files, self.collection_total)
            pct = done / self.collection_total
            width = 24
            filled = int(round(pct * width))
            bar = "█" * filled + "░" * (width - filled)
            to_do = self.collection_total - done
            self.collection.set_description_str(
                f"collection: |{bar}| {pct*100:3.0f}%  "
                f"{done}/{self.collection_total} cached, {to_do} to hash this run"
            )
        except Exception:  # noqa: BLE001
            pass

    def _set_postfix(self) -> None:
        if self.overall is not None:
            self.overall.set_postfix_str(
                f"files {self.done_files}/{self.n_files}", refresh=True)

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
            self._update_collection()   # climb the live collection bar too
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
        try:
            if self.collection is not None:
                self.collection.close()
        except Exception:  # noqa: BLE001
            pass
