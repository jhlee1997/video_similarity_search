"""vidsim command-line interface.

Commands (COLLECTION is the folder you ran cache-build on):
  cache-build COLLECTION        hash every video under it into the cache
  index COLLECTION              (re)build the FAISS frame index from the cache
  find VIDEO --in COLLECTION    report how much of VIDEO the collection contains
  pairs COLLECTION              scan the whole collection for dup/subset pairs

None of these commands ever delete or move a video. Output is a report.
The cache lives WITH the videos (per-folder .vidsim/); only cache-build writes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from tqdm import tqdm

from .aggregate import compare
from .cache import (
    build_record,
    find_collection_root,
    load_records,
    resolve_collection_root,
    sync,
)
from .config import (
    DEFAULT_HASH_PARAMS,
    DEFAULT_MATCH_PARAMS,
    HashParams,
    MatchParams,
)
from . import ui
from .fileid import file_id
from .index import FrameIndex
from .report import dump_json, format_result_line


def _hash_params(seconds_per_hash: float, quality_floor: int) -> HashParams:
    return HashParams(seconds_per_hash=seconds_per_hash, quality_floor=quality_floor)


def _match_params(**overrides) -> MatchParams:
    base = DEFAULT_MATCH_PARAMS.__dict__.copy()
    base.update({k: v for k, v in overrides.items() if v is not None})
    return MatchParams(**base)


@click.group()
def cli() -> None:
    """Fast duplicate & containment (subset) detection for a video hoard."""


@cli.command("cache-build")
@click.argument("collection", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--seconds-per-hash", default=DEFAULT_HASH_PARAMS.seconds_per_hash,
              show_default=True, help="vpdq sampling interval in seconds (~fps).")
@click.option("--quality-floor", default=DEFAULT_HASH_PARAMS.quality_floor,
              show_default=True, help="Drop frames below this PDQ quality (0-100).")
@click.option("--reindex/--no-reindex", default=True, show_default=True,
              help="Rebuild the FAISS index after syncing.")
@click.option("--index-kind", default="flat", type=click.Choice(["flat", "hnsw"]),
              show_default=True)
@click.option("--workers", type=int, default=None,
              help="Parallel hashing workers (default: CPU count - 3).")
def cache_build(collection: Path, seconds_per_hash: float, quality_floor: int,
                reindex: bool, index_kind: str, workers: int | None) -> None:
    """Hash every video under COLLECTION into the per-folder cache (incremental)."""
    root = resolve_collection_root(collection)
    params = _hash_params(seconds_per_hash, quality_floor)

    detail = f"collection: {root}"
    if root != collection.resolve():
        detail += "  (attached to existing collection)"
    ui.header("Building cache", step=(1, 2), detail=detail)
    t0 = time.perf_counter()
    stats = sync(root, params, workers=workers)
    dt = time.perf_counter() - t0
    hashed = stats.added + stats.updated
    rows = [
        ("hashed (new)", stats.added),
        ("rehashed", stats.updated),
        ("unchanged", stats.unchanged),
        ("duplicate copies", stats.duplicate_copies),
        ("skipped (oversized)", stats.skipped_oversized),
        ("removed", stats.removed),
        ("failed", stats.failed),
        ("elapsed", ui.elapsed(dt)),
    ]
    if hashed and dt > 0:
        rows.append(("throughput", f"{hashed / dt * 60:.0f} videos/min"))
    ui.summary(rows)

    if reindex:
        ui.header("Building index", step=(2, 2))
        _do_index(root, index_kind)
    else:
        ui.header("Index", step=(2, 2), detail="skipped (--no-reindex)")


def _do_index(root: Path, kind: str) -> None:
    records = load_records(root)
    idx = FrameIndex.build(records, kind=kind)
    idx.save(root)
    ui.summary([
        ("index type", kind),
        ("videos", len(records)),
        ("frames", len(idx)),
    ])


def _resolve_for_read(collection: Path) -> Path:
    """Locate the collection root for a read command, or exit with guidance.

    Accepts a subfolder of a collection and walks up to the folder that actually
    holds the index, so reads behave consistently with cache-build.
    """
    root = find_collection_root(collection)
    if root is None:
        click.echo(
            f"No vidsim index found at or above {collection}. "
            f"Run `vidsim cache-build` on the collection first.",
            err=True,
        )
        sys.exit(1)
    return root


@cli.command("index")
@click.argument("collection", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--index-kind", default="flat", type=click.Choice(["flat", "hnsw"]),
              show_default=True)
def index_cmd(collection: Path, index_kind: str) -> None:
    """(Re)build the FAISS frame index for COLLECTION from its cache."""
    # Attach to an existing collection root if COLLECTION is a subfolder, so we
    # never fragment one collection into two divergent indexes.
    root = resolve_collection_root(collection)
    ui.header("Building index", detail=f"collection: {root}")
    _do_index(root, index_kind)


@cli.command("find")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--in", "collection", required=True,
              type=click.Path(exists=True, file_okay=False, path_type=Path),
              help="Collection root to search (the folder you ran cache-build on).")
@click.option("--seconds-per-hash", default=DEFAULT_HASH_PARAMS.seconds_per_hash)
@click.option("--quality-floor", default=DEFAULT_HASH_PARAMS.quality_floor)
@click.option("--frame-radius", type=int, default=None,
              help="Per-frame Hamming match radius (bits).")
@click.option("--min-coverage", type=float, default=None,
              help="Only report targets reaching this directional coverage.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of text.")
def find_cmd(video: Path, collection: Path, seconds_per_hash: float,
             quality_floor: int, frame_radius: int | None,
             min_coverage: float | None, as_json: bool) -> None:
    """Report how much of VIDEO is drawn from anything in the collection."""
    root = _resolve_for_read(collection)
    idx = FrameIndex.load(root)
    if idx is None:
        click.echo(f"No index at {root}. Run `vidsim cache-build` first.", err=True)
        sys.exit(1)

    records = load_records(root)
    params = _hash_params(seconds_per_hash, quality_floor)
    mp = _match_params(frame_hamming_radius=frame_radius, report_coverage=min_coverage)

    fid = file_id(video)

    # Exact byte-identical duplicate? (file_id collision == identical content.)
    if fid in records:
        existing = records[fid].paths
        click.echo(f"EXACT duplicate of cached content: {existing}")

    # Reuse the cached record if the collection already has this content;
    # otherwise hash the query on the fly (don't persist -- find is a read).
    rec = records.get(fid)
    if rec is None or rec.is_stale(params):
        if not as_json:
            ui.header("Checking video against collection",
                      detail=f"hashing {video.name} ({len(records)} videos in collection)")
        rec = build_record(video, fid, params)
    elif not as_json:
        ui.header("Checking video against collection",
                  detail=f"{video.name} (cached) vs {len(records)} videos")

    results = [r for r in compare(rec, idx, mp, records)
               if r.coverage >= mp.report_coverage]
    if as_json:
        click.echo(dump_json(results, records))
        return
    if not results:
        click.echo("No similar videos found.")
        return
    for r in results:
        click.echo(format_result_line(r, records))
        click.echo("")


@cli.command("pairs")
@click.argument("collection", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--frame-radius", type=int, default=None)
@click.option("--min-shared-frames", type=int, default=None,
              help="Min candidate frame matches before scoring a pair.")
@click.option("--min-coverage", type=float, default=None,
              help="Only SHOW a pair if a direction reaches this coverage "
                   "(display filter; the cache stores all pairs).")
@click.option("--refresh", is_flag=True,
              help="Ignore the cache: recompute all pairs from scratch and "
                   "overwrite the cache.")
@click.option("--no-cache", is_flag=True,
              help="Compute fresh without reading or writing the pairs cache.")
@click.option("--show-ignored", is_flag=True,
              help="Also show pairs you've marked ignored (lets you un-ignore "
                   "them at the prompt).")
@click.option("--json", "as_json", is_flag=True)
def pairs_cmd(collection: Path, frame_radius: int | None,
              min_shared_frames: int | None, min_coverage: float | None,
              refresh: bool, no_cache: bool, show_ignored: bool,
              as_json: bool) -> None:
    """Scan COLLECTION; report how much each video draws from another.

    Directional: a pair may appear twice (B drawn from A, and A drawn from B)
    when both directions reach the coverage threshold -- e.g. duplicates show
    ~100% both ways, a subset shows high one way and low the other.

    Results are cached in the collection's .vidsim/pairs.json and updated
    INCREMENTALLY: a rerun recomputes only videos added or removed since the
    last run. --min-coverage is a display filter over the cached pairs (never
    forces a recompute). If you change --frame-radius / --min-shared-frames or
    the matching code, delete .vidsim/pairs.json (or use --refresh) so stale
    pairs aren't reused.
    """
    from . import pairs_cache, pairs_ignore

    root = _resolve_for_read(collection)
    idx = FrameIndex.load(root)
    if idx is None:
        click.echo(f"No index at {root}. Run `vidsim cache-build` first.", err=True)
        sys.exit(1)

    records = load_records(root)
    mp = _match_params(
        frame_hamming_radius=frame_radius,
        min_shared_frames=min_shared_frames,
        report_coverage=min_coverage,
    )

    if not as_json:
        ui.header("Scanning for related pairs",
                  detail=f"{len(records)} videos in {root}")

    t0 = time.perf_counter()
    current = set(records)

    cached = None if (refresh or no_cache) else pairs_cache.load(root)
    if cached is not None:
        cached_fids, cached_results = cached
        all_pairs, added, removed = pairs_cache.incremental_update(
            cached_fids, cached_results, records, idx, mp, progress=not as_json)
        cache_status = f"incremental: +{len(added)} / -{len(removed)} video(s)"
    else:
        all_pairs = pairs_cache.full_scan(records, idx, mp, progress=not as_json)
        cache_status = "full scan (no cache used)"

    # Persist the full (unfiltered) pair set unless caching is disabled.
    if not no_cache:
        try:
            pairs_cache.save(root, current, all_pairs)
        except Exception:  # noqa: BLE001 - a cache write must NEVER fail the read
            pass

    dt = time.perf_counter() - t0

    # --min-coverage is a pure DISPLAY filter over the cached superset.
    collected = [r for r in all_pairs if r.coverage >= mp.report_coverage]
    collected.sort(key=lambda r: r.coverage, reverse=True)

    if as_json:
        # Machine output: complete + predictable. The ignore list is a human
        # viewing convenience only, so it is NOT applied here (and no prompt).
        click.echo(dump_json(collected, records))
        return

    # Ignore list (human report only): pairs the user reviewed and hid. Prune
    # entries whose video is gone, and write that back now so stale entries don't
    # accumulate even if the user never interacts at the prompt.
    loaded_ignored = pairs_ignore.load(root)
    ignored = pairs_ignore.prune(loaded_ignored, current)
    if ignored != loaded_ignored:
        try:
            pairs_ignore.save(root, ignored)
        except Exception:  # noqa: BLE001 - housekeeping must never break the report
            pass

    # Partition by ignore state. By default ignored pairs are hidden; with
    # --show-ignored they're shown too (so they can be un-ignored at the prompt).
    n_ignored = sum(1 for r in collected
                    if pairs_ignore.is_ignored(ignored, r.query_id, r.target_id))
    shown = collected if show_ignored else [
        r for r in collected
        if not pairs_ignore.is_ignored(ignored, r.query_id, r.target_id)]
    n_hidden = 0 if show_ignored else n_ignored

    ui.summary([("videos scanned", len(records)),
                ("related pairs", len(collected)),
                ("hidden (ignored)", n_hidden),
                ("pairs cache", cache_status),
                ("elapsed", ui.elapsed(dt))])
    click.echo(err=True)  # clean blank line before the report on stdout

    if not shown:
        click.echo("No shared-content pairs found."
                   if not collected else "All matching pairs are ignored "
                   "(use --show-ignored to see them).")
        return

    # Label each shown pair so the user can refer to it at the prompt.
    labels = _pair_labels(len(shown))
    for label, r in zip(labels, shown):
        ign = pairs_ignore.is_ignored(ignored, r.query_id, r.target_id)
        tag = "  [ignored]" if (show_ignored and ign) else ""
        click.echo(f"[{label}]{tag}")
        click.echo(format_result_line(r, records))
        click.echo("")

    # Interactive prompt: only on a real terminal (never when piped), so scripts
    # can't hang. Lets the user ignore (or, under --show-ignored, un-ignore).
    _prompt_ignore(root, shown, labels, ignored, show_ignored)


def _pair_labels(n: int) -> list[str]:
    """A, B, ..., Z, AA, AB, ... enough distinct labels for n pairs."""
    import string
    letters = string.ascii_uppercase
    out, i = [], 0
    while len(out) < n:
        q, r = divmod(i, len(letters))
        out.append((letters[q - 1] if q else "") + letters[r])
        i += 1
    return out


def _prompt_ignore(root, shown, labels, ignored, show_ignored) -> None:
    """Ask which labelled pairs to ignore (or un-ignore). Interactive-only; writes
    the ignore list if it changed. Any error here must never break the report."""
    from . import pairs_ignore

    # Only prompt when BOTH input and output are a real terminal. Requiring
    # stdout to be a TTY too means `vidsim pairs > out.txt` (report redirected to
    # a file) does not fire a prompt for pairs the user can't see on screen.
    if not (sys.stdin.isatty() and sys.stdout.isatty() and sys.stderr.isatty()):
        return
    verb = "un-ignore" if show_ignored else "ignore"
    try:
        ans = click.prompt(
            f"\n{verb.capitalize()} any of these going forward? "
            f"Enter labels (e.g. A,C) or press Enter to skip",
            default="", show_default=False, err=True,
        ).strip()
    except (EOFError, click.Abort):
        return
    if not ans:
        return

    by_label = dict(zip(labels, shown))
    tokens = [t for t in
              (tok.strip().upper() for tok in ans.replace(",", " ").split()) if t]
    picked = [by_label[t] for t in tokens if t in by_label]
    unknown = [t for t in tokens if t not in by_label]
    if unknown:
        click.echo(f"Ignoring unrecognised label(s): {', '.join(unknown)}", err=True)
    if not picked:
        click.echo("No matching labels; nothing changed.", err=True)
        return

    # Count DISTINCT unordered pairs changed (two directional rows of the same
    # pair collapse to one canonical key), so the message doesn't overstate.
    changed: set = set()
    for r in picked:
        if show_ignored:
            if pairs_ignore.is_ignored(ignored, r.query_id, r.target_id):
                pairs_ignore.remove(ignored, r.query_id, r.target_id)
                changed.add(tuple(sorted((r.query_id, r.target_id))))
        else:
            if not pairs_ignore.is_ignored(ignored, r.query_id, r.target_id):
                pairs_ignore.add(ignored, r.query_id, r.target_id)
                changed.add(tuple(sorted((r.query_id, r.target_id))))
    if not changed:
        click.echo("Nothing changed (already in that state).", err=True)
        return
    try:
        pairs_ignore.save(root, ignored)
        click.echo(f"{verb.capitalize()}d {len(changed)} pair(s).", err=True)
    except Exception:  # noqa: BLE001 - must never break the report
        click.echo("(could not write the ignore list)", err=True)


@cli.command("duration")
@click.argument("collection", type=click.Path(exists=True, file_okay=False, path_type=Path))
def duration_cmd(collection: Path) -> None:
    """Sum the total video duration under COLLECTION (container metadata only).

    Useful for estimating cache-build runtime, which scales with total duration.
    """
    from .vid_length import total_duration

    ui.header("Summing video durations", detail=f"collection: {collection}")
    t0 = time.perf_counter()
    total, count, failed = total_duration(collection, progress=True)
    dt = time.perf_counter() - t0
    ui.summary([
        ("videos", count),
        ("unreadable", failed),
        ("total duration", ui.hms(total)),
        ("elapsed", ui.elapsed(dt)),
    ])
    # The machine-usable result goes to stdout so it can be piped/captured.
    click.echo(f"{total:.0f}")


def main() -> None:
    cli()
