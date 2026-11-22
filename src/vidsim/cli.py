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
              help="Only report a pair if a direction reaches this coverage.")
@click.option("--json", "as_json", is_flag=True)
def pairs_cmd(collection: Path, frame_radius: int | None,
              min_shared_frames: int | None, min_coverage: float | None,
              as_json: bool) -> None:
    """Scan COLLECTION; report how much each video draws from another.

    Directional: a pair may appear twice (B drawn from A, and A drawn from B)
    when both directions reach the coverage threshold -- e.g. duplicates show
    ~100% both ways, a subset shows high one way and low the other.
    """
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

    collected = []
    # Self-updating bar on stderr (suppressed for --json so machine output stays
    # clean). Live postfix shows how many related pairs found so far.
    t0 = time.perf_counter()
    bar = tqdm(records.items(), total=len(records), desc="comparing",
               unit="vid", disable=as_json)
    for fid, rec in bar:
        for r in compare(rec, idx, mp, records):
            if r.coverage >= mp.report_coverage:
                collected.append(r)
        bar.set_postfix_str(f"{len(collected)} pairs", refresh=False)
    bar.close()
    dt = time.perf_counter() - t0

    collected.sort(key=lambda r: r.coverage, reverse=True)
    if not as_json:
        ui.summary([("videos scanned", len(records)),
                    ("related pairs", len(collected)),
                    ("elapsed", ui.elapsed(dt))])
        click.echo(err=True)  # clean blank line before the report on stdout

    if as_json:
        click.echo(dump_json(collected, records))
        return
    if not collected:
        click.echo("No shared-content pairs found.")
        return
    for r in collected:
        click.echo(format_result_line(r, records))
        click.echo("")


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
