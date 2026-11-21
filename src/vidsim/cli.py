"""vidsim command-line interface.

Commands:
  cache-build <root>   hash every video under <root> into the cache (incremental)
  index                 (re)build the FAISS frame index from the cache
  find <video>          report what the cache already contains like <video>
  pairs                 scan the whole collection for dup/containment pairs

None of these commands ever delete or move a video. Output is a report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .aggregate import compare
from .cache import (
    build_record,
    load_all_records,
    load_record,
    save_record,
    sync,
)
from .config import (
    DEFAULT_HASH_PARAMS,
    DEFAULT_MATCH_PARAMS,
    HashParams,
    MatchParams,
)
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
@click.argument("root", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--seconds-per-hash", default=DEFAULT_HASH_PARAMS.seconds_per_hash,
              show_default=True, help="vpdq sampling interval in seconds (~fps).")
@click.option("--quality-floor", default=DEFAULT_HASH_PARAMS.quality_floor,
              show_default=True, help="Drop frames below this PDQ quality (0-100).")
@click.option("--reindex/--no-reindex", default=True, show_default=True,
              help="Rebuild the FAISS index after syncing.")
@click.option("--index-kind", default="flat", type=click.Choice(["flat", "hnsw"]),
              show_default=True)
def cache_build(root: Path, seconds_per_hash: float, quality_floor: int,
                reindex: bool, index_kind: str) -> None:
    """Hash every video under ROOT into the cache (incremental)."""
    params = _hash_params(seconds_per_hash, quality_floor)
    stats = sync(root, params)
    click.echo(
        f"added={stats.added} updated={stats.updated} unchanged={stats.unchanged} "
        f"removed={stats.removed} failed={stats.failed}"
    )
    if reindex:
        _do_index(index_kind)


def _do_index(kind: str) -> None:
    records = load_all_records()
    idx = FrameIndex.build(records, kind=kind)
    idx.save()
    click.echo(f"index: {len(idx)} frames from {len(records)} videos ({kind})")


@cli.command("index")
@click.option("--index-kind", default="flat", type=click.Choice(["flat", "hnsw"]),
              show_default=True)
def index_cmd(index_kind: str) -> None:
    """(Re)build the FAISS frame index from the current cache."""
    _do_index(index_kind)


@cli.command("find")
@click.argument("video", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--seconds-per-hash", default=DEFAULT_HASH_PARAMS.seconds_per_hash)
@click.option("--quality-floor", default=DEFAULT_HASH_PARAMS.quality_floor)
@click.option("--frame-radius", type=int, default=None,
              help="Per-frame Hamming match radius (bits).")
@click.option("--min-coverage", type=float, default=None,
              help="Only report targets reaching this directional coverage.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of text.")
def find_cmd(video: Path, seconds_per_hash: float, quality_floor: int,
             frame_radius: int | None, min_coverage: float | None,
             as_json: bool) -> None:
    """Report how much of VIDEO is drawn from anything in the collection."""
    idx = FrameIndex.load()
    if idx is None:
        click.echo("No index found. Run `vidsim cache-build <root>` first.", err=True)
        sys.exit(1)

    records = load_all_records()
    params = _hash_params(seconds_per_hash, quality_floor)
    mp = _match_params(frame_hamming_radius=frame_radius, report_coverage=min_coverage)

    fid = file_id(video)

    # Exact byte-identical duplicate? (file_id collision == identical content.)
    if fid in records:
        existing = records[fid].paths
        click.echo(f"EXACT duplicate of cached content: {existing}")

    # Reuse cached hashes if present, else hash on the fly (don't persist here).
    rec = load_record(fid)
    if rec is None or rec.is_stale(params):
        rec = build_record(video, fid, params)

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
@click.option("--seconds-per-hash", default=DEFAULT_HASH_PARAMS.seconds_per_hash)
@click.option("--quality-floor", default=DEFAULT_HASH_PARAMS.quality_floor)
@click.option("--frame-radius", type=int, default=None)
@click.option("--min-shared-frames", type=int, default=None,
              help="Min candidate frame matches before scoring a pair.")
@click.option("--min-coverage", type=float, default=None,
              help="Only report a pair if a direction reaches this coverage.")
@click.option("--json", "as_json", is_flag=True)
def pairs_cmd(seconds_per_hash: float, quality_floor: int, frame_radius: int | None,
              min_shared_frames: int | None, min_coverage: float | None,
              as_json: bool) -> None:
    """Scan the whole collection; report how much each video draws from another.

    Directional: a pair may appear twice (B drawn from A, and A drawn from B)
    when both directions reach the coverage threshold -- e.g. duplicates show
    ~100% both ways, a subset shows high one way and low the other.
    """
    idx = FrameIndex.load()
    if idx is None:
        click.echo("No index found. Run `vidsim cache-build <root>` first.", err=True)
        sys.exit(1)

    records = load_all_records()
    mp = _match_params(
        frame_hamming_radius=frame_radius,
        min_shared_frames=min_shared_frames,
        report_coverage=min_coverage,
    )

    collected = []
    for fid, rec in records.items():
        for r in compare(rec, idx, mp, records):
            if r.coverage >= mp.report_coverage:
                collected.append(r)

    collected.sort(key=lambda r: r.coverage, reverse=True)

    if as_json:
        click.echo(dump_json(collected, records))
        return
    if not collected:
        click.echo("No shared-content pairs found.")
        return
    for r in collected:
        click.echo(format_result_line(r, records))
        click.echo("")


def main() -> None:
    cli()
