# vidsim

Fast duplicate and **containment** detection for a large personal video hoard.

`vidsim` finds videos in your collection that are the same, or where one is cut
or edited out of another — a 30-second clip lifted from a 2-hour stream, a
re-encoded re-upload, or a highlights reel stitched together from a longer
source. It is built to scale to thousands of videos, including hours-long ones,
by caching cheap per-frame fingerprints and only doing expensive work on
promising candidates.

**It never deletes or moves anything.** Every command produces a report. You
decide what to do with it.

---

## What it answers

For any two videos A and B, vidsim reports a directional number:

> **"N% of B's runtime is drawn from A"**, with every shared segment timestamped
> in both videos.

There are deliberately **no fixed categories**. Everything you care about is read
off the two directional numbers for a pair:

| B drawn from A | A drawn from B | Interpretation                              |
|----------------|----------------|---------------------------------------------|
| ~100%          | ~100%          | the same whole video (a duplicate)          |
| ~100%          | small          | B is a clip / subset of A                   |
| high, many segments | small     | B is an edit/highlights built from A        |
| small          | small          | they share an incidental snippet            |

Because it's directional, a clip and its source are **not** symmetric: the clip
is ~100% drawn from the source, but the source is only partly drawn from the
clip. That asymmetry is exactly what tells a subset apart from a true duplicate.

---

## How it works

A video is reduced to one **PDQ perceptual hash per sampled frame** (~1 frame per
second by default, via [`vpdq`](https://pypi.org/project/vpdq/)). PDQ describes a
frame *as an image* — it survives re-encoding, resizing and recompression, but it
knows nothing about time or which video a frame came from.

The pipeline:

```
videos
  └─► per-frame PDQ hashes            (cached on disk, keyed by file content)
        └─► FAISS binary index         (all frames; fast Hamming-distance search)
              └─► candidate frame matches  "B@12s looks like A@48s"
                    └─► temporal Hough voting   (the core; see below)
                          └─► "N% of B is from A" + timestamped segments
```

### The core idea: temporal Hough voting

A single frame match is noise — black frames, logos, intros and near-identical
scenes collide across unrelated videos. The signal is in *time*.

If B genuinely reuses a chunk of A, every true match in that chunk shares a
**constant time offset**. A clip cut starting at 0:48 of A gives:

```
B 0:00 ↔ A 0:48     offset 48
B 0:01 ↔ A 0:49     offset 48
B 0:02 ↔ A 0:50     offset 48   ← all the same offset
```

So each frame match casts a **vote** for its offset `δ = A_time − B_time`. Votes
pile into a histogram; a real shared segment is a tall peak (many matches agree on
one offset), while coincidental matches scatter thinly across offsets. Every tall
peak is recovered as a segment, so a **reordered highlights reel** (goal, then
red card, then penalty — out of order) still produces several clean peaks, one per
reused clip. Bursts of near-identical frames are down-weighted so near-static
scenes can't stuff a single bin.

This is the standard *video copy localization* technique; see the references at
the bottom.

### Content-keyed, per-folder cache

Hashes are cached per video, keyed by a **content fingerprint** (file size + first
and last 1 MiB), not by path. Renaming or reorganising your library does **not**
trigger recomputation, and two files with the same fingerprint are byte-identical
— so exact duplicates are caught for free before any perceptual work.

The cache **lives with your videos**, so it is portable — move a folder and its
hashes travel with it. Each folder that contains videos gets a hidden `.vidsim/`
holding one record per video in it; the **collection root** (the folder you point
`cache-build` at) additionally holds the global search index:

```
A/                            ← the collection: the folder you run cache-build on
  .vidsim/
    index.faiss               ← index over ALL frames under A (enables cross-folder matching)
    rowmap.npz                ← maps index rows back to (video, frame, timestamp)
    collection.json           ← schema/params fingerprint + which folder holds each record
  B/
    .vidsim/<file_id>.json    ← hash records for the videos in B
    D.mp4  E.mp4
  C/
    .vidsim/<file_id>.json
```

A **collection** is simply the tree under the folder you run `cache-build` on —
everything in it is compared against everything else (that's what the single,
root-level index is for; it's what lets a clip in `B` be matched to its source in
`C`). A byte-identical copy that appears in several folders is stored once, in the
first folder seen, with all its paths recorded.

---

## ⚠️ Known limitation: static-background / low-motion videos

PDQ hashes a **whole frame as an image**. When most of the frame is an unchanging
background and only a small region moves (a mostly-still phone video, or a fixed
camera with people moving inside), two things break:

- **Within a video**, nearly every frame hashes almost identically — in testing,
  two of four sample phone videos had 98–100% of their own frames mutually
  near-identical. A clip then "aligns" at essentially every time offset, so the
  reported *segment timings become meaningless* (a high coverage number still
  tells you the videos share content, but *where* is unreliable).
- **Across videos**, two genuinely different clips shot against the *same*
  background (same room, same angle) can be reported as matches, because the
  shared backdrop dominates the hash — a potential **false positive**.

This is the inherent weak spot of whole-frame perceptual hashing. The
industry-grade fix is to replace the per-frame **PDQ hash** with a learned
**self-supervised embedding** (e.g. Meta's SSCD, the basis of the VSC'22 Video
Similarity Challenge pipeline), which is trained to weight discriminative
content over background. That is a heavier backend (PyTorch + a model, float
indexes, ~64× the cache size, GPU-preferred), so it is **not** implemented here;
vidsim deliberately stays a fast, CPU-friendly PDQ tool.

> Planned lighter mitigations on the PDQ path: per-video static-ness detection
> (flag "near-static — timing unreliable" instead of inventing segments), and
> burstiness down-weighting. Not yet implemented.

---

## Requirements

- **Python ≥ 3.12** and [`uv`](https://docs.astral.sh/uv/).
- **FFmpeg** — required at runtime for frame decoding (`vpdq`) and reading
  container metadata (`av` / PyAV).
- **pkg-config + FFmpeg dev libraries** — required at *install* time, because
  `vpdq` compiles from source.

On macOS:

```sh
brew install ffmpeg pkg-config
```

(On Linux: install `ffmpeg`, `pkg-config`, and your distro's FFmpeg `-dev`
packages. `vpdq` does not support Windows.)

## Install

```sh
uv sync
```

This creates the environment and builds `vpdq` and `faiss-cpu`.

> **PyAV note (macOS):** PyAV's prebuilt wheel bundles its own FFmpeg, which on
> macOS collides with Homebrew's FFmpeg (loaded via `vpdq`) and prints harmless
> `objc[...] Class AVF… implemented in both` warnings. To silence them, build
> PyAV against the system FFmpeg so only one copy loads:
> `uv pip install --no-binary av --force-reinstall av`. The warnings are cosmetic
> for this tool (it only reads metadata, never captures from a camera).

---

## Usage

`cache-build` is the **only** command that reads video files or writes the cache.
`find` and `pairs` are pure reads against the index it builds — so **run
`cache-build` first**, and re-run it whenever the collection changes.

### 1. Build the cache + index over your collection

Walks the collection recursively, hashes every video, and builds the search
index. Incremental: re-running only hashes new or changed files, drops records for
files that disappeared, and otherwise reuses what's cached.

```sh
uv run vidsim cache-build /path/to/your/videos
```

Hashing of new/changed videos runs **in parallel** across worker processes
(default: ~3/4 of your cores), so a first build of a large collection is several
times faster than serial. Re-runs are near-instant: an unchanged file is
recognised by its path + size + mtime and skipped without re-reading it. Use
`--workers N` to raise this for an idle machine or lower it to reduce load.

(Pointing it at a subfolder of an existing collection attaches to that collection
rather than starting a new one.)

> **Oversized files are skipped automatically.** A file that is **4K AND longer
> than 20 min AND larger than 3 GB** is excluded from hashing — it is never
> decoded, gets no cache record, and so never appears in the index or in
> `find` / `pairs`. Such files (typically 4K60 phone/camera footage) cost
> *many* times more to hash per second of runtime than ordinary video — a
> single one can occupy a worker for hours — so they are left out to keep a
> build tractable. **The video file itself is never touched**, and if it was
> hashed by an earlier run, that stale cache record is dropped on the next
> `cache-build`. The run summary reports how many were skipped
> (`skipped (oversized): N`). The thresholds live in `config.py`
> (`SKIP_MIN_WIDTH/HEIGHT/DURATION_S/SIZE_BYTES`) if you need to change them.

### 2. Check one video against the collection

Run this on anything you're about to save, to see whether — and how much of it —
you already have. The query video is the argument; `--in` names the collection.

```sh
uv run vidsim find /path/to/new_video.mp4 --in /path/to/your/videos
```

Example output:

```
100% of  new_video.mp4
    is drawn from  archive/full_stream.mp4
    (1 segment(s); B=0:31 A=2:14:00)
      B 0:00–0:31  ⇐  A 1:42:10–1:42:41  (31f)
```

→ *new_video.mp4 is a 31-second clip taken from 1:42:10 of full_stream.mp4.*

### 3. Scan the whole collection for related pairs

```sh
uv run vidsim pairs /path/to/your/videos
```

Reports every pair where one video draws meaningfully from another, strongest
first. Duplicates show ~100% in both directions; subsets show high one way only.
Each side of a pair is annotated with its **duration and file size**, so you can
decide which copy to keep at a glance (usually the longer / larger one):

```
96% of  B.mp4
    is drawn from  A.mp4
    (12 segment(s); B=0:27 12.9 MB  A=2:14:00 8.9 GB)
```

**Results are cached and updated incrementally.** The first `pairs` run on a
collection does the full scan and writes `.vidsim/pairs.json`. Subsequent runs
recompute **only the videos added or removed** since the last run (everything
else is reused), so re-running `pairs` after adding a handful of videos is nearly
instant instead of a full re-scan.

> **Run `cache-build` first after any add/remove.** `pairs` is a pure read: it
> detects added/removed videos by diffing the records that `cache-build` writes,
> not by scanning your folders. So whenever you add, remove, or replace videos,
> run `cache-build` **before** `pairs` (and `find`) — otherwise `pairs` won't see
> new videos and will still report removed ones. Both steps are incremental, so
> the add/remove → `cache-build` → `pairs` sequence stays fast.

- `--min-coverage` is a **display filter** over the cached pairs — the cache
  stores every pair, so changing the threshold (e.g. `--min-coverage 0.5` to see
  only strong matches) never triggers a recompute.
- `--refresh` forces a full recompute and overwrites the cache.
- `--no-cache` computes fresh without reading or writing the cache.
- The cache is **not** versioned against the matching settings. If you change
  `--frame-radius` or `--min-shared-frames`, or update the matching code, **delete
  `.vidsim/pairs.json`** (or use `--refresh`) so stale pairs aren't reused.

> **What the cache stores.** It keeps the directional result for every pair that
> shares content (above the internal `min-shared-frames` floor); a pair with no
> shared frames simply has no entry (which reads as 0% — correct). `--min-coverage`
> filters what's *shown* from this stored set. `--refresh` recomputes the whole
> set from scratch (same matching as an incremental run — use it if you ever
> suspect the cache drifted, e.g. after changing matching settings or the code).

### Rebuild just the index

If you change the index type without re-hashing:

```sh
uv run vidsim index /path/to/your/videos --index-kind hnsw
```

### Estimating cache-build runtime

`cache-build`'s cost is dominated by **frame decoding**. At the default 1
frame/second only ~one hash per second is kept — but `vpdq` still **decodes the
whole stream** to produce those samples (it does not seek), so the real cost
scales with *frames decoded*, i.e. **duration × resolution × frame-rate**, not
duration alone. A 4K60 file therefore costs far more per second of runtime than
an ordinary 1080p30 one — in testing a single 4K60 file ran at only ~1.7×
real-time on a 12-core CPU, versus many× faster for normal clips. (This is also
why such files are [skipped by default](#1-build-the-cache--index-over-your-collection),
and why neither GPU decode nor downsampling helps much — the cost is in decoding
every frame to reach the samples.)

So **total duration is only a rough predictor of build time, and underestimates
badly for high-resolution/high-fps footage.** Sum it quickly (metadata only, no
decoding):

```sh
uv run vidsim duration /path/to/your/videos
```

It prints a summary to stderr and the raw total seconds to stdout (so the number
is easy to capture, e.g. `uv run vidsim duration /videos 2>/dev/null`):

```
            videos : 1000
        unreadable : 2
    total duration : 50h 40m 0s
           elapsed : 3.1s
182400
```

To turn that into a rough time estimate, measure your machine's throughput once
— run `cache-build` on a small sample and divide its wall-clock time by that
sample's total duration to get a *seconds-of-build per second-of-video* factor,
then multiply by the figure above. Because decode cost depends on
resolution/fps, use a sample whose resolution mix resembles the whole
collection, or expect the estimate to run long on 4K-heavy libraries. Raising
`--seconds-per-hash` keeps *fewer* samples but still decodes the full stream, so
it gives only a modest speedup — not a proportional one.

### Progress & output streams

Long-running commands (`cache-build`, `pairs`, `duration`) narrate their work.
For `cache-build` specifically, while hashing you see, on a terminal:

- a **library line** — how much of the whole collection is cached across runs,
  e.g. `library: |███████░░░| 70%  700/1000 cached, 300 to hash this run`;
- a **per-worker line for each parallel worker**, naming the file it is hashing
  right now (so you can see real concurrent progress, and a long pause reads as
  "worker 2 is on the big one", not a hang);
- an **overall hashing bar weighted by video duration** (not file count), so a
  2-hour video claims a proportionally large share instead of counting as "1 of
  N", and the `files k/N` counter ticks per file. (The bar advances only as each
  file *finishes*, and the weight is duration — so a cluster of high-resolution
  files, which cost more than their duration implies, can make the bar appear to
  stall while the CPU is in fact fully busy. It is not hung.)

The per-run bar covers only the files hashed *this* run (cached files are
excluded), so on a re-run it restarts at 0%; the **library line** is the
cumulative "how close is my whole collection" view.

All progress is driven entirely by the parent process (workers just hash and
return — the display can never stall the build). It is written to **stderr**;
the actual results — the report, or `--json` — go to **stdout**, so you can pipe
or redirect results cleanly while still watching progress on screen. When stderr
is not a terminal (piped/redirected), the per-worker lines are suppressed and
only a single overall bar is shown.

```sh
uv run vidsim pairs /videos --json > pairs.json   # progress on screen, clean JSON in the file
uv run vidsim duration /videos 2>/dev/null         # prints just the total seconds
```

### Machine-readable output

Both `find` and `pairs` accept `--json` for scripting a review/dedup workflow
(`--json` also suppresses the progress bar so stdout stays pure JSON).

---

## Tuning

Defaults live in `src/vidsim/config.py`; the common ones are also CLI flags.

| Flag | Default | Meaning |
|------|---------|---------|
| `--seconds-per-hash` | `1.0` | Sampling interval. Lower = finer time resolution, more hashes, slower. **Changing this invalidates the cache** (it's part of the hash identity). |
| `--quality-floor` | `50` | Drop frames whose PDQ quality is below this (0–100). |
| `--frame-radius` | `31` | Per-frame Hamming match radius in bits (max 256). Higher = looser matching, catches heavier edits but more noise. |
| `--min-coverage` | `0.05` | Only *display* a pair if a direction reaches this coverage. A display filter over the cached pairs (changing it never recomputes). |
| `--min-shared-frames` | `2` | A pair needs this many candidate frame matches before it's scored at all (prunes the all-pairs scan). Changing it invalidates the pairs cache — delete `.vidsim/pairs.json`. |
| `--index-kind` | `flat` | `flat` = exact Hamming search (`IndexBinaryFlat`); `hnsw` = approximate, much faster at tens of millions of frames. The cache is unaffected, so you can switch freely. |
| `--workers` | ~¾ of cores | (`cache-build`) Parallel hashing worker processes. Higher saturates the CPU; `1` forces serial. Each worker pins ffmpeg to one thread to avoid oversubscription. |

Internal knobs not exposed as flags (edit `config.py`):
`offset_tolerance_s` (offset bin width, absorbs re-encode jitter),
`min_segment_frames` (minimum frames for a shared segment), and
`max_frame_gap_s` (the floor for how far apart two aligned frames can be and
still count as one continuous segment; the effective gap scales up with the
sampling interval, so coarse `--seconds-per-hash` values still work).

**Oversized-file exclusion** (always on, also in `config.py`):
`SKIP_MIN_WIDTH` / `SKIP_MIN_HEIGHT` / `SKIP_MIN_DURATION_S` /
`SKIP_MIN_SIZE_BYTES`. A file meeting **all** of these (4K AND > 20 min AND
> 3 GB) is skipped by `cache-build` — see
[Build the cache](#1-build-the-cache--index-over-your-collection).

The cache lives in hidden `.vidsim/` folders alongside your videos (see
[Content-keyed, per-folder cache](#content-keyed-per-folder-cache)), so it travels
with them when you move folders. To discard it, delete the `.vidsim/` directories.

---

## Scaling notes

The binding constraint is **total frame count**, not video count. At 1 fps, a few
thousand hour-long videos is tens of millions of 256-bit hashes:

- `IndexBinaryFlat` is exact but scans the whole index per query — fine up to a
  point, then slow.
- Switch to `--index-kind hnsw` (approximate) when flat lookups start to drag.
- The all-pairs `pairs` scan stays near-linear because FAISS generates candidate
  pairs first; the expensive aligner only runs on pairs that share real frames.

---

## Project layout

```
src/vidsim/
  config.py      versioned hash params + tunable match params, cache paths
  fileid.py      content-based cache key (free exact-dup detection)
  cache.py       per-folder JSON records; incremental, parallel, resumable sync
  hashing.py     vpdq wrapper → (N, 32) uint8 frame-hash matrix
  index.py       FAISS binary index + row-map (row → file_id/frame/timestamp)
  aggregate.py   THE CORE: temporal Hough voting → shared segments
  pairs_cache.py incremental pairs cache (add/remove diff, both-direction add)
  report.py      directional % + segment list + size/length (text / JSON)
  vid_length.py  parallel PyAV duration reader (runtime estimate + bar weighting)
  ui.py          parent-side progress bars + narration helpers (stderr only)
  cli.py         cache-build · index · find · pairs · duration
```

---

## References

- Facebook/Meta **ThreatExchange** — PDQ & vPDQ perceptual hashing.
- **FAISS** binary indexes (Hamming-distance nearest neighbour).
- Temporal alignment / Hough voting for video copy localization
  (e.g. the VCDB partial-copy-detection formulation).
