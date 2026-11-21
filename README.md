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

## ⚠️ Known limitation: near-static videos

Temporal alignment only means something if a video *changes over time*. A phone
video of a mostly-still scene can have ~every frame look identical to every other
frame. For such a video, a clip "aligns" at essentially every offset, and segment
timings become meaningless.

This is **common in real phone footage** — in testing, two of four sample phone
videos had 98–100% of their own frames mutually near-identical. On those, treat
vidsim's segment timestamps with suspicion: a high coverage number still tells you
the videos share content, but *where* is not reliable.

> Per-video static-ness detection — flagging these and reporting "near-static,
> timing can't be localized" instead of inventing segments — is a planned
> improvement, not yet implemented.

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

(Pointing it at a subfolder of an existing collection attaches to that collection
rather than starting a new one.)

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

### Rebuild just the index

If you change the index type without re-hashing:

```sh
uv run vidsim index /path/to/your/videos --index-kind hnsw
```

### Estimating cache-build runtime

`cache-build`'s cost is dominated by frame hashing, and the work per video scales
with its **duration** — at the default 1 frame/second, a video contributes roughly
one hash per second of runtime, regardless of resolution or file size (FFmpeg
seeks to ~1 fps rather than decoding every frame). So **total duration is a good
predictor of build time**. Sum it quickly (metadata only, no decoding):

```sh
uv run vidsim duration /path/to/your/videos
```

```
1000 videos, 2 unreadable
total: 182400s = 50h 40m 0s
```

To turn that into a time estimate, measure your machine's throughput once — run
`cache-build` on a small sample and divide its wall-clock time by that sample's
total duration to get a *seconds-of-build per second-of-video* factor, then
multiply by the figure above. The factor shrinks proportionally if you raise
`--seconds-per-hash` (fewer frames hashed) and grows if you lower it.

### Progress & output streams

Long-running commands (`cache-build`, `pairs`, `duration`) narrate their work: a
header saying what's starting and the item count, a **self-updating progress bar**
with a live count and ETA (for `cache-build`, the bar names the file currently
being hashed, so a long pause reads as "hashing the big one", not a hang), and a
summary of what completed / was skipped / failed.

All of that narration is written to **stderr**; the actual results — the report,
or `--json` — go to **stdout**. So you can pipe or redirect results cleanly while
still watching progress on screen:

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
| `--min-coverage` | `0.15` | Only report a pair if a direction reaches this coverage. |
| `--min-shared-frames` | `4` | A pair needs this many candidate frame matches before it's scored at all (prunes the all-pairs scan). |
| `--index-kind` | `flat` | `flat` = exact Hamming search (`IndexBinaryFlat`); `hnsw` = approximate, much faster at tens of millions of frames. The cache is unaffected, so you can switch freely. |

Internal knobs not exposed as flags (edit `config.py`):
`offset_tolerance_s` (offset bin width, absorbs re-encode jitter) and
`min_segment_frames` (~3 s minimum shared segment).

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
  cache.py       per-video JSON records; incremental sync
  hashing.py     vpdq wrapper → (N, 32) uint8 frame-hash matrix
  index.py       FAISS binary index + row-map (row → file_id/frame/timestamp)
  aggregate.py   THE CORE: temporal Hough voting → shared segments
  report.py      directional % + segment list (text / JSON)
  vid_length.py  parallel PyAV duration sum (runtime estimation)
  cli.py         cache-build · index · find · pairs · duration
```

---

## References

- Facebook/Meta **ThreatExchange** — PDQ & vPDQ perceptual hashing.
- **FAISS** binary indexes (Hamming-distance nearest neighbour).
- Temporal alignment / Hough voting for video copy localization
  (e.g. the VCDB partial-copy-detection formulation).
