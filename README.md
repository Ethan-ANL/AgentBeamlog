# AgentBeamlog

Lightweight capture of beamline **actions + reasoning + observations** during
experiments, for later agent fine-tuning and assistance.

Design goals: **lightweight, zero runtime dependencies (Python stdlib only),
nothing about any site's directory layout hardcoded.**

## How it works

SPEC (planned to expand to bluesky) already writes a session transcript (the log in its `logs/` dir) recording
every command typed *and* its output. AgentBeamlog **reads that
file**; the scientist changes nothing about how they work. No wrapper, no macro,
no SPEC server.

```
SPEC session log ──tail──▶ actions table ──┐
                                           ├──▶ beamlog.db (SQLite)
human/agent reasoning ─────annotate────────┘
```

Reasoning/observation are added **out of band** so they never block commands:
either from the CLI (`bl note`) or a tiny browser **review queue** (`bl gui`).

## Install

Zero dependencies. Either install the `bl` command:

```bash
pip install -e .        # or: uv pip install -e .   -> provides `bl`
```

…or just run the script directly (`python beamlog.py …`). Examples below use `bl`.

## Configure (no paths in the code)

Point AgentBeamlog at your data. Resolution precedence is **CLI arg → env var →
config file** (`beamlog.json`, gitignored). The DB defaults into the Data folder
so the corpus lives beside the data, not in this repo.

```bash
cp beamlog.example.json beamlog.json     # then edit (it's gitignored)
```
```jsonc
{
  "data_root": "/path/to/Data",          // newest log under here is followed automatically
  "log_glob":  "**/logs/*.log"           // optional; default "**/*.log", ignores *scanlist*
  // or set "spec_log" to one exact file, and/or "db" to an explicit DB path
}
```

Env vars override config: `BEAMLOG_DB`, `BEAMLOG_SPEC_LOG`, `BEAMLOG_DATA_ROOT`,
`BEAMLOG_LOG_GLOB`, `BEAMLOG_CONFIG` (path to a non-default config file).

Check what got resolved:
```bash
bl resolve
```

## Use

```bash
# once per experiment
bl experiment --user alice --material "sample X" \
              --technique "single-crystal XRD" --goal "align (0 0 L) rod"

# capture: follow the live SPEC log (leave running in a terminal)
bl tail            # path resolved from config, or: bl tail /path/to/session.log

# annotate, anytime, without blocking SPEC
bl gui             # browser review queue (recommended)
bl note --why "checking (0 0 2) before the rod scan"      # most recent action
bl note --id 42 --obs "peak centered, fwhm ~0.05 in eta"

# review / export
bl recent
bl recent --json   # clean tuples for building a fine-tuning set
bl experiments
```

The **review queue** (`bl gui`) shows the un-reviewed backlog one row each, with
the command + its SPEC output. Type *why*/*observation* and press `Enter` to save
and move to the next; `Esc` skips (nothing written, leaves the queue). It runs on
`127.0.0.1` only and picks up newly-tailed commands automatically.

## Detector frames (optional)

Optionally capture an **area-detector frame** on each command completion. This is
the **only** part of the project with a runtime dependency, and it's opt-in:

```bash
pip install -e ".[frames]"     # adds pvapy + numpy, for this feature only
```

Point it at the detector's EPICS **PVA Image** channel (the `NDPluginPva` output)
in `beamlog.json`:

```jsonc
{
  "frame_pv": "13SIM1:Pva1:Image",   // or "synthetic" to test without a beamline
  "frame_filter": "^(a|d)?scan|^count\\b"  // optional: only these commands (default: all)
  // frames_dir, frame_timeout, frame_ttl_hours, frame_cache_max_mb also configurable
}
```

While `bl tail` runs, each completed command grabs the **current** frame and
**caches** it as *pending*: a raw `.npy` (full bit-depth, the corpus artifact)
plus an 8-bit thumbnail PNG. In the review queue each row then shows the
thumbnail and a **"keep frame"** checkbox — **unchecked by default**. Save/skip
*without* checking it discards the frame (files deleted); check it to keep the
frame as part of the corpus. Pending frames left undecided are pruned by a TTL +
disk-budget sweep (also runnable via `bl frames gc`; `bl frames` lists them).

Capture is **best-effort and never blocks logging**: a down detector, a missing
`pvapy`, a compressed codec, or an unsupported color mode just records an `error`
on the frame row and ingestion continues. Confirm what's configured (and whether
`pvapy` is importable) with `bl resolve`.

> **Timing caveat:** because beamlog reads a log file, the captured frame reflects
> detector state ~1 poll *after* the command completed — i.e. "what the detector
> showed around when this command finished", not a frame intrinsically bound to
> the command. Mono and RGB1 frames are supported; compressed codecs and
> RGB2/RGB3 are skipped (decompression would need a C library this project avoids).

## Testing without a beamline

Replay a recorded SPEC log into a file as if it were live (set
`"frame_pv": "synthetic"` to also exercise the full frame pipeline — no pvapy
needed):

```bash
bl tail /tmp/live.log                                          # terminal 1
python replay_spec_log.py test_data/sample.log /tmp/live.log   # terminal 2
bl gui                                                         # browser
```

## Schema

Two core tables (`reviewed_at` is queue bookkeeping for the GUI; it never
pollutes the text columns), plus an optional `frames` table:

```
experiments(id, created_at, user, material, technique, goal)
actions(id, experiment_id → experiments, created_at,
        command, output, reasoning, observation, reviewed_at)
frames(id, action_id → actions, created_at, pv, npy_path, png_path,
       width, height, dtype, unique_id, kept, decided_at, error)
```

- `command` / `output` — captured automatically from the SPEC log.
- `reasoning` / `observation` — added by a human or agent.
- `frames` — one optional cached detector frame per action; `kept=0` /
  `decided_at IS NULL` means *pending* (default outcome is discard).

## Files

| file | what |
|------|------|
| `beamlog.py` | core: config resolution, DB, SPEC-log parsing, CLI |
| `beamlog_gui.py` | browser review-queue annotator (stdlib `http.server`) |
| `beamlog_frames.py` | optional area-detector capture (pvapy) — isolated, lazy-imported |
| `replay_spec_log.py` | dev tool: stream a recorded log in as if live |
| `beamlog.example.json` | config template (copy to gitignored `beamlog.json`) |
| `pyproject.toml` | provides the `bl` command |

Local configs (`*.json` except `*.example.json`), databases (`*.db`), logs
(`*.log`, `test_data/`), and cached frames (`*.npy`, `beamlog_frames/`) are
gitignored so no private paths or data are pushed.

## Roadmap: Bluesky

The schema is deliberately **source-agnostic** — `command`/`output` plus the
reasoning/observation layer describe an action regardless of where it came from.
Adding Bluesky support means writing a second ingester (e.g. subscribing to the
RunEngine document stream, or reading from Tiled/databroker) that inserts the
same `actions` rows; the DB, CLI, and GUI are unchanged. A small `source` column
(`'spec'` | `'bluesky'`) is the natural next step to tag rows by origin.
