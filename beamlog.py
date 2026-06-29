#!/usr/bin/env python3
"""beamlog - lightweight capture of beamline actions + reasoning/observation.

One SQLite file, stdlib only. Two tables:

  experiments(id, created_at, user, material, technique, goal)
  actions(id, experiment_id -> experiments.id, created_at,
          command, output, reasoning, observation)

User uses SPEC normally. SPEC already writes a session transcript
(the log file in its logs/ dir) recording every command + its output. We read
that file -- no wrapper, no macro, no change to how they work.

Typical flow:

  # once, at the start of an experiment
  bl experiment --user alice --material "sample X" \
                --technique "single-crystal XRD" --goal "align (0 0 L) rod"

  # follow SPEC's log live; every command lands in the DB automatically
  bl tail /path/to/data/.../logs/session.log     # or just: bl tail  (resolved from config)

  # (or one-shot import an existing log)
  bl ingest /path/to/spec_log.txt

  # annotate reasoning/observation later, out of band (never blocks SPEC)
  bl note --why "checking the (0 0 2) before the rod scan"   # most recent action
  bl note --id 42 --obs "peak centered, fwhm ~0.05 in eta"

  # review / manual entry
  bl recent
  bl experiments
  bl log "ascan tth 20 40 100 1" --why "..."     # manual, e.g. for an agent

log / tail / ingest / recent default to the most recent experiment (--exp N to pick).

Paths are never hardcoded -- the DB and the SPEC log are resolved from a config
file (beamlog.json, gitignored) or env vars; see `beamlog.py resolve` and
beamlog.example.json. The DB defaults into the Data folder, beside the data.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta

# --------------------------------------------------------------------------- #
# configuration -- nothing about any site's directory layout is baked into the
# code. The two paths are independent settings, each resolved by precedence:
#   CLI arg (where applicable) > env var > config file (beamlog.json) > default
# beamlog.json is gitignored.
#
#   db        -> the SQLite database          ($BEAMLOG_DB / "db")
#   spec log  -> the SPEC session transcript  ($BEAMLOG_SPEC_LOG / "spec_log",
#                or $BEAMLOG_DATA_ROOT / "data_root" + a glob)
#
# By default the DB lives in the Data folder (data_root), beside the data it
# describes -- not in this repo.
# --------------------------------------------------------------------------- #

def _load_config() -> dict:
    path = os.environ.get("BEAMLOG_CONFIG", os.path.join(os.getcwd(), "beamlog.json"))
    if os.path.isfile(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001 - config is best-effort
            print(f"warning: could not read {path}: {e}", file=sys.stderr)
    return {}


CONFIG = _load_config()


def _setting(key: str, env: str, default=None):
    """A single setting, resolved env var > config file > default."""
    if env in os.environ:
        return os.environ[env]
    if key in CONFIG:
        return CONFIG[key]
    return default


def resolve_data_root():
    root = _setting("data_root", "BEAMLOG_DATA_ROOT")
    return os.path.expanduser(root) if root else None


def resolve_db() -> str:
    """Path to the SQLite DB. Explicit setting wins; otherwise it defaults
    into the Data folder so the corpus lives beside the experiment data."""
    db = _setting("db", "BEAMLOG_DB")
    if db:
        return os.path.expanduser(db)
    root = resolve_data_root()
    if root:
        return os.path.join(root, "beamlog.db")
    return os.path.join(os.getcwd(), "beamlog.db")


# --------------------------------------------------------------------------- #
# optional area-detector frame capture (off unless `frame_pv` is configured).
# All of this is best-effort: nothing here ever gates command logging.
# --------------------------------------------------------------------------- #

def resolve_frame_pv():
    """The areaDetector PVA channel to capture from (e.g. '13SIM1:Pva1:Image'),
    or the literal 'synthetic' for testing. Unset -> frame capture is off."""
    return _setting("frame_pv", "BEAMLOG_FRAME_PV")


def resolve_frames_dir() -> str:
    """Where cached frames live. Explicit setting wins; otherwise a
    'beamlog_frames' dir beside the DB (i.e. in the Data folder, not the repo)."""
    d = _setting("frames_dir", "BEAMLOG_FRAMES_DIR")
    if d:
        return os.path.expanduser(d)
    return os.path.join(os.path.dirname(DB_PATH), "beamlog_frames")


def resolve_frame_timeout() -> float:
    return float(_setting("frame_timeout", "BEAMLOG_FRAME_TIMEOUT", 1.0))


def resolve_frame_filter():
    """Optional regex; only commands matching it are captured (default: all)."""
    return _setting("frame_filter", "BEAMLOG_FRAME_FILTER")


def resolve_frame_ttl_hours() -> float:
    return float(_setting("frame_ttl_hours", "BEAMLOG_FRAME_TTL_HOURS", 48))


def resolve_frame_cache_max_mb() -> float:
    return float(_setting("frame_cache_max_mb", "BEAMLOG_FRAME_CACHE_MAX_MB", 2048))


def resolve_logfile(arg: str | None = None):
    """Find the SPEC log to read, without hardcoding any layout.

    Precedence:
      1. an explicit path argument
      2. $BEAMLOG_SPEC_LOG / config "spec_log"          (one exact file)
      3. $BEAMLOG_DATA_ROOT / config "data_root" + a glob -- picks the
         most-recently-modified match, so it follows the *current* experiment.
    Returns (path, how) or (None, None) if nothing is configured/found.
    """
    if arg:
        return os.path.abspath(os.path.expanduser(arg)), "argument"

    explicit = _setting("spec_log", "BEAMLOG_SPEC_LOG")
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit)), "spec_log"

    root = resolve_data_root()
    if root:
        pattern = _setting("log_glob", "BEAMLOG_LOG_GLOB", "**/*.log")
        matches = [
            p for p in glob.glob(os.path.join(root, pattern), recursive=True)
            if "scanlist" not in os.path.basename(p).lower()  # SPEC's scan *list*, not the session log
        ]
        if matches:
            return max(matches, key=os.path.getmtime), "data_root (newest match)"

    return None, None


DB_PATH = resolve_db()


def _need_logfile(arg):
    """Resolve a logfile or print actionable guidance and return None."""
    path, _how = resolve_logfile(arg)
    if path is None:
        print(
            "no SPEC log configured. Do one of:\n"
            "  - pass the path:    beamlog.py tail /path/to/session.log\n"
            "  - set an env var:   export BEAMLOG_SPEC_LOG=/path/to/session.log\n"
            "  - or a search root: export BEAMLOG_DATA_ROOT=/path/to/Data\n"
            "  - or a config file: cp beamlog.example.json beamlog.json  (then edit)",
            file=sys.stderr,
        )
        return None
    return path

# A SPEC prompt line, e.g. "3016.PSIC6IDB> ca 0 0 2"
PROMPT_RE = re.compile(r"^\d+\.[A-Za-z0-9_]+>\s?(.*)$")

# Timestamp embedded in SPEC "#C" comment lines, e.g.
# "#C Tue Jun 22 17:23:03 2021.  g_aa reset ..."
SPEC_DATE_RE = re.compile(
    r"#C\s+([A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\d{4})\."
)

# A bare timestamp line SPEC writes when a command returns, e.g.
# "Fri Jun 26 10:57:21 2026" (same date format as above, no "#C", no period)
# if enabled in config, it is the last line of the output block
BARE_DATE_RE = re.compile(
    r"^([A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\d{4})\s*$"
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    user        TEXT,
    material    TEXT,
    technique   TEXT,
    goal        TEXT
);

CREATE TABLE IF NOT EXISTS actions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,   -- the action index
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    created_at    TEXT NOT NULL,
    command       TEXT NOT NULL,
    output        TEXT,                                -- raw SPEC response (auto)
    reasoning     TEXT,                                -- WHY (human/agent)
    observation   TEXT,                                -- what was learned (human/agent)
    reviewed_at   TEXT                                 -- queue marker: set when annotated OR skipped
);
CREATE INDEX IF NOT EXISTS idx_actions_experiment ON actions(experiment_id, id);

-- bookkeeping so `tail`/`ingest` can resume without duplicating rows
CREATE TABLE IF NOT EXISTS ingest_state (
    logfile          TEXT PRIMARY KEY,
    prompts_consumed INTEGER NOT NULL,
    experiment_id    INTEGER
);

-- optional: one cached area-detector frame per action (see beamlog_frames.py).
-- kept=0 / decided_at NULL means "pending" -- captured but not yet kept; the
-- default outcome is discard. npy_path/png_path are NULL when capture failed
-- (the reason is in `error`), which distinguishes "detector down" from
-- "feature off" (no row at all).
CREATE TABLE IF NOT EXISTS frames (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    action_id  INTEGER NOT NULL UNIQUE REFERENCES actions(id),
    created_at TEXT NOT NULL,
    pv         TEXT,
    npy_path   TEXT,
    png_path   TEXT,
    width      INTEGER,
    height     INTEGER,
    dtype      TEXT,
    unique_id  INTEGER,
    kept       INTEGER NOT NULL DEFAULT 0,
    decided_at TEXT,
    error      TEXT
);
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _parse_spec_date(s: str) -> str | None:
    """Parse a SPEC date like 'Fri Jun 26 10:57:21 2026' to ISO, or None."""
    try:
        return datetime.strptime(
            " ".join(s.split()), "%a %b %d %H:%M:%S %Y"
        ).isoformat(timespec="seconds")
    except ValueError:
        return None


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # The DB normally lives on the shared/NFS Data folder, reached from several
    # beamline desktops. WAL mode is UNSAFE there -- it needs shared memory and a
    # single host, and on a network FS it surfaces as "attempt to write a
    # readonly database". So we keep SQLite's default rollback journal (works on
    # NFS and across machines) and only widen the lock wait, which is the safe way
    # to let `tail` and `gui` write concurrently.
    conn.execute("PRAGMA busy_timeout = 5000")
    # If an earlier build left this DB in WAL mode, flip it back so it's portable
    # again (no-op if already in a rollback journal mode).
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() == "wal":
            conn.execute("PRAGMA journal_mode = DELETE")
    except sqlite3.Error:
        pass
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Tiny forward migrations so existing DBs gain new columns."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(actions)")}
    if "reviewed_at" not in cols:
        conn.execute("ALTER TABLE actions ADD COLUMN reviewed_at TEXT")
        conn.commit()


def latest_experiment_id(conn: sqlite3.Connection) -> int | None:
    row = conn.execute("SELECT id FROM experiments ORDER BY id DESC LIMIT 1").fetchone()
    return row["id"] if row else None


def resolve_experiment(conn: sqlite3.Connection, explicit: int | None) -> int | None:
    if explicit is not None:
        row = conn.execute("SELECT id FROM experiments WHERE id=?", (explicit,)).fetchone()
        if row is None:
            print(f"no experiment with id {explicit}", file=sys.stderr)
            return None
        return explicit
    exp = latest_experiment_id(conn)
    if exp is None:
        print("no experiments yet -- create one with `bl experiment ...`", file=sys.stderr)
    return exp


# --------------------------------------------------------------------------- #
# SPEC log parsing
# --------------------------------------------------------------------------- #

def parse_blocks(text: str) -> list[dict]:
    """Split a SPEC transcript into one block per prompt line.

    Returns a list of dicts: {command, output, ts, complete}. Blocks with an
    empty command (a bare Enter at the prompt) are dropped. `ts` is taken from a
    trailing bare-timestamp line (SPEC's command-completion stamp) when present,
    else a "#C <date>." comment, else None. `complete` is True when a trailing
    timestamp was found -- i.e. the command has finished -- which lets live
    `tail` store the last block without waiting for the next prompt.
    """
    lines = text.splitlines()
    prompts: list[tuple[int, str]] = []  # (line index, command)
    for i, line in enumerate(lines):
        m = PROMPT_RE.match(line)
        if m:
            prompts.append((i, m.group(1).strip()))

    blocks: list[dict] = []
    for k, (idx, command) in enumerate(prompts):
        end = prompts[k + 1][0] if k + 1 < len(prompts) else len(lines)
        out_lines = lines[idx + 1:end]
        # trim surrounding blank lines
        while out_lines and not out_lines[0].strip():
            out_lines.pop(0)
        while out_lines and not out_lines[-1].strip():
            out_lines.pop()

        # A bare timestamp as the final line is SPEC's command-completion stamp:
        # it is logged when the command returns, *before* the next prompt. Use it
        # as the action time, drop it from the stored output, and flag the block
        # complete so live `tail` can store it without waiting for the next
        # prompt. Not every SPEC config emits it -- when absent we fall back to a
        # "#C <date>." comment for the time, and the block stays "incomplete".
        ts = None
        complete = False
        if out_lines:
            bm = BARE_DATE_RE.match(out_lines[-1])
            if bm:
                ts = _parse_spec_date(bm.group(1))
                complete = True
                out_lines.pop()
                while out_lines and not out_lines[-1].strip():
                    out_lines.pop()

        output = "\n".join(out_lines)
        if ts is None:
            dm = SPEC_DATE_RE.search(output)
            if dm:
                ts = _parse_spec_date(dm.group(1))
        blocks.append(
            {"command": command, "output": output, "ts": ts, "complete": complete}
        )
    return blocks


def insert_actions(conn, exp_id, blocks):
    """Insert a list of parsed blocks (skipping empty commands).

    Returns one dict per inserted row -- {id, command, ts, complete} -- so live
    `tail` can decide whether to capture a detector frame for a just-completed
    command. (Callers that only want a count use len() of the result.)

    created_at uses the SPEC "#C" timestamp when the block has one, otherwise
    the current time (accurate to the poll interval during live `tail`)."""
    inserted = []
    for b in blocks:
        if not b["command"]:
            continue
        cur = conn.execute(
            """INSERT INTO actions (experiment_id, created_at, command, output)
               VALUES (?,?,?,?)""",
            (exp_id, b["ts"] or now_iso(), b["command"], b["output"] or None),
        )
        inserted.append({
            "id": cur.lastrowid, "command": b["command"],
            "ts": b["ts"], "complete": b["complete"],
        })
    return inserted


def ingest_file(conn, logfile, exp_id, follow_tail=False):
    """Parse `logfile` and insert any new complete command blocks.

    When `follow_tail` is True the final block is held back only while it looks
    unfinished -- i.e. it carries no completion timestamp yet -- so a command
    that has logged its closing stamp is stored at once, without waiting for the
    next prompt. Configs that don't emit the stamp fall back to that wait.
    Resumes via ingest_state.prompts_consumed. Returns one dict per inserted
    row ({id, command, ts, complete}); callers wanting a count use len().
    """
    with open(logfile, "r", errors="replace") as f:
        text = f.read()
    blocks = parse_blocks(text)
    n_prompts = len(blocks)        # every prompt line, finished or not -- stable
    # How far we ingest this pass: everything, except a still-running final block
    # (no completion stamp yet) which we hold back. A *completed* final block is
    # stored at once -- that is the whole point of the stamp.
    end = n_prompts
    if follow_tail and n_prompts and not blocks[-1]["complete"]:
        end -= 1

    row = conn.execute(
        "SELECT prompts_consumed FROM ingest_state WHERE logfile=?", (logfile,)
    ).fetchone()
    consumed = row["prompts_consumed"] if row else 0
    # Rotation/truncation == fewer prompt lines than we have already consumed.
    # Compare against the prompt COUNT, never against `end`: a completed final
    # block that later gains trailing text (an async message, a "Scan stored in
    # ..." summary) flips back to "incomplete" and lowers `end` below `consumed`
    # -- that is not a rotation, and treating it as one re-ingested the whole
    # file and duplicated every action on the next poll.
    if consumed > n_prompts:
        consumed = 0

    new_blocks = blocks[consumed:end]
    inserted = insert_actions(conn, exp_id, new_blocks)
    # The cursor is a high-water mark: once a block is stored it must never be
    # re-ingested, so the held-back/flip case (end < consumed) leaves it put.
    new_consumed = max(consumed, end)
    conn.execute(
        """INSERT INTO ingest_state (logfile, prompts_consumed, experiment_id)
           VALUES (?,?,?)
           ON CONFLICT(logfile) DO UPDATE SET prompts_consumed=excluded.prompts_consumed,
                                              experiment_id=excluded.experiment_id""",
        (logfile, new_consumed, exp_id),
    )
    conn.commit()
    return inserted


# --------------------------------------------------------------------------- #
# subcommands
# --------------------------------------------------------------------------- #

def cmd_experiment(args):
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO experiments (created_at, user, material, technique, goal)
               VALUES (?,?,?,?,?)""",
            (now_iso(), args.user, args.material, args.technique, args.goal),
        )
        exp_id = cur.lastrowid
    print(f"experiment #{exp_id} created" + (f" -- goal: {args.goal}" if args.goal else ""))
    return 0


def cmd_log(args):
    command = " ".join(args.command).strip()
    if not command:
        print("nothing to log (empty command)", file=sys.stderr)
        return 2
    with connect() as conn:
        exp = resolve_experiment(conn, args.exp)
        if exp is None:
            return 1
        cur = conn.execute(
            """INSERT INTO actions (experiment_id, created_at, command, reasoning)
               VALUES (?,?,?,?)""",
            (exp, now_iso(), command, args.why),
        )
        new_id = cur.lastrowid
    print(f"#{new_id} (exp {exp}) {command}")
    return 0


def cmd_ingest(args):
    logfile = _need_logfile(args.logfile)
    if logfile is None:
        return 1
    with connect() as conn:
        exp = resolve_experiment(conn, args.exp)
        if exp is None:
            return 1
        # historical import never captures frames -- the detector no longer
        # shows what these past commands produced.
        inserted = ingest_file(conn, logfile, exp, follow_tail=False)
    print(f"ingested {len(inserted)} new action(s) into experiment {exp}")
    return 0


# --------------------------------------------------------------------------- #
# live frame capture (best-effort; orchestrated only from `tail`)
# --------------------------------------------------------------------------- #

def _capture_one_frame(conn, inserted, frame_pv, frames_dir, timeout, filt):
    """Capture at most one detector frame for this poll, for the most-recently
    completed newly-inserted command. Best-effort: any failure is recorded in
    the frames row's `error` column and never raises. Returns a status string
    for the tail log, or None if nothing was attempted."""
    import beamlog_frames  # lazy: keeps the optional dep off the import path

    # only fully completed commands -- pick the newest one this poll.
    done = [b for b in inserted if b.get("complete")]
    if not done:
        return None
    block = done[-1]
    if filt is not None and not filt.search(block["command"]):
        return None  # command doesn't match the capture filter

    action_id = block["id"]
    try:
        cap = beamlog_frames.capture(frame_pv, timeout=timeout)
        npy_path, png_path = beamlog_frames.save_pending(frames_dir, action_id, cap)
        conn.execute(
            """INSERT OR REPLACE INTO frames
               (action_id, created_at, pv, npy_path, png_path,
                width, height, dtype, unique_id, kept, decided_at, error)
               VALUES (?,?,?,?,?,?,?,?,?,0,NULL,NULL)""",
            (action_id, now_iso(), cap.get("pv"), npy_path, png_path,
             cap.get("width"), cap.get("height"), cap.get("dtype"),
             cap.get("unique_id")),
        )
        conn.commit()
        return f"frame for #{action_id} ({cap.get('width')}x{cap.get('height')})"
    except beamlog_frames.FrameError as e:
        conn.execute(
            """INSERT OR REPLACE INTO frames
               (action_id, created_at, pv, kept, decided_at, error)
               VALUES (?,?,?,0,NULL,?)""",
            (action_id, now_iso(), frame_pv, str(e)),
        )
        conn.commit()
        return f"frame for #{action_id} failed: {e}"
    except Exception as e:  # noqa: BLE001 - capture must never break ingestion
        return f"frame for #{action_id} skipped: {e}"


def _prune_frames(conn, frames_dir, ttl_hours, cache_max_mb):
    """Discard *pending* (un-kept, undecided) frames so the cache can't fill a
    disk: those whose action is already reviewed, those past the TTL, and -- if
    still over budget -- the oldest first. Kept frames are never touched."""
    import beamlog_frames

    def _drop(rows):
        for r in rows:
            beamlog_frames.discard(r["npy_path"], r["png_path"])
            conn.execute("DELETE FROM frames WHERE id=?", (r["id"],))

    # 1) action already reviewed but frame never decided -> it'll never be kept
    _drop(conn.execute(
        """SELECT f.id, f.npy_path, f.png_path FROM frames f
           JOIN actions a ON a.id = f.action_id
           WHERE f.decided_at IS NULL AND a.reviewed_at IS NOT NULL"""
    ).fetchall())

    # 2) past the TTL
    if ttl_hours > 0:
        cutoff = (datetime.now() - timedelta(hours=ttl_hours)).isoformat(timespec="seconds")
        _drop(conn.execute(
            """SELECT id, npy_path, png_path FROM frames
               WHERE decided_at IS NULL AND created_at < ?""", (cutoff,)
        ).fetchall())

    # 3) over the disk budget -> evict oldest pending first (FIFO)
    if cache_max_mb > 0 and beamlog_frames.dir_size_bytes(frames_dir) > cache_max_mb * 1e6:
        pending = conn.execute(
            """SELECT id, npy_path, png_path FROM frames
               WHERE decided_at IS NULL ORDER BY created_at ASC, id ASC"""
        ).fetchall()
        budget = cache_max_mb * 1e6
        for r in pending:
            if beamlog_frames.dir_size_bytes(frames_dir) <= budget:
                break
            _drop([r])
    conn.commit()


def cmd_tail(args):
    logfile = _need_logfile(args.logfile)
    if logfile is None:
        return 1
    with connect() as conn:
        exp = resolve_experiment(conn, args.exp)
        if exp is None:
            return 1

    # resolve the (optional) frame-capture settings once, up front.
    frame_pv = None if getattr(args, "no_frames", False) else resolve_frame_pv()
    frames_dir = resolve_frames_dir()
    frame_timeout = resolve_frame_timeout()
    ttl_hours = resolve_frame_ttl_hours()
    cache_max_mb = resolve_frame_cache_max_mb()
    filt_pat = resolve_frame_filter()
    filt = re.compile(filt_pat) if filt_pat else None

    print(f"tailing {logfile} -> experiment {exp} (Ctrl-C to stop)")
    if frame_pv:
        print(f"  frames: capturing from {frame_pv} -> {frames_dir}"
              + (f"  filter={filt_pat!r}" if filt_pat else ""))
    try:
        while True:
            if os.path.exists(logfile):
                with connect() as conn:
                    inserted = ingest_file(conn, logfile, exp, follow_tail=True)
                    if inserted:
                        print(f"  +{len(inserted)} action(s) @ {now_iso()}")
                    if frame_pv and inserted:
                        msg = _capture_one_frame(conn, inserted, frame_pv,
                                                 frames_dir, frame_timeout, filt)
                        if msg:
                            print(f"  {msg}")
                    if frame_pv:
                        _prune_frames(conn, frames_dir, ttl_hours, cache_max_mb)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


def cmd_note(args):
    if args.why is None and args.obs is None:
        print("give --why and/or --obs", file=sys.stderr)
        return 2
    with connect() as conn:
        if args.id is not None:
            target = args.id
        else:
            row = conn.execute("SELECT id FROM actions ORDER BY id DESC LIMIT 1").fetchone()
            if row is None:
                print("no actions logged yet", file=sys.stderr)
                return 1
            target = row["id"]
        sets, vals = [], []
        if args.why is not None:
            sets.append("reasoning=?"); vals.append(args.why)
        if args.obs is not None:
            sets.append("observation=?"); vals.append(args.obs)
        sets.append("reviewed_at=?"); vals.append(now_iso())  # leaves the review queue
        vals.append(target)
        n = conn.execute(f"UPDATE actions SET {', '.join(sets)} WHERE id=?", vals).rowcount
    if n == 0:
        print(f"no action with id {target}", file=sys.stderr)
        return 1
    print(f"annotated #{target}")
    return 0


def _short(text, width=70):
    if not text:
        return ""
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:width] + ("..." if len(first) > width else "")


def cmd_recent(args):
    with connect() as conn:
        exp = resolve_experiment(conn, args.exp)
        if exp is None:
            return 1
        rows = conn.execute(
            "SELECT * FROM actions WHERE experiment_id=? ORDER BY id DESC LIMIT ?",
            (exp, args.n),
        ).fetchall()
    if args.json:
        print(json.dumps([dict(r) for r in rows], indent=2))
        return 0
    for r in reversed(rows):
        print(f"#{r['id']:<4} {r['created_at']}  {r['command']}")
        if r["output"]:
            print(f"        out: {_short(r['output'])}")
        if r["reasoning"]:
            print(f"        why: {r['reasoning']}")
        if r["observation"]:
            print(f"        obs: {r['observation']}")
    return 0


def cmd_experiments(args):
    with connect() as conn:
        rows = conn.execute(
            """SELECT e.*, (SELECT COUNT(*) FROM actions a WHERE a.experiment_id=e.id) AS n
               FROM experiments e ORDER BY e.id DESC LIMIT ?""",
            (args.n,),
        ).fetchall()
    if args.json:
        print(json.dumps([dict(r) for r in rows], indent=2))
        return 0
    for r in reversed(rows):
        bits = [b for b in (r["user"], r["material"], r["technique"]) if b]
        print(f"#{r['id']:<3} {r['created_at']}  ({r['n']} actions)  " + "  |  ".join(bits))
        if r["goal"]:
            print(f"      goal: {r['goal']}")
    return 0


def cmd_resolve(args):
    """Show which paths the tool resolved, and how -- a quick sanity check."""
    logfile, how = resolve_logfile(args.logfile)
    print(f"db:       {DB_PATH}")
    print(f"data_root:{resolve_data_root() or ' (unset)'}")
    if logfile:
        print(f"spec log: {logfile}   [{how}]")
    else:
        print("spec log: (none configured -- run `tail`/`ingest` for guidance)")

    # optional frame capture -- show what's configured and whether pvapy is here.
    frame_pv = resolve_frame_pv()
    if frame_pv:
        import beamlog_frames  # lazy
        print(f"frame pv: {frame_pv}")
        print(f"frames:   {resolve_frames_dir()}  "
              f"(timeout {resolve_frame_timeout()}s, ttl {resolve_frame_ttl_hours()}h, "
              f"cap {resolve_frame_cache_max_mb():.0f}MB)")
        filt = resolve_frame_filter()
        print(f"  filter: {filt!r}" if filt else "  filter: (none -- all commands)")
        if frame_pv == beamlog_frames.SYNTHETIC:
            print("  pvapy:  not needed (synthetic test source)")
        else:
            print(f"  pvapy:  {'available' if beamlog_frames.pvapy_available() else 'NOT installed -- pip install .[frames]'}")
    else:
        print("frame pv: (unset -- frame capture off; set frame_pv to enable)")
    return 0


def cmd_frames(args):
    """List cached frames, or garbage-collect pending ones (`bl frames gc`)."""
    with connect() as conn:
        if args.gc:
            _prune_frames(conn, resolve_frames_dir(),
                          resolve_frame_ttl_hours(), resolve_frame_cache_max_mb())
            print("pruned pending frames")
            return 0
        rows = conn.execute(
            """SELECT f.*, a.command FROM frames f
               JOIN actions a ON a.id = f.action_id
               ORDER BY f.id DESC LIMIT ?""",
            (args.n,),
        ).fetchall()
    if args.json:
        print(json.dumps([dict(r) for r in rows], indent=2))
        return 0
    if not rows:
        print("no frames captured yet")
        return 0
    for r in reversed(rows):
        if r["error"]:
            state = f"ERROR: {r['error']}"
        elif r["kept"]:
            state = "kept"
        elif r["decided_at"]:
            state = "discarded"
        else:
            state = "pending"
        dims = f"{r['width']}x{r['height']}" if r["width"] else "-"
        print(f"#{r['action_id']:<4} [{state:<9}] {dims:>11}  {r['command']}")
    return 0


def cmd_gui(args):
    """Launch the browser-based review-queue annotator (stdlib http.server)."""
    import beamlog_gui  # lazy import to avoid a load-time cycle
    return beamlog_gui.serve(exp=args.exp, host=args.host, port=args.port,
                             open_browser=not args.no_browser)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="bl", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("experiment", help="create an experiment")
    pe.add_argument("--user")
    pe.add_argument("--material")
    pe.add_argument("--technique")
    pe.add_argument("--goal")
    pe.set_defaults(func=cmd_experiment)

    pt = sub.add_parser("tail", help="follow a SPEC log file live")
    pt.add_argument("logfile", nargs="?", help="path (default: resolved from config/env)")
    pt.add_argument("--exp", type=int, help="experiment id (default: most recent)")
    pt.add_argument("--interval", type=float, default=1.0, help="poll seconds (default 1)")
    pt.add_argument("--no-frames", action="store_true",
                    help="disable detector-frame capture even if frame_pv is set")
    pt.set_defaults(func=cmd_tail)

    pi = sub.add_parser("ingest", help="one-shot import of an existing SPEC log")
    pi.add_argument("logfile", nargs="?", help="path (default: resolved from config/env)")
    pi.add_argument("--exp", type=int, help="experiment id (default: most recent)")
    pi.set_defaults(func=cmd_ingest)

    prs = sub.add_parser("resolve", help="show resolved db + log paths and exit")
    prs.add_argument("logfile", nargs="?", help="optional path to test resolution")
    prs.set_defaults(func=cmd_resolve)

    pg = sub.add_parser("gui", help="browser review-queue annotator")
    pg.add_argument("--exp", type=int, help="experiment id (default: most recent)")
    pg.add_argument("--host", default="127.0.0.1", help="bind host (default localhost)")
    pg.add_argument("--port", type=int, default=8765, help="port (default 8765)")
    pg.add_argument("--no-browser", action="store_true", help="don't auto-open a browser")
    pg.set_defaults(func=cmd_gui)

    pl = sub.add_parser("log", help="manually log a single action")
    pl.add_argument("command", nargs=argparse.REMAINDER, help="e.g. ascan tth 20 40 100 1")
    pl.add_argument("--why", help="optional reasoning to record immediately")
    pl.add_argument("--exp", type=int, help="experiment id (default: most recent)")
    pl.set_defaults(func=cmd_log)

    pn = sub.add_parser("note", help="attach reasoning/observation (non-blocking)")
    pn.add_argument("--id", type=int, help="action id (default: most recent action)")
    pn.add_argument("--why", help="reasoning text")
    pn.add_argument("--obs", help="observation text")
    pn.set_defaults(func=cmd_note)

    pf = sub.add_parser("frames", help="list cached detector frames / garbage-collect")
    pf.add_argument("gc", nargs="?", help="pass 'gc' to prune pending frames now")
    pf.add_argument("-n", type=int, default=30, help="how many to list (default 30)")
    pf.add_argument("--json", action="store_true")
    pf.set_defaults(func=cmd_frames)

    pr = sub.add_parser("recent", help="show recent actions")
    pr.add_argument("-n", type=int, default=15, help="how many (default 15)")
    pr.add_argument("--exp", type=int, help="experiment id (default: most recent)")
    pr.add_argument("--json", action="store_true")
    pr.set_defaults(func=cmd_recent)

    px = sub.add_parser("experiments", help="list experiments")
    px.add_argument("-n", type=int, default=20, help="how many (default 20)")
    px.add_argument("--json", action="store_true")
    px.set_defaults(func=cmd_experiments)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
