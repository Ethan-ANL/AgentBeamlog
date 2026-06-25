#!/usr/bin/env python3
"""beamlog - lightweight capture of beamline actions + reasoning/observation.

One SQLite file, stdlib only. Two tables:

  experiments(id, created_at, user, material, technique, goal)
  actions(id, experiment_id -> experiments.id, created_at,
          command, output, reasoning, observation)

The scientist uses SPEC normally. SPEC already writes a session transcript
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
from datetime import datetime

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
"""


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
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

    Returns a list of dicts: {command, output, ts}. Blocks with an empty
    command (a bare Enter at the prompt) are dropped. `ts` is taken from a
    "#C <date>." line inside the output when present, else None.
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
        output = "\n".join(out_lines)
        ts = None
        dm = SPEC_DATE_RE.search(output)
        if dm:
            try:
                ts = datetime.strptime(dm.group(1), "%a %b %d %H:%M:%S %Y").isoformat(
                    timespec="seconds"
                )
            except ValueError:
                ts = None
        blocks.append({"command": command, "output": output, "ts": ts})
    return blocks


def insert_actions(conn, exp_id, blocks):
    """Insert a list of parsed blocks (skipping empty commands). Returns count.

    created_at uses the SPEC "#C" timestamp when the block has one, otherwise
    the current time (accurate to the poll interval during live `tail`)."""
    n = 0
    for b in blocks:
        if not b["command"]:
            continue
        conn.execute(
            """INSERT INTO actions (experiment_id, created_at, command, output)
               VALUES (?,?,?,?)""",
            (exp_id, b["ts"] or now_iso(), b["command"], b["output"] or None),
        )
        n += 1
    return n


def ingest_file(conn, logfile, exp_id, follow_tail=False):
    """Parse `logfile` and insert any new complete command blocks.

    When `follow_tail` is True the final block is held back (its output may
    still be growing), so only blocks terminated by a later prompt are stored.
    Resumes via ingest_state.prompts_consumed. Returns number of rows inserted.
    """
    with open(logfile, "r", errors="replace") as f:
        text = f.read()
    blocks = parse_blocks(text)
    total = len(blocks)
    if follow_tail and total:
        total -= 1  # hold the last (possibly-incomplete) block

    row = conn.execute(
        "SELECT prompts_consumed FROM ingest_state WHERE logfile=?", (logfile,)
    ).fetchone()
    consumed = row["prompts_consumed"] if row else 0
    if consumed > total:          # file rotated/truncated -> start over
        consumed = 0

    new_blocks = blocks[consumed:total]
    inserted = insert_actions(conn, exp_id, new_blocks)
    conn.execute(
        """INSERT INTO ingest_state (logfile, prompts_consumed, experiment_id)
           VALUES (?,?,?)
           ON CONFLICT(logfile) DO UPDATE SET prompts_consumed=excluded.prompts_consumed,
                                              experiment_id=excluded.experiment_id""",
        (logfile, total, exp_id),
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
        n = ingest_file(conn, logfile, exp, follow_tail=False)
    print(f"ingested {n} new action(s) into experiment {exp}")
    return 0


def cmd_tail(args):
    logfile = _need_logfile(args.logfile)
    if logfile is None:
        return 1
    with connect() as conn:
        exp = resolve_experiment(conn, args.exp)
        if exp is None:
            return 1
    print(f"tailing {logfile} -> experiment {exp} (Ctrl-C to stop)")
    try:
        while True:
            if os.path.exists(logfile):
                with connect() as conn:
                    n = ingest_file(conn, logfile, exp, follow_tail=True)
                if n:
                    print(f"  +{n} action(s) @ {now_iso()}")
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
