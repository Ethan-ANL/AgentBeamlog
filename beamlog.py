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
  bl experiment --user jwkim --material "CrI3 / Br3" \
                --technique "single-crystal XRD" --goal "align (0 0 L) rod"

  # follow SPEC's log live; every command lands in the DB automatically
  bl tail /path/to/.../logs/crixbr3-x_1.log

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
DB is ./beamlog.db (override with $BEAMLOG_DB).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime

DB_PATH = os.environ.get("BEAMLOG_DB", os.path.join(os.getcwd(), "beamlog.db"))

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
    observation   TEXT                                 -- what was learned (human/agent)
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
    return conn


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
    with connect() as conn:
        exp = resolve_experiment(conn, args.exp)
        if exp is None:
            return 1
        n = ingest_file(conn, os.path.abspath(args.logfile), exp, follow_tail=False)
    print(f"ingested {n} new action(s) into experiment {exp}")
    return 0


def cmd_tail(args):
    logfile = os.path.abspath(args.logfile)
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
    pt.add_argument("logfile")
    pt.add_argument("--exp", type=int, help="experiment id (default: most recent)")
    pt.add_argument("--interval", type=float, default=1.0, help="poll seconds (default 1)")
    pt.set_defaults(func=cmd_tail)

    pi = sub.add_parser("ingest", help="one-shot import of an existing SPEC log")
    pi.add_argument("logfile")
    pi.add_argument("--exp", type=int, help="experiment id (default: most recent)")
    pi.set_defaults(func=cmd_ingest)

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
