#!/usr/bin/env python3
"""replay_spec_log.py - stream an existing SPEC log into a file as if it were live.

Lets you exercise `beamlog.py tail` and the GUI without a beamline: it copies a
recorded SPEC transcript into a destination file a bit at a time, so a watching
`tail` sees commands arrive one by one.

  # terminal 1: watch the (not-yet-existing) live file
  python3 beamlog.py tail /tmp/live.log

  # terminal 2: drip the recorded log into it
  python3 replay_spec_log.py test_data/JW_test_spec_log.txt /tmp/live.log

  # browser: annotate as actions appear
  python3 beamlog.py gui

--by block (default) releases one SPEC command block at a time (feels like
commands being typed); --by line drips line-by-line. The dest is truncated
first unless --append is given.
"""

from __future__ import annotations

import argparse
import re
import time

PROMPT_RE = re.compile(r"^\d+\.[A-Za-z0-9_]+>")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="recorded SPEC log to replay")
    ap.add_argument("dest", help="file to stream into (what `bl tail` watches)")
    ap.add_argument("--delay", type=float, default=0.6, help="seconds per unit (default 0.6)")
    ap.add_argument("--by", choices=["block", "line"], default="block")
    ap.add_argument("--append", action="store_true", help="append instead of truncating dest")
    args = ap.parse_args()

    with open(args.source, "r", errors="replace") as f:
        lines = f.read().splitlines(keepends=True)

    with open(args.dest, "a" if args.append else "w") as out:
        if args.by == "line":
            for ln in lines:
                out.write(ln); out.flush()
                time.sleep(args.delay)
        else:
            block: list[str] = []
            for ln in lines:
                if PROMPT_RE.match(ln) and block:
                    out.write("".join(block)); out.flush()
                    block = []
                    time.sleep(args.delay)
                block.append(ln)
            if block:
                out.write("".join(block)); out.flush()
    print(f"replayed {args.source} -> {args.dest}")


if __name__ == "__main__":
    main()
