#!/usr/bin/env python3
"""beamlog_gui - a tiny browser annotator for the beamlog review queue.

Stdlib only (http.server). Presents un-reviewed actions one at a time, oldest
first. Type a reason and/or observation and press Enter to save & advance, or
just press Enter (or Esc) to skip -- either way the action leaves the queue.

Run via:  python3 beamlog.py gui        (or: python3 beamlog_gui.py)

Binds to 127.0.0.1 by default: the SPEC log / DB stay on the local machine and
are not exposed to the network.
"""

from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import beamlog


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #

def _experiment(conn, exp):
    if exp is None:
        exp = beamlog.latest_experiment_id(conn)
    if exp is None:
        return None
    return conn.execute("SELECT * FROM experiments WHERE id=?", (exp,)).fetchone()


def queue_state(exp):
    """Return the next un-reviewed action + counts for an experiment."""
    with beamlog.connect() as conn:
        row = _experiment(conn, exp)
        if row is None:
            return {"experiment": None}
        eid = row["id"]
        remaining = conn.execute(
            "SELECT COUNT(*) c FROM actions WHERE experiment_id=? AND reviewed_at IS NULL",
            (eid,),
        ).fetchone()["c"]
        nxt = conn.execute(
            """SELECT id, created_at, command, output, reasoning, observation
               FROM actions WHERE experiment_id=? AND reviewed_at IS NULL
               ORDER BY id ASC LIMIT 1""",
            (eid,),
        ).fetchone()
        recent = conn.execute(
            """SELECT id, command, reasoning, observation
               FROM actions WHERE experiment_id=? AND reviewed_at IS NOT NULL
               ORDER BY reviewed_at DESC, id DESC LIMIT 5""",
            (eid,),
        ).fetchall()
    return {
        "experiment": {
            "id": row["id"], "user": row["user"], "material": row["material"],
            "technique": row["technique"], "goal": row["goal"],
        },
        "remaining": remaining,
        "item": dict(nxt) if nxt else None,
        "recent": [dict(r) for r in recent],
    }


def review(action_id, reasoning, observation, skip):
    """Mark an action reviewed; write text unless skipping. Returns rowcount."""
    with beamlog.connect() as conn:
        if skip:
            n = conn.execute(
                "UPDATE actions SET reviewed_at=? WHERE id=?",
                (beamlog.now_iso(), action_id),
            ).rowcount
        else:
            n = conn.execute(
                "UPDATE actions SET reasoning=?, observation=?, reviewed_at=? WHERE id=?",
                (reasoning or None, observation or None, beamlog.now_iso(), action_id),
            ).rowcount
    return n


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    exp = None  # set by serve()

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/api/queue":
            qs = parse_qs(urlparse(self.path).query)
            exp = int(qs["exp"][0]) if "exp" in qs else self.exp
            self._send(200, json.dumps(queue_state(exp)))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if urlparse(self.path).path != "/api/review":
            self._send(404, json.dumps({"error": "not found"}))
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, json.dumps({"error": "bad json"}))
            return
        try:
            review(
                int(body["id"]),
                (body.get("reasoning") or "").strip(),
                (body.get("observation") or "").strip(),
                bool(body.get("skip")),
            )
        except (KeyError, ValueError, TypeError):
            self._send(400, json.dumps({"error": "bad request"}))
            return
        # return the refreshed queue so the client advances in one round-trip
        self._send(200, json.dumps(queue_state(self.exp)))


# --------------------------------------------------------------------------- #
# page (single embedded HTML/CSS/JS file)
# --------------------------------------------------------------------------- #

PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>beamlog review</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font: 15px/1.5 system-ui, sans-serif; margin: 0; background: #0f1115; color: #e6e6e6; }
  .wrap { max-width: 760px; margin: 0 auto; padding: 18px; }
  header { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
  header h1 { font-size: 16px; margin: 0; font-weight: 600; }
  header .meta { color: #9aa4b2; font-size: 13px; }
  .badge { margin-left: auto; background: #1f6feb; color: #fff; border-radius: 999px;
           padding: 2px 11px; font-size: 13px; font-weight: 600; }
  .badge.done { background: #2ea043; }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 12px; padding: 16px; }
  .cmd { font-family: ui-monospace, monospace; font-size: 16px; color: #7ee787;
         background: #0c0e13; border-radius: 8px; padding: 10px 12px; word-break: break-all; }
  .when { color: #6e7681; font-size: 12px; margin: 6px 2px 0; }
  details.out { margin-top: 10px; }
  details.out summary { cursor: pointer; color: #9aa4b2; font-size: 13px; }
  pre.out { font-family: ui-monospace, monospace; font-size: 12px; color: #b8c0cc;
            background: #0c0e13; border-radius: 8px; padding: 10px; margin: 8px 0 0;
            max-height: 220px; overflow: auto; white-space: pre-wrap; }
  label { display: block; margin: 14px 2px 5px; font-size: 13px; color: #9aa4b2; }
  textarea { width: 100%; background: #0c0e13; color: #e6e6e6; border: 1px solid #2a3140;
             border-radius: 8px; padding: 9px 11px; font: inherit; resize: vertical; min-height: 46px; }
  textarea:focus { outline: none; border-color: #1f6feb; }
  .row { display: flex; gap: 10px; margin-top: 14px; align-items: center; }
  button { font: inherit; border: 0; border-radius: 8px; padding: 9px 16px; cursor: pointer; }
  .save { background: #1f6feb; color: #fff; font-weight: 600; }
  .skip { background: #21262d; color: #c9d1d9; }
  .hint { color: #6e7681; font-size: 12px; margin-left: auto; }
  .empty { text-align: center; color: #9aa4b2; padding: 40px 0; }
  .recent { margin-top: 22px; }
  .recent h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: #6e7681; }
  .recent .r { border-top: 1px solid #20252f; padding: 8px 2px; font-size: 13px; }
  .recent .rc { font-family: ui-monospace, monospace; color: #9aa4b2; }
  .recent .rt { color: #768; }
  kbd { background: #21262d; border: 1px solid #30363d; border-bottom-width: 2px;
        border-radius: 5px; padding: 0 5px; font-size: 11px; font-family: ui-monospace, monospace; }
</style></head>
<body><div class="wrap">
  <header>
    <h1>beamlog review</h1>
    <span class="meta" id="meta"></span>
    <span class="badge" id="badge">…</span>
  </header>
  <div id="main"></div>
  <div class="recent" id="recent"></div>
</div>
<script>
let cur = null;

function esc(s){ return (s||"").replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

function render(state){
  const meta = document.getElementById('meta');
  const badge = document.getElementById('badge');
  const main = document.getElementById('main');
  const recent = document.getElementById('recent');

  if(!state.experiment){
    meta.textContent=""; badge.textContent="—";
    main.innerHTML = '<div class="card empty">No experiment yet. Create one with <code>bl experiment …</code>.</div>';
    recent.innerHTML=""; cur=null; return;
  }
  const e = state.experiment;
  meta.textContent = [e.user, e.material, e.technique, e.goal ? "· " + e.goal : ""].filter(Boolean).join("  ·  ");

  const it = state.item;
  cur = it;
  if(!it){
    badge.textContent = "all caught up"; badge.className = "badge done";
    main.innerHTML = '<div class="card empty">✓ Nothing to review. New actions will appear here automatically.</div>';
  } else {
    badge.textContent = state.remaining + " to review"; badge.className = "badge";
    main.innerHTML = `
      <div class="card">
        <div class="cmd">#${it.id}&nbsp; ${esc(it.command)}</div>
        <div class="when">${esc(it.created_at)}</div>
        ${it.output ? `<details class="out"><summary>SPEC output</summary><pre class="out">${esc(it.output)}</pre></details>` : ``}
        <label>Why (reasoning)</label>
        <textarea id="why" placeholder="why this command…">${esc(it.reasoning)}</textarea>
        <label>Observation (what happened / what you learned)</label>
        <textarea id="obs" placeholder="optional…">${esc(it.observation)}</textarea>
        <div class="row">
          <button class="save" onclick="save()">Save &amp; next</button>
          <button class="skip" onclick="skip()">Skip</button>
          <span class="hint"><kbd>Enter</kbd> save · <kbd>Shift</kbd>+<kbd>Enter</kbd> newline · <kbd>Esc</kbd> skip</span>
        </div>
      </div>`;
    const why = document.getElementById('why');
    why.focus();
    for(const el of [why, document.getElementById('obs')]){
      el.addEventListener('keydown', ev => {
        if(ev.key === 'Enter' && !ev.shiftKey){ ev.preventDefault(); save(); }
        else if(ev.key === 'Escape'){ ev.preventDefault(); skip(); }
      });
    }
  }

  recent.innerHTML = state.recent && state.recent.length ? '<h2>recently reviewed</h2>' +
    state.recent.map(r => `<div class="r"><span class="rc">#${r.id} ${esc(r.command)}</span>` +
      (r.reasoning ? ` — ${esc(r.reasoning)}` : ' — <span class="rt">(skipped)</span>') + `</div>`).join('') : '';
}

async function load(){ render(await (await fetch('/api/queue')).json()); }

async function post(skip){
  if(!cur) return;
  const why = document.getElementById('why'), obs = document.getElementById('obs');
  const r = await fetch('/api/review', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({id: cur.id, reasoning: why?why.value:"", observation: obs?obs.value:"", skip})});
  render(await r.json());
}
function save(){ post(false); }
function skip(){ post(true); }

// poll for new actions when the queue is empty (tail keeps adding them)
setInterval(() => { if(!cur) load(); }, 3000);
load();
</script>
</body></html>
"""


def serve(exp=None, host="127.0.0.1", port=8765, open_browser=True):
    Handler.exp = exp
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"beamlog review queue at {url}  (Ctrl-C to stop)")
    print(f"  db: {beamlog.DB_PATH}")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - headless is fine, just print the URL
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="beamlog browser review queue")
    ap.add_argument("--exp", type=int, help="experiment id (default: most recent)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    raise SystemExit(serve(exp=a.exp, host=a.host, port=a.port, open_browser=not a.no_browser))
