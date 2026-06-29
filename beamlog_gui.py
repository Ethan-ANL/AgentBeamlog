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


QUEUE_CAP = 200  # most un-reviewed rows sent to the page at once


def _frame_info(conn, action_id):
    """Per-action frame summary for the queue (None if no frame row). Exposes
    only id-keyed metadata -- never a filesystem path (path-traversal safety)."""
    r = conn.execute(
        "SELECT width, height, kept, error, png_path FROM frames WHERE action_id=?",
        (action_id,),
    ).fetchone()
    if r is None:
        return None
    return {
        "present": bool(r["png_path"]),   # a viewable thumbnail exists
        "width": r["width"], "height": r["height"],
        "kept": bool(r["kept"]),
        "error": r["error"],
    }


def queue_state(exp):
    """Return the un-reviewed backlog (oldest first) + recently reviewed."""
    with beamlog.connect() as conn:
        row = _experiment(conn, exp)
        if row is None:
            return {"experiment": None}
        eid = row["id"]
        remaining = conn.execute(
            "SELECT COUNT(*) c FROM actions WHERE experiment_id=? AND reviewed_at IS NULL",
            (eid,),
        ).fetchone()["c"]
        items = conn.execute(
            """SELECT id, created_at, command, output, reasoning, observation
               FROM actions WHERE experiment_id=? AND reviewed_at IS NULL
               ORDER BY id ASC LIMIT ?""",
            (eid, QUEUE_CAP),
        ).fetchall()
        recent = conn.execute(
            """SELECT id, command, reasoning, observation
               FROM actions WHERE experiment_id=? AND reviewed_at IS NOT NULL
               ORDER BY reviewed_at DESC, id DESC LIMIT 8""",
            (eid,),
        ).fetchall()
        item_dicts = []
        for r in items:
            d = dict(r)
            d["frame"] = _frame_info(conn, r["id"])
            item_dicts.append(d)
    return {
        "experiment": {
            "id": row["id"], "user": row["user"], "material": row["material"],
            "technique": row["technique"], "goal": row["goal"],
        },
        "remaining": remaining,
        "items": item_dicts,
        "recent": [dict(r) for r in recent],
    }


def frame_png(action_id):
    """Return (bytes, None) for the cached thumbnail of `action_id`, or
    (None, reason). Looks the path up server-side by row id -- the client never
    supplies a path."""
    with beamlog.connect() as conn:
        r = conn.execute(
            "SELECT png_path FROM frames WHERE action_id=?", (action_id,)
        ).fetchone()
    if r is None or not r["png_path"]:
        return None, "no frame"
    try:
        with open(r["png_path"], "rb") as f:
            return f.read(), None
    except OSError as e:
        return None, str(e)


def review(action_id, reasoning, observation, skip, keep_frame=False):
    """Mark an action reviewed; write text unless skipping. Also decide the
    action's cached frame: keep_frame True -> retain (kept=1); otherwise discard
    -- delete both cached files. Default is discard. Returns rowcount."""
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
        _decide_frame(conn, action_id, keep_frame)
    return n


def _decide_frame(conn, action_id, keep_frame):
    """Apply the keep/discard decision to a pending frame (no-op if none)."""
    r = conn.execute(
        "SELECT id, npy_path, png_path FROM frames WHERE action_id=?", (action_id,)
    ).fetchone()
    if r is None:
        return
    if keep_frame:
        conn.execute(
            "UPDATE frames SET kept=1, decided_at=? WHERE id=?",
            (beamlog.now_iso(), r["id"]),
        )
    else:
        import beamlog_frames  # lazy; only needed to unlink files
        beamlog_frames.discard(r["npy_path"], r["png_path"])
        conn.execute(
            "UPDATE frames SET kept=0, decided_at=?, npy_path=NULL, png_path=NULL WHERE id=?",
            (beamlog.now_iso(), r["id"]),
        )


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
        elif path == "/api/frame":
            qs = parse_qs(urlparse(self.path).query)
            try:
                action_id = int(qs["id"][0])
            except (KeyError, ValueError, IndexError):
                self._send(400, json.dumps({"error": "bad id"}))
                return
            png, err = frame_png(action_id)
            if png is None:
                self._send(404, json.dumps({"error": err}))
            else:
                self._send(200, png, "image/png")
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
                bool(body.get("keep_frame")),
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
<title>beamlog</title>
<style>
  body { font: 13px/1.45 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0; color: #111; background: #fff; }
  .wrap { max-width: 920px; margin: 0 auto; padding: 10px 14px 40px; }
  header { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
           border-bottom: 1px solid #ddd; padding-bottom: 7px; position: sticky; top: 0; background: #fff; }
  h1 { font-size: 13px; margin: 0; font-weight: 700; }
  .meta { color: #666; }
  .count { margin-left: auto; font-weight: 700; }
  .hint { color: #999; font-size: 11px; width: 100%; margin-top: 2px; }
  kbd { font-family: ui-monospace, monospace; background: #f1f1f1; border: 1px solid #ccc;
        border-radius: 3px; padding: 0 4px; font-size: 10px; }
  .row { border: 1px solid #ddd; border-radius: 4px; padding: 6px 8px; margin-top: 6px; }
  .row.new { border-color: #6aa3e0; }
  .head { display: flex; align-items: baseline; gap: 6px; }
  .id { color: #999; }
  .cmd { font-family: ui-monospace, Menlo, Consolas, monospace; word-break: break-all; }
  .tog { color: #06c; cursor: pointer; font-size: 11px; user-select: none; }
  .when { color: #bbb; font-size: 11px; margin-left: auto; }
  .fields { display: flex; gap: 6px; margin-top: 5px; }
  .fields input { font: inherit; padding: 3px 6px; border: 1px solid #ccc; border-radius: 3px; }
  .fields input.why { flex: 1.3; }
  .fields input.obs { flex: 1; }
  .fields input:focus { outline: none; border-color: #06c; }
  button { font: inherit; padding: 3px 9px; border: 1px solid #bbb; background: #f6f6f6;
           border-radius: 3px; cursor: pointer; }
  button:hover { background: #ececec; }
  pre.out { font-family: ui-monospace, monospace; font-size: 11px; color: #444; background: #f7f7f7;
            border: 1px solid #eee; border-radius: 3px; padding: 6px; margin: 5px 0 0;
            white-space: pre-wrap; max-height: 170px; overflow: auto; }
  .frame { display: flex; align-items: center; gap: 8px; margin-top: 5px; }
  .frame img { max-height: 80px; max-width: 120px; border: 1px solid #ddd; border-radius: 3px;
               cursor: zoom-in; background: #f7f7f7; }
  .frame img.big { max-height: 420px; max-width: 100%; cursor: zoom-out; }
  .frame label { display: flex; align-items: center; gap: 4px; cursor: pointer; user-select: none; }
  .frame .fmeta { color: #999; font-size: 11px; }
  .frame .ferr { color: #b00; font-size: 11px; }
  .empty { color: #888; padding: 22px; text-align: center; }
  .reviewed { margin-top: 18px; border-top: 1px solid #ddd; padding-top: 7px; }
  .reviewed h2 { font-size: 11px; text-transform: uppercase; letter-spacing: .03em; color: #999; margin: 0 0 3px; }
  .rev { font-size: 12px; color: #555; padding: 1px 0; }
  .rev .rc { font-family: ui-monospace, monospace; color: #333; }
  .rev .muted { color: #aaa; }
</style></head>
<body><div class="wrap">
  <header>
    <h1>beamlog</h1>
    <span class="meta" id="meta"></span>
    <span class="count" id="count"></span>
    <span class="hint"><kbd>Enter</kbd> save &amp; next row · <kbd>Esc</kbd> skip · click <b>output</b> to expand</span>
  </header>
  <div id="backlog"></div>
  <div class="reviewed" id="reviewed"></div>
</div>
<script>
const known = new Set();          // ids currently shown in the backlog
const backlog = document.getElementById('backlog');

function el(tag, cls, txt){ const e = document.createElement(tag); if(cls) e.className = cls; if(txt != null) e.textContent = txt; return e; }

function makeRow(it){
  const row = el('div', 'row new'); row.dataset.id = it.id;
  setTimeout(() => row.classList.remove('new'), 1500);

  const head = el('div', 'head');
  head.append(el('span', 'id', '#' + it.id), el('span', 'cmd', it.command));
  let pre = null;
  if(it.output){
    pre = el('pre', 'out'); pre.hidden = true; pre.textContent = it.output;
    const tog = el('span', 'tog', 'output');
    tog.onclick = () => { pre.hidden = !pre.hidden; };
    head.append(tog);
  }
  head.append(el('span', 'when', (it.created_at || '').replace('T', ' ')));
  row.append(head);

  const f = el('div', 'fields');
  const why = el('input', 'why'); why.placeholder = 'why…'; if(it.reasoning) why.value = it.reasoning;
  const obs = el('input', 'obs'); obs.placeholder = 'observation…'; if(it.observation) obs.value = it.observation;
  for(const inp of [why, obs]){
    inp.addEventListener('keydown', ev => {
      if(ev.key === 'Enter'){ ev.preventDefault(); doReview(it.id, false); }
      else if(ev.key === 'Escape'){ ev.preventDefault(); doReview(it.id, true); }
    });
  }
  const save = el('button', null, 'save'); save.onclick = () => doReview(it.id, false);
  const skip = el('button', null, 'skip'); skip.onclick = () => doReview(it.id, true);
  f.append(why, obs, save, skip);
  row.append(f);
  if(pre) row.append(pre);

  // detector frame (optional): thumbnail + a "keep frame" checkbox, default off.
  if(it.frame){
    const fr = el('div', 'frame');
    if(it.frame.error){
      fr.append(el('span', 'ferr', 'frame: ' + it.frame.error));
    } else if(it.frame.present){
      const img = el('img'); img.src = '/api/frame?id=' + it.id; img.alt = 'detector frame';
      img.onclick = () => img.classList.toggle('big');
      const keep = el('input'); keep.type = 'checkbox'; keep.className = 'keepframe';
      const lab = el('label'); lab.append(keep, el('span', null, 'keep frame'));
      const dims = (it.frame.width ? it.frame.width + '×' + it.frame.height : '');
      fr.append(img, lab);
      if(dims) fr.append(el('span', 'fmeta', dims));
    }
    if(fr.childNodes.length) row.append(fr);
  }
  return row;
}

async function doReview(id, skip){
  const row = backlog.querySelector('.row[data-id="' + id + '"]');
  if(!row) return;
  const why = row.querySelector('.why').value;
  const obs = row.querySelector('.obs').value;
  const keepEl = row.querySelector('.keepframe');
  const keep_frame = keepEl ? keepEl.checked : false;   // default discard
  const next = row.nextElementSibling;
  const state = await (await fetch('/api/review', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({id, reasoning: skip ? '' : why, observation: skip ? '' : obs, skip, keep_frame})
  })).json();
  known.delete(id);
  row.remove();
  updateCount(state); updateReviewed(state); showEmpty();
  const focus = (next && next.classList.contains('row')) ? next : backlog.querySelector('.row');
  if(focus) focus.querySelector('.why').focus();
}

function updateCount(s){
  const c = document.getElementById('count');
  if(!s.experiment){ c.textContent = ''; return; }
  let t = s.remaining + ' to review';
  if(s.remaining > (s.items ? s.items.length : 0)) t += ' (showing ' + s.items.length + ')';
  c.textContent = s.remaining ? t : 'all caught up';
}

function updateReviewed(s){
  const r = document.getElementById('reviewed');
  const list = s.recent || [];
  r.innerHTML = list.length ? '<h2>recently reviewed</h2>' + list.map(x =>
    '<div class="rev"><span class="rc">#' + x.id + ' ' + escHtml(x.command) + '</span>' +
    (x.reasoning ? ' — ' + escHtml(x.reasoning) : ' <span class="muted">(skipped)</span>') + '</div>'
  ).join('') : '';
}

function escHtml(s){ const d = el('span'); d.textContent = s || ''; return d.innerHTML; }

function showEmpty(){
  const has = backlog.querySelector('.row');
  let e = document.getElementById('emptymsg');
  if(has){ if(e) e.remove(); return; }
  if(!e){ e = el('div', 'empty'); e.id = 'emptymsg'; e.textContent = 'Nothing to review — new commands appear here automatically.'; backlog.append(e); }
}

function apply(state, full){
  const meta = document.getElementById('meta');
  if(!state.experiment){
    meta.textContent = ''; updateCount(state); backlog.innerHTML = ''; known.clear();
    backlog.innerHTML = '<div class="empty">No experiment yet — create one with <code>bl experiment …</code></div>';
    updateReviewed(state); return;
  }
  const e = state.experiment;
  meta.textContent = [e.user, e.material, e.technique, e.goal && ('· ' + e.goal)].filter(Boolean).join('  ·  ');
  updateCount(state); updateReviewed(state);
  if(full){ backlog.innerHTML = ''; known.clear(); }
  for(const it of (state.items || [])){
    if(!known.has(it.id)){ known.add(it.id); backlog.append(makeRow(it)); }
  }
  showEmpty();
}

async function poll(){ try { apply(await (await fetch('/api/queue')).json(), false); } catch(e){} }

(async () => {
  apply(await (await fetch('/api/queue')).json(), true);
  const first = backlog.querySelector('.row');
  if(first) first.querySelector('.why').focus();
  setInterval(poll, 3000);   // pick up newly-tailed actions without disturbing in-progress rows
})();
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
