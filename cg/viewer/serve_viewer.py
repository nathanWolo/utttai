"""Local viewer for crossfish vs uttt.ai matches (standard library only).

Serves the dashboard (/, /api/dashboard: parsed training and match logs), the
game viewer (/games), /api/matches (every *.live directory under utttai/ and
utttai/train/, newest first) and /api/state?match=<name>, which merges one match's
feed written by bench_vs_utttai.py (<log>.live/summary.json and g<id>.json). Without
?match it follows the most recently started match, so it can be started before a
match and picks the match up when it begins. Finished matches stay on disk and
replay from their files.

usage: python serve_viewer.py [port=8765] [live_dir]
"""
import json
import sys

import dashboard_data
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOTS = [HERE.parent, HERE.parent / "train"]
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
FIXED = Path(sys.argv[2]) if len(sys.argv) > 2 else None


def all_dirs():
    if FIXED:
        return [FIXED]
    dirs = [d for r in ROOTS for d in r.glob("*.live") if (d / "summary.json").exists()]
    return sorted(dirs, key=lambda d: (read(d / "summary.json") or {}).get("started", 0), reverse=True)


def live_dir(name=None):
    dirs = all_dirs()
    if name:
        return next((d for d in dirs if d.name == name), None)
    return dirs[0] if dirs else None


def matches():
    out = []
    for d in all_dirs():
        sm = read(d / "summary.json") or {}
        out.append(dict(name=d.name, title=sm.get("title"), started=sm.get("started"), done=sm.get("done"),
                        total=sm.get("total"), finished=sm.get("finished"), cg=sm.get("cg")))
    return out


def read(path):
    for _ in range(3):  # a writer may be mid-replace
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None


def state(name=None):
    d = live_dir(name)
    if d is None:
        return {"summary": None, "games": [], "source": None}
    games = [g for g in (read(f) for f in sorted(d.glob("g*.json"))) if g]
    return {"summary": read(d / "summary.json"), "games": games, "source": d.name}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        url = urlparse(self.path)
        if url.path == "/api/state":
            body = json.dumps(state((parse_qs(url.query).get("match") or [None])[0])).encode()
            ctype = "application/json"
        elif url.path == "/api/dashboard":
            body = json.dumps(dashboard_data.dashboard()).encode()
            ctype = "application/json"
        elif url.path == "/api/matches":
            body = json.dumps(matches()).encode()
            ctype = "application/json"
        elif url.path in ("/", "/index.html", "/dashboard"):
            body = (HERE / "dashboard.html").read_bytes()
            ctype = "text/html; charset=utf-8"
        elif url.path == "/analysis":
            body = (HERE / "analysis.html").read_bytes()
            ctype = "text/html; charset=utf-8"
        elif url.path == "/api/analysis":
            body = (HERE.parent / "analysis" / "analysis_90ms.json").read_bytes()
            ctype = "application/json"
        elif url.path == "/games":
            body = (HERE / "viewer.html").read_bytes()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"viewer on http://localhost:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
