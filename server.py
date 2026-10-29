#!/usr/bin/env python3
"""
server.py: live operator console server (standard library + your existing deps only).

    python server.py                 -> http://127.0.0.1:8000
    python server.py --port 9000 --host 0.0.0.0 --reset

What it does
  - Holds a LIVE copy of telemetry (state/live_data/) so your original data/ is never modified.
  - Re-runs the full pipeline (change detection, forecast, attribution, cause, decision,
    audit log) whenever data changes, then pushes the update to every open browser (SSE).
  - Read-only toward the facility: it only writes inside this folder (state/, audit/).

API
  GET  /api/results      latest forecasts + cause + decision (+ meta)
  GET  /api/events       server-sent events; a message arrives after every recompute
  GET  /api/audit        ?rack=rack-A1&resource=storage_pct&limit=5   recent forecast runs
  GET  /api/backtest     last backtest output (reports/backtest_output.json)
  GET  /api/checklists   shared operator checklist progress
  POST /api/ingest       {"rows":[{"date","rack_id","tenant_id","resource","value"}, ...]}
  POST /api/advance      {"days":7, "surge":{"tenant":"tenant-helios","resource":"storage_pct","factor":3}}
  POST /api/live         {"on":true,"interval":15}   auto-advance one simulated day per interval
  POST /api/checklists   {"key":"rack-A1|storage_pct","done":[0,2]}
  POST /api/reset        restore the original dataset
"""
import argparse
import json
import shutil
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from pipeline import run_pipeline

BASE = Path(__file__).resolve().parent
SRC, STATE, AUDIT = BASE / "data", BASE / "state", BASE / "audit"
LIVE = STATE / "live_data"
CHECKS = STATE / "checklists.json"
HISTORY_DAYS = 150
MAX_BODY = 5_000_000
COLS = ["date", "rack_id", "tenant_id", "resource", "value"]

cond = threading.Condition(threading.RLock())
S = {"version": 0, "payload": None, "df": None, "limits": None,
     "auto": False, "interval": 15, "last_auto": 0.0, "busy": False}


# ---------------------------------------------------------------- state
def ensure_source_data():
    """If data/ is missing or incomplete, generate the synthetic dataset instead of crashing."""
    need = ["telemetry.csv", "capacity_limits.csv"]
    if all((SRC / n).exists() for n in need):
        return
    print(f"Source data not found in {SRC}. Generating synthetic telemetry there...")
    try:
        from data_generator import generate_telemetry
        generate_telemetry(SRC)
    except ImportError:
        raise SystemExit(f"Missing {SRC}\\telemetry.csv and data_generator.py. Put the four CSV files "
                         f"(telemetry, capacity_limits, incident_log, ground_truth) in {SRC} and retry.")


def init_data(reset: bool = False):
    ensure_source_data()
    LIVE.mkdir(parents=True, exist_ok=True)
    if reset or not (LIVE / "telemetry.csv").exists():
        shutil.copy(SRC / "telemetry.csv", LIVE / "telemetry.csv")
    shutil.copy(SRC / "capacity_limits.csv", LIVE / "capacity_limits.csv")
    S["df"] = pd.read_csv(LIVE / "telemetry.csv", parse_dates=["date"])
    S["limits"] = set(pd.read_csv(LIVE / "capacity_limits.csv")["resource"])


def recompute():
    """Persist live data, run the whole pipeline, publish a new version."""
    with cond:
        t0 = time.time()
        S["df"] = S["df"].sort_values(["rack_id", "tenant_id", "resource", "date"]).reset_index(drop=True)
        S["df"].to_csv(LIVE / "telemetry.csv", index=False)
        out = run_pipeline(LIVE, AUDIT)
        for r in out["results"]:
            r["history"] = {k: v[-HISTORY_DAYS:] for k, v in r["history"].items()}
        df = S["df"]
        pairs = df.groupby("tenant_id")["resource"].unique()
        out["meta"] = {
            "version": S["version"] + 1,
            "data_through": str(df["date"].max().date()),
            "n_rows": int(len(df)),
            "run_ms": int((time.time() - t0) * 1000),
            "auto": S["auto"], "interval": S["interval"],
            "tenants": {t: sorted(map(str, rs)) for t, rs in pairs.items()},
        }
        S["payload"] = out
        S["version"] += 1
        cond.notify_all()


def advance(days: int, surge: dict | None = None):
    """Simulated live feed: extend every series by `days` using its recent trend and noise.
    Stands in for a real collector; real data goes through /api/ingest instead."""
    rng = np.random.default_rng()
    df = S["df"]
    if surge:
        ok = ((df.tenant_id == surge.get("tenant")) & (df.resource == surge.get("resource"))).any()
        if not ok:
            raise ValueError("That tenant has no series for that resource.")
    new = []
    start = df["date"].max()
    for (rack, ten, res), g in df.groupby(["rack_id", "tenant_id", "resource"]):
        v = g.sort_values("date")["value"].to_numpy()[-90:]
        x = np.arange(len(v))
        slope, icpt = np.polyfit(x, v, 1)
        sd = float((v - (icpt + slope * x)).std()) or 0.01
        cur = float(v[-1])
        if surge and ten == surge["tenant"] and res == surge["resource"]:
            slope = max(slope, 1e-3 * abs(cur)) * float(surge.get("factor", 3))
        floor = 15.0 if res == "thermal_c" else 0.0
        for d in range(1, days + 1):
            cur = max(floor, cur + slope + rng.normal(0, sd))
            new.append({"date": start + pd.Timedelta(days=d), "rack_id": rack,
                        "tenant_id": ten, "resource": res, "value": round(cur, 4)})
    with cond:
        S["df"] = pd.concat([S["df"], pd.DataFrame(new)], ignore_index=True)
        recompute()


def ingest(rows: list):
    clean = []
    for i, r in enumerate(rows):
        try:
            if r["resource"] not in S["limits"]:
                raise ValueError(f"unknown resource '{r['resource']}'")
            clean.append({"date": pd.Timestamp(r["date"]).normalize(), "rack_id": str(r["rack_id"]),
                          "tenant_id": str(r["tenant_id"]), "resource": r["resource"],
                          "value": float(r["value"])})
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"row {i}: {e}")
    if not clean:
        raise ValueError("no rows")
    with cond:
        df = pd.concat([S["df"], pd.DataFrame(clean)], ignore_index=True)
        S["df"] = df.drop_duplicates(["date", "rack_id", "tenant_id", "resource"], keep="last")
        recompute()
    return len(clean)


def auto_loop():
    while True:
        time.sleep(1)
        if S["auto"] and time.time() - S["last_auto"] >= S["interval"]:
            S["last_auto"] = time.time()
            try:
                advance(1)
            except Exception as e:  # keep the feed alive
                print("auto-advance failed:", e)


def read_checks():
    try:
        return json.loads(CHECKS.read_text())
    except (OSError, ValueError):
        return {}


def audit_tail(rack, resource, limit):
    path, rows = AUDIT / "forecast_audit_log.jsonl", deque(maxlen=limit)
    if path.exists():
        with open(path) as f:
            for line in f:
                if f'"{rack}"' in line and f'"{resource}"' in line:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if rec["rack_id"] == rack and rec["resource"] == resource:
                        fc = rec["forecast"]
                        rows.append({
                            "run": rec["run_timestamp_utc"], "data_through": rec["input_data_window"]["end_date"],
                            "n_points": rec["input_data_window"]["n_points"],
                            "data_hash": rec["input_data_window"]["data_hash"],
                            "eta_days": (fc.get("to_safety_margin") or {}).get("eta_days"),
                            "status": fc.get("status")})
    return list(rows)[::-1]


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        pass

    def _send(self, body: bytes, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, default=str).encode(), code=code)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        if u.path in ("/", "/index.html"):
            self._send((BASE / "console_template.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/api/health":
            self._json({"ok": True, "version": S["version"]})
        elif u.path == "/api/results":
            self._json(S["payload"])
        elif u.path == "/api/audit":
            self._json(audit_tail(q.get("rack", ""), q.get("resource", ""), min(int(q.get("limit", 5)), 50)))
        elif u.path == "/api/backtest":
            p = BASE / "reports" / "backtest_output.json"
            self._send(p.read_bytes() if p.exists() else b"{}")
        elif u.path == "/api/checklists":
            self._json(read_checks())
        elif u.path == "/api/events":
            self._events()
        else:
            self._json({"error": "not found"}, 404)

    def _events(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        seen = -1
        try:
            while True:
                with cond:
                    if S["version"] == seen:
                        cond.wait(15)
                    v = S["version"]
                msg = f"data: {v}\n\n" if v != seen else ": ping\n\n"
                seen = v
                self.wfile.write(msg.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_POST(self):
        u = urlparse(self.path)
        try:
            b = self._body()
            if u.path == "/api/ingest":
                n = ingest(b.get("rows", []))
                self._json({"ok": True, "rows": n, "version": S["version"]})
            elif u.path == "/api/advance":
                days = int(b.get("days", 1))
                if not 1 <= days <= 90:
                    raise ValueError("days must be 1 to 90")
                advance(days, b.get("surge"))
                self._json({"ok": True, "version": S["version"]})
            elif u.path == "/api/live":
                S["auto"] = bool(b.get("on"))
                S["interval"] = max(5, int(b.get("interval", 15)))
                S["last_auto"] = time.time()
                with cond:
                    S["payload"]["meta"].update(auto=S["auto"], interval=S["interval"])
                    S["version"] += 1
                    cond.notify_all()
                self._json({"ok": True, "auto": S["auto"], "interval": S["interval"]})
            elif u.path == "/api/checklists":
                c = read_checks()
                c[str(b["key"])] = [int(i) for i in b.get("done", [])]
                STATE.mkdir(exist_ok=True)
                CHECKS.write_text(json.dumps(c))
                self._json({"ok": True})
            elif u.path == "/api/reset":
                S["auto"] = False
                with cond:
                    init_data(reset=True)
                    recompute()
                self._json({"ok": True, "version": S["version"]})
            else:
                self._json({"error": "not found"}, 404)
        except (ValueError, KeyError, TypeError) as e:
            self._json({"ok": False, "error": str(e)}, 400)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reset", action="store_true", help="discard live data and start from data/")
    a = ap.parse_args()
    init_data(a.reset)
    recompute()
    threading.Thread(target=auto_loop, daemon=True).start()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    print(f"Capacity console running at http://{a.host}:{a.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
