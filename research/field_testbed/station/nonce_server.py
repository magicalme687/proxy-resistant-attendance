"""
Nonce server for latency campaigns.

GET /nonce?device_id=..&kind=genuine|websocket_relay|messaging_relay&network=4g|wifi
returns a fresh 8-byte nonce and logs its issue time (ms). The phone (or relay
chain) must advertise Truncate8(HMAC(K_dev, nonce)) so the station can detect it.
For the testbed, K_dev is a per-device test key in devices.csv (never a real key).

Usage:
    pip install aiohttp
    python nonce_server.py --out ../data/nonce_issue.csv --port 8080
"""
import argparse
import csv
import os
import secrets
import time

from aiohttp import web


def make_app(out):
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["trial_id", "kind", "device_id", "network", "nonce", "t_issue_ms"])
    counter = {"n": 0}

    async def nonce(req):
        q = req.query
        n = secrets.token_hex(8)
        counter["n"] += 1
        tid = f"{int(time.time())}-{counter['n']}"
        t = int(time.time() * 1000)
        w.writerow([tid, q.get("kind", "genuine"), q.get("device_id", ""), q.get("network", ""), n, t])
        f.flush()
        return web.json_response({"trial_id": tid, "nonce": n, "t_issue_ms": t})

    app = web.Application()
    app.router.add_get("/nonce", nonce)
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    web.run_app(make_app(a.out), port=a.port)
