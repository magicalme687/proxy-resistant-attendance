"""
A10 attacker relay (for authorised red-team testing only).

The absent phone connects as role=absent and the in-room accomplice phone as
role=accomplice. Any message the absent phone sends is forwarded immediately to
the accomplice, which advertises it as a BLE chirp. Host this on a public VM (or
a phone hotspot) so the path really crosses the cellular network.

Usage:
    pip install aiohttp
    python relay_server.py --port 9000
    ws://HOST:9000/ws?role=absent   |   ws://HOST:9000/ws?role=accomplice
"""
import argparse
import time

from aiohttp import WSMsgType, web

peers = {}


async def ws(req):
    role = req.query.get("role")
    sock = web.WebSocketResponse()
    await sock.prepare(req)
    peers[role] = sock
    async for msg in sock:
        if msg.type == WSMsgType.TEXT and role == "absent" and "accomplice" in peers:
            await peers["accomplice"].send_str(msg.data)
            print(f"{int(time.time()*1000)} forwarded {msg.data[:32]}")
    peers.pop(role, None)
    return sock


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    app = web.Application()
    app.router.add_get("/ws", ws)
    web.run_app(app, port=a.port)
