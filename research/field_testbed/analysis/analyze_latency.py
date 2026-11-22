"""
Campaign 2 analysis: genuine vs. relay chirp RTT.

Joins nonce issue times (nonce_server.py) to station receptions (station_logger.py)
by recomputing each trial's expected 4-byte tag = HMAC-SHA256(K_dev, nonce)[:4]
with the per-device TEST key in devices.csv (column test_key_hex).
RTT = first station reception of the tag - issue time.

Input : data/nonce_issue.csv, data/station_rx.csv, data/devices.csv
Output: results/latency_trials.csv, results/latency_summary.csv,
        research/experiments/results/testbed_latency_params.json (read by exp4)
"""
import hashlib
import hmac
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
EXP_RES = os.path.join(HERE, "..", "..", "experiments", "results")
os.makedirs(OUT, exist_ok=True)
TIMEOUT_MS = 5000


def tag(key_hex, nonce_hex):
    return hmac.new(bytes.fromhex(key_hex), nonce_hex.encode(), hashlib.sha256).digest()[:4].hex()


def main():
    iss = pd.read_csv(os.path.join(DATA, "nonce_issue.csv"), dtype={"nonce": str, "device_id": str})
    rx = pd.read_csv(os.path.join(DATA, "station_rx.csv"), dtype={"payload_hex": str})
    dev = pd.read_csv(os.path.join(DATA, "devices.csv"), dtype=str).set_index("device_id")
    rx["tag"] = rx.payload_hex.str[:8].str.lower()
    first = rx.sort_values("t_rx_ms").groupby("tag").first()

    rows = []
    for _, t in iss.iterrows():
        tg = tag(dev.loc[t.device_id, "test_key_hex"], t.nonce)
        hit = first.loc[tg] if tg in first.index else None
        rtt = float(hit.t_rx_ms - t.t_issue_ms) if hit is not None else np.nan
        if not np.isnan(rtt) and (rtt < 0 or rtt > TIMEOUT_MS):
            rtt = np.nan
        rows.append({**t.to_dict(), "rtt_ms": rtt,
                     "rssi_station_dbm": float(hit.rssi_dbm) if hit is not None else np.nan})
    tr = pd.DataFrame(rows)
    tr.to_csv(os.path.join(OUT, "latency_trials.csv"), index=False)

    summ = (tr.groupby(["kind", "network"]).rtt_ms
              .describe(percentiles=[.5, .9, .95, .99]).round(1))
    summ["lost_pct"] = tr.groupby(["kind", "network"]).rtt_ms.apply(lambda s: round(100 * s.isna().mean(), 2))
    summ.to_csv(os.path.join(OUT, "latency_summary.csv"))
    print(summ.to_string())

    g = tr[tr.kind == "genuine"].rtt_ms.dropna()
    ws = tr[tr.kind == "websocket_relay"].rtt_ms.dropna()
    ms = tr[tr.kind == "messaging_relay"].rtt_ms.dropna()
    params = {}
    if len(g) >= 30:
        params.update({"genuine_rtt_model": "empirical", "genuine_rtt_samples_ms": g.round(1).tolist(),
                       "genuine_rtt_median_ms": float(g.median()), "genuine_rtt_p99_ms": float(g.quantile(.99))})
    if len(ws) >= 30:
        params.update({"relay_rtt_model": "empirical", "relay_rtt_samples_ms": ws.round(1).tolist()})
    if len(ms) >= 30:
        params["relay_rtt_normal_ms"] = [float(ms.mean()), float(ms.std(ddof=1))]
    if params:
        os.makedirs(EXP_RES, exist_ok=True)
        with open(os.path.join(EXP_RES, "testbed_latency_params.json"), "w") as f:
            json.dump(params, f, indent=1)
        print("wrote testbed_latency_params.json")


if __name__ == "__main__":
    main()
