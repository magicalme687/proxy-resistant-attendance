"""
Teacher-station BLE logger (Raspberry Pi / Linux laptop with BLE 5.0 adapter).

Scans continuously and logs every advertisement that carries the protocol's
manufacturer data (company ID 0xFFFF in the prototype; use your assigned ID in
production). Each row: host receive time (ms, CLOCK_REALTIME synced by chrony),
device address, RSSI, payload hex.

Mode-I chirps carry Truncate8(HMAC(K_dev, N_s)); analysis/analyze_latency.py joins
them to nonce issue times from nonce_server.py.

Usage:
    pip install bleak
    python station_logger.py --out ../data/station_rx.csv --company 0xFFFF
"""
import argparse
import asyncio
import csv
import time

from bleak import BleakScanner


async def main(out, company):
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if f.tell() == 0:
        w.writerow(["t_rx_ms", "address", "rssi_dbm", "payload_hex"])

    def cb(device, adv):
        data = adv.manufacturer_data.get(company)
        if data is None:
            return
        w.writerow([int(time.time() * 1000), device.address, adv.rssi, data.hex()])
        f.flush()

    scanner = BleakScanner(detection_callback=cb, scanning_mode="active")
    await scanner.start()
    print(f"Logging company 0x{company:04X} advertisements to {out}. Ctrl-C to stop.")
    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await scanner.stop()
        f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--company", default="0xFFFF")
    a = ap.parse_args()
    asyncio.run(main(a.out, int(a.company, 16)))
