"""
Campaign 1 analysis: measured channel model and boundary-gate performance.

Input : testbed/data/walktest.csv, testbed/data/devices.csv
Output: testbed/results/
  channel_fit.json            RSSI0, gamma, sigma (LOS, in-room), per-device offsets
  gate_inroom_frr.csv         genuine FRR per device/posture at gate radii {4,6,8,10} m
  gate_leakage.csv            outside-room leakage per partition at the same radii,
                              RF-only and RF + optical-LOS (los_to_screen==0 blocks)
  table_walktest.tex          LaTeX rows for the paper
Per-point decisions use the median RSSI over the 30 s window (what the app would
report from a short scan burst); per-sample rates are also given.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
RADII = [4, 6, 8, 10]


def cp(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return lo, hi


def main():
    df = pd.read_csv(os.path.join(DATA, "walktest.csv"))
    pts = (df.groupby(["session_id", "room_id", "device_id", "posture", "point_id",
                       "inside_room", "partition_type", "distance_m", "los_to_screen"])
             .rssi_dbm.agg(["median", "count"]).reset_index().rename(columns={"median": "rssi"}))

    # 1) channel fit on in-room LOS points, reference device(s) in hand
    los = pts[(pts.inside_room == 1) & (pts.distance_m >= 1)]
    x = 10 * np.log10(los.distance_m.values)
    slope, icpt, r, _, _ = stats.linregress(x, los.rssi.values)
    gamma, rssi0 = -slope, icpt
    resid = los.rssi.values - (icpt + slope * x)
    offsets = (los.assign(res=resid).groupby("device_id").res.mean().round(2).to_dict())
    sigma_dev_corrected = float(np.std(resid - los.device_id.map(offsets).values, ddof=1))
    fit = {"rssi0_dbm": round(float(rssi0), 2), "gamma": round(float(gamma), 3),
           "sigma_db": round(float(np.std(resid, ddof=1)), 2),
           "sigma_db_after_device_offset": round(sigma_dev_corrected, 2),
           "r2": round(float(r ** 2), 3),
           # per-sample shadowing (what a single short scan sees) vs. per-point median above
           "sigma_db_per_sample": round(float(np.std(
               df[(df.inside_room == 1) & (df.distance_m >= 1)].pipe(
                   lambda d: d.rssi_dbm - (icpt + slope * 10 * np.log10(d.distance_m))), ddof=1)), 2), "n_points": int(len(los)),
           "device_offsets_db": offsets,
           "posture_offsets_db": los.assign(res=resid).groupby("posture").res.mean().round(2).to_dict()}
    json.dump(fit, open(os.path.join(OUT, "channel_fit.json"), "w"), indent=2)

    def thr(R):  # gate threshold using the *fitted* model
        return rssi0 - 10 * gamma * np.log10(R)

    rows_frr, rows_leak = [], []
    inside = pts[pts.inside_room == 1]
    for R in RADII:
        g = inside[inside.distance_m <= R]
        for (dev, post), s in g.groupby(["device_id", "posture"]):
            k = int((s.rssi < thr(R)).sum()); n = len(s)
            lo, hi = cp(k, n)
            rows_frr.append({"gate_m": R, "device_id": dev, "posture": post, "n": n,
                             "frr_pct": round(100 * k / n, 2), "ci_lo": round(100 * lo, 2), "ci_hi": round(100 * hi, 2)})
    outside = pts[pts.inside_room == 0]
    for R in RADII:
        for part, s in outside.groupby("partition_type"):
            rf = s.rssi >= thr(R)
            both = rf & (s.los_to_screen == 1)
            n = len(s)
            for label, acc in [("rf_only", rf), ("rf_and_optical", both)]:
                k = int(acc.sum()); lo, hi = cp(k, n)
                rows_leak.append({"gate_m": R, "partition": part, "gating": label, "n": n,
                                  "leak_pct": round(100 * k / n, 2), "ci_lo": round(100 * lo, 2), "ci_hi": round(100 * hi, 2)})
    frr = pd.DataFrame(rows_frr); leak = pd.DataFrame(rows_leak)
    frr.to_csv(os.path.join(OUT, "gate_inroom_frr.csv"), index=False)
    leak.to_csv(os.path.join(OUT, "gate_leakage.csv"), index=False)

    with open(os.path.join(OUT, "table_walktest.tex"), "w") as f:
        for _, r in leak[leak.gate_m.isin([4, 6, 10])].iterrows():
            f.write(f"{r.partition} & {r.gate_m} m & {r.gating.replace('_', ' ')} & {r.n} & "
                    f"{r.leak_pct:.2f}\\% [{r.ci_lo:.2f}, {r.ci_hi:.2f}] \\\\\n")
    print(json.dumps(fit, indent=1))
    print(leak.to_string(index=False))


if __name__ == "__main__":
    main()
