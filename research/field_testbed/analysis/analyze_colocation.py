"""
Campaign 3 analysis: co-location detector (|dRSSI| < 1.5 dB within 213 ms).

Input : data/colocation.csv (label in {relay, genuine_pair})
Output: results/colocation_summary.json; updates p_coloc_detect / coloc_fp_rate in
        research/experiments/results/testbed_latency_params.json
Also sweeps the dRSSI threshold to give the detector's ROC.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
PARAMS = os.path.join(HERE, "..", "..", "experiments", "results", "testbed_latency_params.json")
os.makedirs(OUT, exist_ok=True)


def flag(df, thr_db=1.5, win_ms=213):
    return (np.abs(df.rssi_a_dbm - df.rssi_b_dbm) < thr_db) & (df.dt_ms.abs() < win_ms)


def cp(k, n):
    return (stats.beta.ppf(.025, k, n - k + 1) if k else 0.0,
            stats.beta.ppf(.975, k + 1, n - k) if k < n else 1.0)


def main():
    df = pd.read_csv(os.path.join(DATA, "colocation.csv"))
    pos, neg = df[df.label == "relay"], df[df.label == "genuine_pair"]
    tp, fp = int(flag(pos).sum()), int(flag(neg).sum())
    res = {"p_detect": tp / len(pos), "p_detect_ci": cp(tp, len(pos)), "n_relay": len(pos),
           "fp_rate": fp / len(neg), "fp_ci": cp(fp, len(neg)), "n_genuine_pairs": len(neg),
           "roc": [{"thr_db": t, "tpr": float(flag(pos, t).mean()), "fpr": float(flag(neg, t).mean())}
                   for t in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]]}
    json.dump(res, open(os.path.join(OUT, "colocation_summary.json"), "w"), indent=2)
    p = json.load(open(PARAMS)) if os.path.exists(PARAMS) else {}
    p.update({"p_coloc_detect": res["p_detect"], "coloc_fp_rate": res["fp_rate"]})
    json.dump(p, open(PARAMS, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
