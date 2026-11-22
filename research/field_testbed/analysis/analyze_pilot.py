"""
Campaign 4 analysis: classroom pilot.

Input : data/pilot_attempts.csv, data/sus.csv
Output: results/pilot_summary.json, results/pilot_gate_failures.csv
Reports per-mode end-to-end GAR (first attempt and within 3 attempts), which gate
rejected genuine students, time-to-mark (median, p90), red-team attack acceptance
per attack type with Clopper-Pearson CIs, and the SUS score (mean, SD).
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


def cp(k, n):
    return [round(100 * (stats.beta.ppf(.025, k, n - k + 1) if k else 0.0), 2),
            round(100 * (stats.beta.ppf(.975, k + 1, n - k) if k < n else 1.0), 2)]


def sus_score(row):
    odd = sum(row[f"q{i}"] - 1 for i in (1, 3, 5, 7, 9))
    even = sum(5 - row[f"q{i}"] for i in (2, 4, 6, 8, 10))
    return 2.5 * (odd + even)


def main():
    a = pd.read_csv(os.path.join(DATA, "pilot_attempts.csv"))
    if "is_attack" not in a:
        a["is_attack"] = 0
    a["dur_s"] = (a.t_end_ms - a.t_start_ms) / 1000
    gen = a[a.is_attack == 0].sort_values("t_start_ms")
    out = {}
    for mode, g in gen.groupby("mode"):
        first = g.groupby(["session_id", "student_hash"]).head(1)
        by_student = g.groupby(["session_id", "student_hash"]).head(3).groupby(
            ["session_id", "student_hash"]).accepted.max()
        k1, n1 = int(first.accepted.sum()), len(first)
        k3, n3 = int(by_student.sum()), len(by_student)
        out[mode] = {"gar_first_attempt_pct": round(100 * k1 / n1, 2), "ci": cp(k1, n1), "n": n1,
                     "gar_within_3_pct": round(100 * k3 / n3, 2), "ci3": cp(k3, n3),
                     "time_to_mark_s_median": round(float(g[g.accepted == 1].dur_s.median()), 2),
                     "time_to_mark_s_p90": round(float(g[g.accepted == 1].dur_s.quantile(.9)), 2)}
    gen[gen.accepted == 0].groupby(["mode", "gate_failed"]).size().rename("count").to_csv(
        os.path.join(OUT, "pilot_gate_failures.csv"))
    att = a[a.is_attack == 1]
    out["attacks"] = {f"{m}|{t}": {"accepted": int(s.accepted.sum()), "n": len(s),
                                   "accept_pct": round(100 * s.accepted.mean(), 2), "ci": cp(int(s.accepted.sum()), len(s))}
                      for (m, t), s in att.groupby(["mode", "attack_type"])}
    sp = os.path.join(DATA, "sus.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp).apply(sus_score, axis=1)
        out["sus"] = {"mean": round(float(s.mean()), 1), "sd": round(float(s.std(ddof=1)), 1), "n": len(s)}
    json.dump(out, open(os.path.join(OUT, "pilot_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
