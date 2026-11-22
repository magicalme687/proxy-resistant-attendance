"""
Experiment 4: analytical composition of per-gate pass rates (Table IV).

This is an analytical model, not a Monte-Carlo experiment. Every input is
either (M) measured on real data or (S) produced by our channel simulation, or
(A) an explicit assumption. Each input's provenance is written to
results/exp4_assumptions.csv, which is printed as a table in the paper. The
cascade assumes gate outcomes are independent (stated in the paper).

Threat columns: S2 static messaging relay, S3 live video relay, S4 corridor
behind a solid partition (no line of sight), S4' open doorway / thin panel
(line of sight possible), S5 in-room buddy proxy, S6 presentation attack.
Mean mitigation averages S2..S6 (S4' is reported separately as a known gap).
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def bsi(gar, mit):
    return 2 * gar * mit / (gar + mit) if gar + mit > 0 else 0.0


def main():
    lfw = json.load(open(os.path.join(RES, "real_lfw_facenet_summary.json")))
    pad = json.load(open(os.path.join(RES, "real_celeba_spoof_classifier_summary.json")))
    exp1 = json.load(open(os.path.join(RES, "exp1_metrics_summary.json")))
    part = pd.read_csv(os.path.join(RES, "c1_partition_attenuation_benchmark.csv"))
    p = pad["proposed_calibrated_multispectral_classifier"]

    face_gar = lfw["theta_eer_metrics"]["gar_percent"] / 100
    face_far = lfw["theta_eer_metrics"]["far_percent"] / 100
    pad_gar = 1 - p["eer_operating_point"]["bpcer_pct"] / 100
    pad_mit = 1 - p["eer_operating_point"]["apcer_pct"] / 100
    pad_gar_s = 1 - p["high_security_operating_point"]["bpcer_pct"] / 100
    pad_mit_s = 1 - p["high_security_operating_point"]["apcer_pct"] / 100
    ble_gar = exp1["threat_scenario_metrics"]["A_Legitimate_InRoom"]["proposed_accepted"] / \
        exp1["threat_scenario_metrics"]["A_Legitimate_InRoom"]["total_trials"]
    qr_gar = 0.9895
    leak = lambda name: float(part[(part.partition_type.str.startswith(name)) & (part.gate_radius_m == 10.0)].leakage_rate_pct.iloc[0]) / 100
    s4_concrete_rf = 1 - leak("Reinforced")
    s4_door_rf = 1 - leak("Open Doorway")

    assumptions = [
        ("Face match pass rate (theta_EER)", face_gar, "M", "LFW 6,000 pairs, InceptionResnetV1 + MTCNN"),
        ("Buddy-proxy acceptance (FAR at theta_EER)", face_far, "M", "LFW imposter pairs"),
        ("PAD bona-fide pass (1-BPCER, tau_EER)", pad_gar, "M", "CelebA-Spoof shard 0, 5-fold CV"),
        ("PAD spoof rejection (1-APCER, tau_EER)", pad_mit, "M", "CelebA-Spoof shard 0, 5-fold CV"),
        ("PAD spoof rejection at strict tau", pad_mit_s, "M", "CelebA-Spoof shard 0, 5-fold CV"),
        ("In-room pass rate of BLE gate", ble_gar, "S", "Exp. 1 channel simulation (d up to 8 m)"),
        ("QR decode success rate (optical)", qr_gar, "A", "assumed; to be measured in pilot (Campaign 4)"),
        ("Rejection by RF gate alone, concrete wall, R=10 m", s4_concrete_rf, "S", "Exp. 1 extension, Table VI"),
        ("Rejection by RF gate alone, open doorway, R=10 m", s4_door_rf, "S", "Exp. 1 extension, Table VI"),
        ("Remote relays (S2,S3) cannot obtain the in-room BLE frame", 1.0, "A", "honest client RSSI (Mode-C) / station-measured (Mode-I)"),
        ("Solid partition hides the screen from outside", 1.0, "A", "geometry assumption; verified per room in Campaign 1"),
        ("Spoof acceptance with no PAD stage", 1.0, "A", "conservative: a photo of the enrolled student matches"),
        ("Gate outcomes independent", 1.0, "A", "to be checked against pilot joint pass rate"),
    ]
    pd.DataFrame(assumptions, columns=["input", "value", "type", "source"]).to_csv(
        os.path.join(RES, "exp4_assumptions.csv"), index=False)

    # columns: S2, S3, S4 (solid), S5, S6 ; plus S4' (doorway) reported separately
    rows = []

    def add(name, gar, s2, s3, s4, s5, s6, s4door):
        mits = [s2, s3, s4, s5, s6]
        m = float(np.mean(mits))
        rows.append({"architecture": name, "gar": round(gar, 4), "s2": s2, "s3": s3, "s4": round(s4, 4),
                     "s5": round(s5, 4), "s6": round(s6, 4), "s4_doorway": round(s4door, 4),
                     "mean_mitigation": round(m, 4), "bsi": round(bsi(gar, m), 4)})

    buddy = 1 - face_far
    add("BLE only", ble_gar, 1.0, 1.0, s4_concrete_rf, 0.0, 0.0, s4_door_rf)
    add("Rotating QR only", qr_gar, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    add("Face only (no PAD)", face_gar, 0.0, 0.0, 0.0, buddy, 0.0, 0.0)
    add("BLE + face (no PAD)", ble_gar * face_gar, 1.0, 1.0, s4_concrete_rf, buddy, 0.0, s4_door_rf)
    add("Proposed tri-modal, PAD at EER threshold", qr_gar * ble_gar * face_gar * pad_gar,
        1.0, 1.0, 1.0, buddy, pad_mit, s4_door_rf)
    add("Proposed tri-modal, strict PAD threshold", qr_gar * ble_gar * face_gar * pad_gar_s,
        1.0, 1.0, 1.0, buddy, pad_mit_s, s4_door_rf)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp4_trimodal_ablation.csv"), index=False)
    json.dump({"rows": rows, "independence_assumed": True},
              open(os.path.join(RES, "exp4_trimodal_summary.json"), "w"), indent=2)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
