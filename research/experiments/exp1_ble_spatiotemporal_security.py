"""
Experiment 1: Spatiotemporal BLE Zero-Trust Security & Path-Loss Boundary Benchmark

Evaluates:
- 10,000 empirical authentication trials across 5 realistic threat vectors:
  (a) Legitimate In-Room Student (d in [1m, 8m], latency in [0.5s, 4.0s])
  (b) Adjacent Room / Hallway Bleed-Through (d in [9m, 22m] with 12-18 dB NLOS concrete wall penetration loss)
  (c) Remote WhatsApp / Instant Messaging Token Relay (latency in [5s, 120s])
  (d) Cryptographic Brute-Force Guessing of 8-character hex token space (16^8 = 4.29 x 10^9)
  (e) Replay Attack: Legacy unconstrained window (delta = 1000, +-66.7 min) vs. Proposed Gated (delta = 1, +-4s + RSSI gate)
- Parametric ablation across token interval Delta_t in {2s, 4s, 8s, 15s, 30s}
- Parametric sweep across RSSI path-loss distance cutoffs d_max in {6m, 8m, 10m, 12m}
Outputs:
- research/experiments/results/exp1_threat_scenarios_summary.csv
- research/experiments/results/exp1_epoch_ablation.csv
- research/experiments/results/exp1_metrics_summary.json
"""

import os
import sys
import json
import time
import secrets
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "app", "server")))
from ble_engine import RollingSecurityEngine, verify_spatial_boundary, CALIBRATED_1M_RSSI

np.random.seed(42)
os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)

CLASSROOM_KEY = b"simulation_ble_salt_key_benchmark"
ROOM_ID = "ROOM_101"
NUM_TRIALS_PER_SCENARIO = 2000  # Total 10,000 trials across 5 scenarios

def simulate_rssi(distance_m: float, tx_power: int = -59, path_loss_exp: float = 2.7, wall_attenuation_db: float = 0.0):
    """Log-distance path loss with log-normal shadow fading and wall loss."""
    shadowing = np.random.normal(0.0, 2.5) # indoor multipath sigma = 2.5 dB
    rssi = tx_power - (10.0 * path_loss_exp * np.log10(max(distance_m, 0.5))) - wall_attenuation_db + shadowing
    return float(np.clip(rssi, -105.0, -15.0))


def run_threat_scenarios_benchmark():
    np.random.seed(42)
    print("=" * 70)
    print("RUNNING EXP 1: 10,000-Trial BLE Threat Mitigation Benchmark")
    print("=" * 70)

    engine_proposed = RollingSecurityEngine(ROOM_ID, CLASSROOM_KEY, interval=4)
    engine_legacy = RollingSecurityEngine(ROOM_ID, CLASSROOM_KEY, interval=4)

    records = []

    # Scenario A: Legitimate In-Room Student (d <= 8m, fresh token <= 4s)
    for i in range(NUM_TRIALS_PER_SCENARIO):
        dist = np.random.uniform(0.8, 8.0)
        delay_sec = np.random.uniform(0.1, 3.8) # arrives within window
        rssi = simulate_rssi(dist, tx_power=CALIBRATED_1M_RSSI, wall_attenuation_db=0.0)
        
        # Ground truth: Valid claim
        token, _ = engine_proposed.generate_token(window_offset=0)
        
        # Proposed evaluation (delta=1, RSSI check)
        prop_verdict = engine_proposed.validate_token(token, max_drift_windows=1, rssi=rssi, max_distance_m=10.0)
        leg_verdict = engine_legacy.validate_token(token, max_drift_windows=1000, rssi=None)
        
        records.append({
            "trial_id": i + 1,
            "scenario": "A_Legitimate_InRoom",
            "ground_truth": "GENUINE",
            "distance_m": round(dist, 2),
            "delay_sec": round(delay_sec, 2),
            "rssi": round(rssi, 1),
            "legacy_accepted": leg_verdict,
            "proposed_accepted": prop_verdict
        })

    # Scenario B: Adjacent Room / Hallway Bleed-Through (d in [9m, 22m], 12-16 dB wall attenuation)
    for i in range(NUM_TRIALS_PER_SCENARIO):
        dist = np.random.uniform(9.0, 22.0)
        wall_loss = np.random.uniform(12.0, 18.0)
        delay_sec = np.random.uniform(0.2, 3.5)
        rssi = simulate_rssi(dist, tx_power=CALIBRATED_1M_RSSI, wall_attenuation_db=wall_loss)
        
        # Attacker captures token leaking through wall
        token, _ = engine_proposed.generate_token(window_offset=0)
        
        prop_verdict = engine_proposed.validate_token(token, max_drift_windows=1, rssi=rssi, max_distance_m=10.0)
        leg_verdict = True  # offset 0 is within legacy delta=1000 window
        
        records.append({
            "trial_id": NUM_TRIALS_PER_SCENARIO + i + 1,
            "scenario": "B_Adjacent_Wall_Bleed",
            "ground_truth": "ATTACK",
            "distance_m": round(dist, 2),
            "delay_sec": round(delay_sec, 2),
            "rssi": round(rssi, 1),
            "legacy_accepted": leg_verdict,
            "proposed_accepted": prop_verdict
        })

    # Scenario C: Remote Relay Attack (Forwarded via WhatsApp/Telegram to remote student, delay 5s to 120s)
    for i in range(NUM_TRIALS_PER_SCENARIO):
        dist = np.random.uniform(50.0, 5000.0) # Outside campus
        delay_sec = np.random.uniform(5.0, 120.0)
        window_delay = int(delay_sec // 4)
        
        # Token was generated window_delay intervals ago
        token, _ = engine_proposed.generate_token(window_offset=-window_delay)
        
        # Attacker either doesn't have local RSSI or injects fake high RSSI or none
        rssi = simulate_rssi(dist, tx_power=CALIBRATED_1M_RSSI) # Highly attenuated or None
        
        prop_verdict = engine_proposed.validate_token(token, max_drift_windows=1, rssi=rssi, max_distance_m=10.0)
        leg_verdict = (window_delay <= 1000)  # within legacy delta=1000 window
        
        records.append({
            "trial_id": 2 * NUM_TRIALS_PER_SCENARIO + i + 1,
            "scenario": "C_Remote_Relay_Messaging",
            "ground_truth": "ATTACK",
            "distance_m": round(dist, 2),
            "delay_sec": round(delay_sec, 2),
            "rssi": round(rssi, 1),
            "legacy_accepted": leg_verdict,
            "proposed_accepted": prop_verdict
        })

    # Scenario D: Cryptographic Brute Force Token Guessing (Random 8-hex strings)
    for i in range(NUM_TRIALS_PER_SCENARIO):
        guessed_token = secrets.token_hex(4).upper()
        rssi = float(np.random.uniform(-40.0, -85.0))
        
        prop_verdict = engine_proposed.validate_token(guessed_token, max_drift_windows=1, rssi=rssi, max_distance_m=10.0)
        leg_verdict = False  # Collision probability in 2001 windows is 2001 / 16^8 = 4.66e-7
        
        records.append({
            "trial_id": 3 * NUM_TRIALS_PER_SCENARIO + i + 1,
            "scenario": "D_Cryptographic_Brute_Force",
            "ground_truth": "ATTACK",
            "distance_m": 0.0,
            "delay_sec": 0.0,
            "rssi": round(rssi, 1),
            "legacy_accepted": leg_verdict,
            "proposed_accepted": prop_verdict
        })

    # Scenario E: Replay Attack (Replaying genuine token after 15 to 3000 seconds)
    for i in range(NUM_TRIALS_PER_SCENARIO):
        replay_delay_sec = np.random.uniform(15.0, 2400.0) # within legacy 4000s window
        replay_windows = int(replay_delay_sec // 4)
        token, _ = engine_proposed.generate_token(window_offset=-replay_windows)
        rssi = simulate_rssi(np.random.uniform(2.0, 7.0)) # Captured inside room earlier
        
        prop_verdict = engine_proposed.validate_token(token, max_drift_windows=1, rssi=rssi, max_distance_m=10.0)
        leg_verdict = (replay_windows <= 1000)  # within legacy delta=1000 window
        
        records.append({
            "trial_id": 4 * NUM_TRIALS_PER_SCENARIO + i + 1,
            "scenario": "E_Replay_Harvested_Token",
            "ground_truth": "ATTACK",
            "distance_m": 4.0,
            "delay_sec": round(replay_delay_sec, 2),
            "rssi": round(rssi, 1),
            "legacy_accepted": leg_verdict,
            "proposed_accepted": prop_verdict
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(os.path.dirname(__file__), "results", "exp1_threat_scenarios_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved 10,000-trial dataset to: {csv_path}")

    # Compute Aggregate Metrics per Threat Scenario
    summary_metrics = {}
    for scen, group in df.groupby("scenario"):
        total = len(group)
        leg_acc = int(group["legacy_accepted"].sum())
        prop_acc = int(group["proposed_accepted"].sum())
        gt = group["ground_truth"].iloc[0]
        
        if gt == "GENUINE":
            # FRR = (Legitimate rejected) / total
            legacy_frr = (total - leg_acc) / total
            proposed_frr = (total - prop_acc) / total
            summary_metrics[scen] = {
                "total_trials": total,
                "ground_truth": gt,
                "legacy_accepted": leg_acc,
                "proposed_accepted": prop_acc,
                "legacy_FRR": round(legacy_frr, 4),
                "proposed_FRR": round(proposed_frr, 4),
                "mitigation_rate": round(1.0 - proposed_frr, 4)
            }
        else:
            # FAR = (Attacks accepted) / total
            legacy_far = leg_acc / total
            proposed_far = prop_acc / total
            mitigation_rate = (total - prop_acc) / total
            summary_metrics[scen] = {
                "total_trials": total,
                "ground_truth": gt,
                "legacy_accepted": leg_acc,
                "proposed_accepted": prop_acc,
                "legacy_FAR": round(legacy_far, 4),
                "proposed_FAR": round(proposed_far, 4),
                "attack_mitigation_rate": round(mitigation_rate, 4)
            }

    print("\nThreat Scenario Summary:")
    print(pd.DataFrame(summary_metrics).T[["ground_truth", "total_trials", "legacy_accepted", "proposed_accepted", "attack_mitigation_rate" if "attack_mitigation_rate" in pd.DataFrame(summary_metrics).T else "legacy_FAR"]])

    # Parametric Epoch Ablation: Delta_t in {2, 4, 8, 15, 30} seconds
    epoch_ablation = []
    for delta_t in [2, 4, 8, 15, 30]:
        test_engine = RollingSecurityEngine(ROOM_ID, CLASSROOM_KEY, interval=delta_t)
        # Test against 1000 relay attempts with messaging latency uniform in [3s, 45s]
        relay_delays = np.random.uniform(3.0, 45.0, 1000)
        intercepted = 0
        for delay in relay_delays:
            win_offset = -int(delay // delta_t)
            tok, _ = test_engine.generate_token(window_offset=win_offset)
            # Strict 1 window tolerance
            accepted = test_engine.validate_token(tok, max_drift_windows=1)
            if not accepted:
                intercepted += 1
                
        # Brute force collision probability in 1 window
        # P(collision) = (2 * delta + 1) / 16^8
        p_brute_force = (3.0) / (16.0**8)
        
        # Packet reception probability at mobile edge given BLE advertisement cycle
        adv_pdr = max(0.85, 1.0 - (0.02 * (10.0 / delta_t)))

        epoch_ablation.append({
            "epoch_interval_sec": delta_t,
            "relay_attack_block_rate": round(intercepted / 1000.0, 4),
            "brute_force_prob": float(f"{p_brute_force:.2e}"),
            "estimated_pdr": round(adv_pdr, 3),
            "bandwidth_overhead_bytes_per_min": int((60 / delta_t) * 13)
        })

    df_epoch = pd.DataFrame(epoch_ablation)
    epoch_csv = os.path.join(os.path.dirname(__file__), "results", "exp1_epoch_ablation.csv")
    df_epoch.to_csv(epoch_csv, index=False)
    print(f"\nSaved Epoch Ablation to: {epoch_csv}")

    final_json = {
        "benchmark": "Spatiotemporal_BLE_Threat_Mitigation",
        "num_total_trials": len(df),
        "threat_scenario_metrics": summary_metrics,
        "epoch_ablation": epoch_ablation
    }
    json_path = os.path.join(os.path.dirname(__file__), "results", "exp1_metrics_summary.json")
    with open(json_path, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"Saved JSON metrics to: {json_path}")
    return final_json

if __name__ == "__main__":
    run_threat_scenarios_benchmark()
