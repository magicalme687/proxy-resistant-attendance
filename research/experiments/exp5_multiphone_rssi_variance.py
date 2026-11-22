"""
Experiment C5: Multi-Phone Hardware RSSI Dispersion & False Rejection Benchmark
Quantifies the impact of phone-to-phone antenna gain offsets (+-5 to 10 dB) on
Genuine In-Room False Rejection Rate (FRR) across different gate radii:
  - Phone Types:
    * Standard Baseline (0 dB offset, e.g., calibrated reference)
    * High-Sensitivity Handset (+3 to +6 dB antenna gain)
    * Low-Gain Compact Handset (-3 to -6 dB antenna loss)
    * Severe Antenna Detuning (-7 to -10 dB attenuation)

Evaluates:
  - 2,000 legitimate in-room trials per device model (d in [1.0, 8.0] m)
  - Resulting False Rejection Rate (FRR) across gate thresholds (d_max in {6m, 8m, 10m})
  - Recommends adaptive normalization / teacher-side sensing mitigation

Outputs:
  - research_experiments/results/c5_multiphone_rssi_variance.csv
  - research_experiments/results/c5_multiphone_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_TRIALS = 2000
TX_POWER = -59.0
PATH_LOSS_EXP = 2.7
SHADOW_SIGMA = 2.4

DEVICE_MODELS = {
    "Calibrated Reference (0 dB)": {"offset_min": -0.5, "offset_max": 0.5},
    "High-Gain Premium (+3 to +6 dB)": {"offset_min": 3.0, "offset_max": 6.0},
    "Low-Gain Compact (-3 to -6 dB)": {"offset_min": -6.0, "offset_max": -3.0},
    "Detuned / Protective Case (-7 to -10 dB)": {"offset_min": -10.0, "offset_max": -7.0}
}

GATE_RADII = [10.0, 8.0, 6.0]

def clopper_pearson(k, n, alpha=0.05):
    low = stats.beta.ppf(alpha / 2.0, k, n - k + 1) if k > 0 else 0.0
    high = stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k) if k < n else 1.0
    return float(low), float(high)

def run_multiphone_benchmark():
    print("=" * 75)
    print("RUNNING C5: Multi-Phone RSSI Hardware Offset & FRR Benchmark")
    print("=" * 75)

    records = []

    for dev_name, params in DEVICE_MODELS.items():
        for d_gate in GATE_RADII:
            cutoff_rssi = TX_POWER - 10.0 * PATH_LOSS_EXP * np.log10(d_gate)
            
            # Legitimate student in-room distances d in [1.0, 8.0] meters
            d_true = np.random.uniform(1.0, 8.0, NUM_TRIALS)
            hw_offset = np.random.uniform(params["offset_min"], params["offset_max"], NUM_TRIALS)
            shadowing = np.random.normal(0.0, SHADOW_SIGMA, NUM_TRIALS)
            
            # Received RSSI includes hardware antenna bias
            rx_rssi = TX_POWER - (10.0 * PATH_LOSS_EXP * np.log10(d_true)) + hw_offset + shadowing
            d_est = 1.0 * (10.0 ** ((TX_POWER - rx_rssi) / (10.0 * PATH_LOSS_EXP)))
            
            # Genuine attendee rejected if RSSI falls below cutoff or d_est exceeds gate
            passed = (rx_rssi >= cutoff_rssi) & (d_est <= d_gate)
            false_rejections = int(np.sum(~passed))
            frr_pct = (false_rejections / NUM_TRIALS) * 100.0
            gar_pct = 100.0 - frr_pct
            ci_low, ci_high = clopper_pearson(false_rejections, NUM_TRIALS)

            records.append({
                "device_category": dev_name,
                "hardware_offset_db": f"{params['offset_min']} to {params['offset_max']}",
                "gate_radius_m": d_gate,
                "cutoff_rssi_dbm": round(cutoff_rssi, 1),
                "num_trials": NUM_TRIALS,
                "false_rejections": false_rejections,
                "frr_pct": round(frr_pct, 2),
                "gar_pct": round(gar_pct, 2),
                "frr_ci_95_low": round(ci_low * 100.0, 2),
                "frr_ci_95_high": round(ci_high * 100.0, 2)
            })

            print(f"[{dev_name[:20]:20s}] Gate {d_gate:4.1f}m | "
                  f"FRR: {false_rejections:4d}/{NUM_TRIALS} ({frr_pct:5.2f}%) | "
                  f"GAR: {gar_pct:5.2f}% (95% CI [{ci_low*100.0:4.1f}%, {ci_high*100.0:4.1f}%])")

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "c5_multiphone_rssi_variance.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")

    summary_path = os.path.join(RESULTS_DIR, "c5_multiphone_summary.json")
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved JSON summary to: {summary_path}")

if __name__ == "__main__":
    run_multiphone_benchmark()
