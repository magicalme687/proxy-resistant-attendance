"""
Experiment C1: RF Partition Attenuation & Distance Gate Leakage Benchmark
Evaluates BLE RSSI gate pass rates for exterior corridor positions d in [1, 10] m
across standard architectural wall partitions:
  - Partition 1: Interior drywall (3 - 6 dB attenuation)
  - Partition 2: Brick masonry (7 - 12 dB attenuation)
  - Partition 3: Reinforced concrete (12 - 18 dB attenuation)
  - Partition 4: Open doorway / Thin wooden panel (1 - 4 dB attenuation)

Evaluates gate pass rates under:
  - Standard gate radius: d_max = 10.0 m (cutoff -86.0 dBm)
  - Tighter gate radius:   d_max = 6.0 m  (cutoff -79.9 dBm)
  - Ultra-tight gate radius: d_max = 4.0 m (cutoff -75.3 dBm)

Outputs:
  - research_experiments/results/c1_partition_attenuation_benchmark.csv
  - research_experiments/results/c1_partition_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NUM_TRIALS_PER_CONFIG = 1000

PARTITIONS = {
    "Drywall (Gypsum 10cm)": {"loss_min": 3.0, "loss_max": 6.0},
    "Brick Masonry (15cm)": {"loss_min": 7.0, "loss_max": 12.0},
    "Reinforced Concrete (20cm)": {"loss_min": 12.0, "loss_max": 18.0},
    "Open Doorway / Thin Panel": {"loss_min": 1.0, "loss_max": 4.0}
}

GATE_RADII = [10.0, 6.0, 4.0]
TX_POWER = -59.0  # RSSI_0 at 1m for Tx = -12 dBm
PATH_LOSS_EXP = 2.7
SHADOW_SIGMA = 2.4

def clopper_pearson(k, n, alpha=0.05):
    low = stats.beta.ppf(alpha / 2.0, k, n - k + 1) if k > 0 else 0.0
    high = stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k) if k < n else 1.0
    return float(low), float(high)

def run_rf_partition_benchmark():
    print("=" * 75)
    print("RUNNING C1: RF Partition Attenuation & Distance Gate Leakage Benchmark")
    print("=" * 75)

    records = []

    for part_name, loss_params in PARTITIONS.items():
        for d_gate in GATE_RADII:
            cutoff_rssi = TX_POWER - 10.0 * PATH_LOSS_EXP * np.log10(d_gate)
            
            # Sample exterior distances d in [1.0, 10.0] meters outside partition
            distances = np.random.uniform(1.0, 10.0, NUM_TRIALS_PER_CONFIG)
            wall_losses = np.random.uniform(loss_params["loss_min"], loss_params["loss_max"], NUM_TRIALS_PER_CONFIG)
            shadowing = np.random.normal(0.0, SHADOW_SIGMA, NUM_TRIALS_PER_CONFIG)
            
            rx_rssi = TX_POWER - (10.0 * PATH_LOSS_EXP * np.log10(np.maximum(distances, 0.5))) - wall_losses + shadowing
            d_est = 1.0 * (10.0 ** ((TX_POWER - rx_rssi) / (10.0 * PATH_LOSS_EXP)))
            
            # Exterior position passes gate if estimated distance <= d_gate and RSSI >= cutoff
            passed_gate = (rx_rssi >= cutoff_rssi) & (d_est <= d_gate)
            n_passed = int(np.sum(passed_gate))
            pass_rate = (n_passed / NUM_TRIALS_PER_CONFIG) * 100.0
            ci_low, ci_high = clopper_pearson(n_passed, NUM_TRIALS_PER_CONFIG)
            mitigation_rate = 100.0 - pass_rate

            records.append({
                "partition_type": part_name,
                "attenuation_range_db": f"{loss_params['loss_min']}-{loss_params['loss_max']}",
                "gate_radius_m": d_gate,
                "cutoff_rssi_dbm": round(cutoff_rssi, 1),
                "num_trials": NUM_TRIALS_PER_CONFIG,
                "false_accepts": n_passed,
                "leakage_rate_pct": round(pass_rate, 2),
                "mitigation_pct": round(mitigation_rate, 2),
                "ci_95_low_pct": round(ci_low * 100.0, 2),
                "ci_95_high_pct": round(ci_high * 100.0, 2)
            })

            print(f"[{part_name[:18]:18s}] Gate {d_gate:4.1f}m (Cutoff {cutoff_rssi:5.1f} dBm) | "
                  f"Leakage: {n_passed:3d}/{NUM_TRIALS_PER_CONFIG} ({pass_rate:5.2f}%) | "
                  f"Mitigation: {mitigation_rate:5.2f}% (95% CI [{ci_low*100.0:4.1f}%, {ci_high*100.0:4.1f}%])")

    df = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "c1_partition_attenuation_benchmark.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to: {csv_path}")

    summary_path = os.path.join(RESULTS_DIR, "c1_partition_summary.json")
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Saved JSON summary to: {summary_path}")

if __name__ == "__main__":
    run_rf_partition_benchmark()
