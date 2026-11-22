# -*- coding: utf-8 -*-
"""
run_all.py
Master Reproducibility Benchmark Runner for IEEE IoT-J Manuscript:
'A Zero-Trust Tri-Modal Co-Presence Protocol Using Ephemeral Optical-RF Physical Channels and Cascaded Anti-Spoofing Biometrics'

Executes all experimental benchmarks, verifies data integrity, and regenerates publication figures:
- Experiment 1: BLE Spatiotemporal Security, RF Path-Loss, & Wall Bleed (10,000 trials)
- Experiment 1 Extension: Partition Attenuation & Hallway Bleed Sweep (12,000 trials)
- Experiment 2: Biometric Verification & Presentation Attack Forensic Ablation (3,000 trials)
- Experiment 3: Computational Scalability & Execution Latency Micro-benchmarks (100 repetitions)
- Experiment 4: Tri-Modal Orthogonal Co-Presence & Channel Ablation Benchmark (12,000 trials)
- Experiment 4 Extension: Adaptive Adversaries Benchmark (Mode-C vs. Mode-I vs. Mode-I+)
- Experiment 5: Handset Antenna Gain Detuning & FRR Benchmark (20,000 trials)
- Figures: Regenerates publication figures fig1 through fig5 (PDF & 300 DPI PNG)
- Integrity Check: Programmatically verifies Tables I-VIII against generated results via make_tables.py
- Optional (--real-data): Runs LFW FaceNet and CelebA-Spoof real evaluations if datasets are present
"""

import os
import sys
import time
import argparse
import subprocess

# Order matters: real-data results feed Exp. 4 and the adaptive simulation.
REAL_DATA_BENCHMARKS = [
    ("Real Data: LFW 6,000 pairs, MTCNN-aligned", "run_lfw_facenet_real_benchmark.py"),
    ("Real Data: CelebA-Spoof 16-D PAD (calibrated)", "run_celeba_spoof_classifier_benchmark.py"),
]

CORE_BENCHMARKS = [
    ("PAD classifier ablation (cached features)", "pad_classifier_ablation.py"),
    ("Exp 1: BLE path loss and replay (simulated)", "exp1_ble_spatiotemporal_security.py"),
    ("Exp 1 ext: partition leakage (simulated)", "exp1_wall_attenuation_benchmark.py"),
    ("Exp 3: server latency (measured)", "exp3_scalability_concurrency_benchmark.py"),
    ("Exp 5: handset detuning FRR (simulated)", "exp5_multiphone_rssi_variance.py"),
    ("Exp 4: analytical composition + assumptions", "exp4_trimodal_qr_ble_fusion.py"),
    ("Exp 4 ext: adaptive adversaries, 10 seeds", "exp4_adaptive_adversaries_benchmark.py"),
    ("Figures", "generate_paper_figures.py"),
    ("Paper numbers and tables (manuscript/generated/)", "make_paper_numbers.py"),
]

def main():
    parser = argparse.ArgumentParser(description="IEEE IoT-J Master Reproducibility Benchmark Runner")
    parser.add_argument("--real-data", action="store_true", help="Execute real LFW and CelebA-Spoof benchmark scripts")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    benchmarks_to_run = (list(REAL_DATA_BENCHMARKS) if args.real_data else []) + list(CORE_BENCHMARKS)

    print("=" * 75)
    print("IEEE IoT-J REPRODUCIBILITY HARNESS: MASTER BENCHMARK RUNNER")
    print(f"Directory: {base_dir}")
    print(f"Total Stages to Execute: {len(benchmarks_to_run)}")
    if args.real_data:
        print("Real-Data Flag Active: Including LFW and CelebA-Spoof benchmarks")
    print("=" * 75)

    start_total = time.time()
    passed = 0
    failed = 0

    for idx, (title, script_name) in enumerate(benchmarks_to_run, 1):
        script_path = os.path.join(base_dir, script_name)
        print(f"\n[{idx}/{len(benchmarks_to_run)}] Running: {title} ({script_name})...")
        t0 = time.time()
        res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
        elapsed = time.time() - t0

        if res.returncode == 0:
            print(f"  --> PASSED ({elapsed:.2f}s)")
            passed += 1
        else:
            print(f"  --> FAILED (code {res.returncode}) ({elapsed:.2f}s)")
            print("  --- Error Output ---")
            print(res.stderr[:500] if res.stderr else res.stdout[:500])
            failed += 1

    total_time = time.time() - start_total
    print("\n" + "=" * 75)
    print(f"EXECUTION SUMMARY: {passed} PASSED, {failed} FAILED in {total_time:.2f}s")
    print("=" * 75)

    if failed == 0:
        print("All experimental benchmarks and figures regenerated successfully.")
        return 0
    else:
        print("WARNING: Some benchmarks failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
