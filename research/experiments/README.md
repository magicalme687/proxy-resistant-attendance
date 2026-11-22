# Experiments

Every number, table and figure in the paper is produced here.

    python run_all.py              # simulations, figures, paper numbers
    python run_all.py --real-data  # also LFW and CelebA-Spoof (needs ../datasets)

| Script | What it produces | Type |
|---|---|---|
| `run_lfw_facenet_real_benchmark.py` | Face verification on the 6,000 LFW pairs (MTCNN-aligned) | measured |
| `run_celeba_spoof_classifier_benchmark.py` | 16-feature PAD with Platt calibration, CelebA-Spoof shard 0 | measured |
| `pad_classifier_ablation.py` | Five classifiers on the same PAD features | measured |
| `exp3_scalability_concurrency_benchmark.py` | Server latency (re-run on your own hardware) | measured |
| `exp1_ble_spatiotemporal_security.py` | Token and RSSI gate under five scenarios | simulated |
| `exp1_wall_attenuation_benchmark.py` | Leakage through partitions | simulated |
| `exp5_multiphone_rssi_variance.py` | Genuine rejection by handset profile | simulated |
| `exp4_adaptive_adversaries_benchmark.py` | Mode-C / I / I+ against adaptive attacks, 10 seeds; reads field latency files when present | simulated |
| `exp4_trimodal_qr_ble_fusion.py` | Analytical composition; writes the assumptions table | analytical |
| `generate_paper_figures.py` | `../../manuscript/figures/` | — |
| `make_paper_numbers.py` | `../../manuscript/generated/` (numbers and table rows used by the paper) | — |

Results go to `results/`. Install requirements with `pip install -r requirements.txt`.
