# Proxy-Resistant Smartphone Attendance: Research Code

Evaluation code, results, and field-test kit for a smartphone attendance protocol that checks three things at once: a rotating QR code on the projector, a rotating Bluetooth Low Energy (BLE) token with a distance check, and face verification with liveness detection.

Paper: *Proxy-Resistant Smartphone Attendance Using Combined Optical, Bluetooth Low Energy, and Face Verification* (submitted to IEEE Internet of Things Journal, November 2026).

The attendance application itself (server, mobile apps, web clients) is not part of this repository. Everything needed to reproduce the results in the paper is here.

**License.** All rights reserved. This code is provided only so that reviewers and readers can verify the results in the paper; it may not be copied, reused, or redistributed without written permission. See [LICENSE](LICENSE).

## Layout

```
research/
  experiments/       Benchmarks and simulations; results/ holds every number in the paper
  field_testbed/     Loggers, analysis scripts and consent forms for classroom measurements
  datasets/          LFW and CelebA-Spoof (not tracked in git; see below)
run_all.py           Re-runs everything
```

## Reproducing the paper

```bash
pip install -r research/experiments/requirements.txt
python run_all.py              # simulations, figures, and the numbers and tables used in the paper
python run_all.py --real-data  # also re-runs LFW and CelebA-Spoof (needs the datasets)
```

`run_all.py` executes the full research benchmark suite (`research/experiments/run_all.py`), regenerating all experimental benchmarks, statistical metrics, tables, and figures. Every number in the paper comes from those generated results.

**Datasets.** Put the LFW funneled archive and `pairs.txt` in `research/datasets/lfw/`, and CelebA-Spoof test shard 0 (`test-00000-of-00010.parquet`) in `research/datasets/celeba_spoof/`. Both are public; they are not tracked in git because of their size.

**Field measurements.** The channel and latency parameters used by the simulations are estimates from preliminary hand measurements (see `research/field_testbed/FIELD_MEASUREMENTS.md`). The field-test kit in `research/field_testbed/README.md` replaces them with logged measurements; once measured latency files exist, the adaptive-attack simulation reads them automatically.
