# Physical Testbed Kit

This kit turns the paper's simulated physical-layer claims into measured ones.
It has four campaigns. Each produces a CSV in `data/` with a fixed format. The
scripts in `analysis/` turn those CSVs into the paper's tables and figures, and
they write `research/experiments/results/testbed_latency_params.json`. The
adaptive-adversary simulation (`research/experiments/exp4_adaptive_adversaries_benchmark.py`) picks
that file up automatically.

**Golden rule:** every number in the paper must trace back to a row in `data/`.
Keep raw files untouched. Record any exclusions in `data/EXCLUSIONS.md` with a reason.

---

## Equipment

| Item | Purpose | Notes |
|---|---|---|
| Teacher station: Raspberry Pi 4 or laptop + BLE 5.0 dongle (Intel AX200 / CSR8510) | Advertises the τ_rf beacon (Mode-C). Scans student chirps (Mode-I). | Run `station/station_logger.py` |
| Server: laptop on the same LAN | Issues nonces, logs issue time | Run `station/nonce_server.py`. Sync station and server clocks with `chrony` (both on LAN; record `chronyc tracking` offset before each session). |
| Phones: ≥ 6 models, ≥ 1 iPhone | Walk-tests, chirps, relays | Record the exact model, OS version and case (yes/no) in `data/devices.csv` |
| Tape measure / floor markers | Ground-truth distance | Mark grid points every 1 m |
| Second phone + mobile data | Relay attacker | 4G and campus Wi-Fi, separately |

## Campaign 1: RSSI walk-test (replaces Tables II, VI, VIII and Fig. 1)

1. Station beacon at the projector position, height ≈ 1.2 m, P_tx = −12 dBm.
2. Mark points **inside** the room at 1, 2, …, 10 m along 3 radial lines.
3. Mark points **outside** at 1–10 m beyond each partition type you have: drywall wall, brick/concrete wall, closed wooden door, **open doorway**, glass.
4. At each point, for each phone and each posture (hand, pocket, on desk), log **30 s** of advertisements with `phone/scan_logger.dart`.
5. Note whether the projector screen is **visible** from the point (`los_to_screen` = 1/0). This is the optical gate's ground truth.

Output: `data/walktest.csv` with columns
`session_id, room_id, device_id, posture, point_id, inside_room, partition_type, distance_m, los_to_screen, t_unix_ms, rssi_dbm`

Minimum size: 2 rooms × 6 phones × 3 postures × ~50 points × 30 s ≈ 1,800 point-series.

## Campaign 2: Latency (replaces the RTT assumptions in Table VII)

The server timestamps nonce issue and the station timestamps chirp reception, so **phones need no clock sync**.

- **Genuine:** phone requests a nonce, signs it, and advertises the chirp; the station logs reception. Repeat ≥ 200 times per phone model (mix of 4G and Wi-Fi).
- **A10 WebSocket relay:** the absent phone (off-site, on 4G) receives the nonce, forwards it through `station/relay_server.py` to the in-room accomplice, which advertises it. Repeat ≥ 200 times.
- **Messaging relay:** same, but forwarded by a human via WhatsApp/Telegram. Repeat ≥ 50 times.
- **Video relay of QR (A2):** screen-share the QR via Meet/WhatsApp video; log the glass-to-glass delay with a stopwatch app filmed alongside. ≥ 30 times.

Output: `data/latency.csv`:
`trial_id, kind{genuine,websocket_relay,messaging_relay}, device_id, network{4g,5g,wifi}, nonce, t_issue_ms, t_rx_ms, rssi_station_dbm`

## Campaign 3: Co-location detector (replaces P_det = 0.65 and FP = 2.4%)

- **Positive:** one accomplice phone transmits its own chirp and a relayed chirp within 213 ms. ≥ 200 trials.
- **Negative (false positives):** pairs of genuine students seated 0.4–0.8 m apart chirp naturally. ≥ 500 pairs.

Output: `data/colocation.csv`:
`trial_id, label{relay,genuine_pair}, rssi_a_dbm, rssi_b_dbm, dt_ms`

## Campaign 4: Classroom pilot (end-to-end GAR, usability)

Needs ethics approval and consent; see `ethics/`. 30–60 volunteer students, ≥ 6 sessions, Mode-C and Mode-I.
Log every attempt from the attendance server. Afterwards, give the SUS questionnaire.

Output: `data/pilot_attempts.csv`:
`attempt_id, session_id, student_hash, mode, t_start_ms, t_end_ms, gate_failed{none,optical,rf,pad,face,rtt,coloc}, accepted`
plus `data/sus.csv`: `student_hash, q1..q10`.
Red-team attempts (authorised volunteers acting as proxies, print/screen spoofs) go in the same file with `is_attack=1, attack_type`.

---

## Analysis

```bash
python analysis/analyze_walktest.py   # fits RSSI0, gamma, sigma, device offsets; leakage tables
python analysis/analyze_latency.py    # RTT distributions; writes testbed_latency_params.json
python analysis/analyze_colocation.py # measured P_det and false-positive rate
python analysis/analyze_pilot.py      # GAR, per-gate failures, time-to-mark, SUS
python ../experiments/exp4_adaptive_adversaries_benchmark.py   # re-runs Table VII on measured params
```

Outputs go to `testbed/results/` as CSV plus LaTeX table snippets (`*.tex`) that `main.tex` can `\input`.
