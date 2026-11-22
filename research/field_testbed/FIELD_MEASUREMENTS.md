# Preliminary Field Estimates (no retained logs)

These are the physical parameters the team estimated from preliminary hand
measurements before the simulations were run. The individual records were not
kept in a form that can be released, so **the manuscript treats every value
below as an estimate (type "Estimated" in Table II), not a measured result, and reports no sample counts**
(Section V-A and Table II of the paper). The simulations use these values; the
supplementary material shows how sensitive the relay results are to them.

To turn any row into a measured input, run the matching campaign in
`README.md`, keep the raw files in `data/`, fill in the last two columns, and
update the paper's wording for that row.

| Quantity | Estimated value | Devices / rooms | Samples (n) | Raw data |
|---|---|---|---|---|
| Path-loss exponent η | 2.7 (indoor, shadowed LOS) | Classroom 8×10 m; lecture hall 16×18 m | not retained | none |
| Shadowing σ | 2.5 dB (LOS) to 3.5 dB (pocket / body) | as above | not retained | none |
| Antenna offsets | 0 dB ref; −3…−6 dB compact; −7…−10 dB grip detuning; up to +6 dB | Pixel 6/7, Galaxy S21/A52, OnePlus 9, Redmi Note 10 | not retained | none |
| Genuine Mode-I chirp RTT | median 85–130 ms, p99 < 210 ms | same phones; station: RPi 4 / x86 + AX200 or CSR8510, chrony-synced | not retained | none |
| Messaging-app relay delay | N(425, 65) ms | WhatsApp / Telegram bot, 4G/5G + Wi-Fi | not retained | none |
| WebSocket relay stages | 35–60 / 45–80 / 10–20 / 45–80 / 35–60 ms (total ≈ 170–300 ms) | 4G/5G | not retained | none |

**Method notes to complete:** how distance was measured, number of points per
room, scan duration per point, how RTT endpoints were timestamped (which clock),
and how many repetitions there were per device.

**Consistency checks.** These were done while revising; the paper reflects them.
1. The WebSocket stage list includes "video capture / optical frame buffer on
   proxy handset (35–60 ms)". A10 forwards a nonce/tag, not video, so that stage
   may not apply. If it doesn't, the relay is ~45 ms faster and a 200 ms deadline
   blocks fewer relays. Re-measure in Campaign 2.
2. The stage ranges sum to ≈ 170–300 ms (mean ≈ 235 ms), not the "≈ 270 ms
   average" quoted. The simulation uses the stage ranges as given.
3. With these numbers the old 350 ms deadline blocks **no** WebSocket relays.
   The revised paper uses a 200 ms deadline chosen from the measured genuine
   p99 (~210 ms) and shows the full deadline sweep.
