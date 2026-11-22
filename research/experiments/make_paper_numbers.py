"""
Generate manuscript/generated/numbers.tex (LaTeX macros) and paper/generated/tab_*.tex
(table bodies) directly from research/experiments/results/*. The manuscript
uses only these macros/tables for reported numbers, so text and data cannot
drift apart. Run after run_all.py.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "..", "..", "manuscript", "generated")
os.makedirs(OUT, exist_ok=True)

M = {}


def put(name, value):
    M[name] = value


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def J(name):
    return json.load(open(os.path.join(RES, name)))


def cp_hi(k, n):
    return stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0


def cp_lo(k, n):
    return stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0


# ---------------- LFW ----------------
L = J("real_lfw_facenet_summary.json")
put("lfwAUC", f"{L['roc_auc']:.4f}")
put("lfwAUClo", f"{L['roc_auc_ci_95'][0]:.4f}"); put("lfwAUChi", f"{L['roc_auc_ci_95'][1]:.4f}")
put("lfwEER", pct(L["eer"])); put("lfwEERlo", pct(L["eer_ci_95"][0])); put("lfwEERhi", pct(L["eer_ci_95"][1]))
put("lfwThetaEER", f"{L['eer_threshold']:.4f}")
te = L["theta_eer_metrics"]
put("lfwGAR", f"{te['gar_percent']:.2f}"); put("lfwFAR", f"{te['far_percent']:.2f}")
put("lfwFAcount", str(te["imposter_accept_count"]))
put("lfwBuddyMit", f"{te['buddy_proxy_mitigation_percent']:.2f}")
far = te["far_percent"] / 100
put("lfwFARthree", pct(1 - (1 - far) ** 3))
put("lfwTARlow", f"{L['tar_at_far_01_percent']:.2f}")
put("lfwTARlowLo", f"{L['tar_at_far_01_ci_95'][0]:.2f}"); put("lfwTARlowHi", f"{L['tar_at_far_01_ci_95'][1]:.2f}")
put("lfwThetaLow", f"{L['threshold_at_far_01']:.4f}")
k_low = int(round(L["actual_far_at_far_01"] / 100 * L["imposter_pairs"]))
put("lfwFAlowCount", str(k_low))
put("lfwFARlowThree", pct(1 - (1 - L["actual_far_at_far_01"] / 100) ** 3))
put("lfwGenMean", f"{L['genuine_sim_mean']:.3f}"); put("lfwImpMean", f"{L['imposter_sim_mean']:.3f}")
put("lfwGenSD", f"{L['genuine_sim_std']:.3f}"); put("lfwImpSD", f"{L['imposter_sim_std']:.3f}")
put("lfwDetected", f"{L.get('mtcnn_detected', 0):,}"); put("lfwFallback", str(L.get("mtcnn_fallback_center_crop", 0)))
U = J("real_lfw_facenet_UNALIGNED_summary.json")
put("lfwUnalEER", pct(U["eer"])); put("lfwUnalAUC", f"{U['roc_auc']:.4f}")
put("lfwUnalTARlow", f"{U['tar_at_far_01_percent']:.2f}")

# ---------------- PAD ----------------
P = J("real_celeba_spoof_classifier_summary.json")
pp = P["proposed_calibrated_multispectral_classifier"]; hb = P["baseline_uncalibrated_heuristics"]
put("padN", f"{P['sample_count']:,}"); put("padLive", f"{P['live_count']:,}"); put("padSpoof", f"{P['spoof_count']:,}")
put("padAUC", f"{pp['roc_auc']:.4f}"); put("padAUClo", f"{pp['roc_auc_ci_95'][0]:.4f}"); put("padAUChi", f"{pp['roc_auc_ci_95'][1]:.4f}")
put("padEER", f"{pp['eer_pct']:.2f}"); put("padEERlo", f"{pp['eer_ci_95'][0]:.2f}"); put("padEERhi", f"{pp['eer_ci_95'][1]:.2f}")
put("padTauEER", f"{pp['eer_threshold']:.4f}")
eo = pp["eer_operating_point"]; so = pp["high_security_operating_point"]
put("padAPCER", f"{eo['apcer_pct']:.2f}"); put("padBPCER", f"{eo['bpcer_pct']:.2f}")
put("padACER", f"{(eo['apcer_pct'] + eo['bpcer_pct']) / 2:.2f}")
put("padAPCERcount", str(round(eo["apcer_pct"] / 100 * P["spoof_count"])))
put("padBPCERcount", str(round(eo["bpcer_pct"] / 100 * P["live_count"])))
put("padSpoofMit", f"{100 - eo['apcer_pct']:.2f}")
put("padTauStrict", f"{so['threshold']:.4f}")
put("padAPCERs", f"{so['apcer_pct']:.2f}"); put("padBPCERs", f"{so['bpcer_pct']:.2f}")
put("padACERs", f"{(so['apcer_pct'] + so['bpcer_pct']) / 2:.2f}"); put("padSpoofMitS", f"{100 - so['apcer_pct']:.2f}")
put("padFeatMs", f"{pp['latency']['feature_extraction_ms_median']:.2f}")
_ft = np.load(os.path.join(RES, "celeba_features_cache.npz"))["feature_times"]
put("padFeatIQR", f"{np.percentile(_ft, 75) - np.percentile(_ft, 25):.2f}")
put("padClfMs", f"{pp['latency']['classifier_inference_ms_mean']:.3f}")
put("heurAPCER", f"{hb['apcer_pct']:.2f}"); put("heurBPCER", f"{hb['bpcer_pct']:.2f}")
put("heurACER", f"{hb['acer_pct']:.2f}"); put("heurAUC", f"{hb['roc_auc']:.4f}"); put("heurEER", f"{hb['eer_pct']:.2f}")

ab = pd.read_csv(os.path.join(RES, "pad_classifier_ablation.csv"))
with open(os.path.join(OUT, "tab_pad_ablation.tex"), "w") as f:
    best_eer = ab.eer_pct.min(); best_brier = ab.brier.min()
    for _, r in ab.iterrows():
        e = f"\\textbf{{{r.eer_pct:.2f}}}" if r.eer_pct == best_eer else f"{r.eer_pct:.2f}"
        b = f"\\textbf{{{r.brier:.4f}}}" if r.brier == best_brier else f"{r.brier:.4f}"
        f.write(f"{r.classifier} & {r.roc_auc:.4f} & {e} & {b} & {r.infer_ms_per_sample:.3f} \\\\\n")
rf = ab[ab.classifier.str.startswith("Random")].iloc[0]
put("padRFEER", f"{rf.eer_pct:.2f}"); put("padRFAUC", f"{rf.roc_auc:.4f}")

# ---------------- Exp 1 ----------------
E1 = J("exp1_metrics_summary.json")["threat_scenario_metrics"]
a = E1["A_Legitimate_InRoom"]
put("eOneLegitAcc", f"{a['proposed_accepted']:,}"); put("eOneLegitGAR", pct(a["proposed_accepted"] / a["total_trials"]))
with open(os.path.join(OUT, "tab_exp1.tex"), "w") as f:
    lab = {"A_Legitimate_InRoom": "Legitimate in-room ($d\\le 8$\\,m)",
           "B_Adjacent_Wall_Bleed": "Corridor behind solid wall ($d\\in[9,22]$\\,m)",
           "C_Remote_Relay_Messaging": "Relay via messaging app ($\\Delta t\\in[5,120]$\\,s)",
           "D_Cryptographic_Brute_Force": "Brute-force guess of 32-bit token",
           "E_Replay_Harvested_Token": "Replay of a captured token"}
    for k, v in E1.items():
        n = v["total_trials"]; acc = v["proposed_accepted"]; leg = v["legacy_accepted"]
        if v["ground_truth"] == "GENUINE":
            f.write(f"{lab[k]} & {n:,} & {leg:,} & {acc:,} & GAR {pct(acc / n)}\\% \\\\\n")
        else:
            f.write(f"{lab[k]} & {n:,} & {leg:,} & {acc:,} & FAR {pct(acc / n)}\\% [0, {pct(cp_hi(acc, n))}] \\\\\n")
put("cpZeroOfTwoK", pct(cp_hi(0, 2000)))
ep = pd.read_csv(os.path.join(RES, "exp1_epoch_ablation.csv"))
with open(os.path.join(OUT, "tab_epoch.tex"), "w") as f:
    for _, r in ep.iterrows():
        f.write(f"{int(r.epoch_interval_sec)} & {pct(r.relay_attack_block_rate)}\\% & {pct(r.estimated_pdr)}\\% & {int(r.bandwidth_overhead_bytes_per_min)} \\\\\n")

# ---------------- partition & multiphone ----------------
pt = pd.read_csv(os.path.join(RES, "c1_partition_attenuation_benchmark.csv"))
with open(os.path.join(OUT, "tab_partition.tex"), "w") as f:
    last = None
    for _, r in pt.iterrows():
        if last is not None and r.partition_type != last:
            f.write("\\midrule\n")
        last = r.partition_type
        f.write(f"{r.partition_type} & {r.attenuation_range_db.replace('.0', '')} & {int(r.gate_radius_m)}\\,m & "
                f"{r.leakage_rate_pct:.2f}\\% [{r.ci_95_low_pct:.2f}, {r.ci_95_high_pct:.2f}] \\\\\n")
    put("leakConcreteTen", f"{pt[(pt.partition_type.str.startswith('Reinforced')) & (pt.gate_radius_m == 10)].leakage_rate_pct.iloc[0]:.2f}")
    put("leakConcreteFour", f"{pt[(pt.partition_type.str.startswith('Reinforced')) & (pt.gate_radius_m == 4)].leakage_rate_pct.iloc[0]:.2f}")
    put("leakDryTen", f"{pt[(pt.partition_type.str.startswith('Drywall')) & (pt.gate_radius_m == 10)].leakage_rate_pct.iloc[0]:.2f}")
    put("leakDoorTen", f"{pt[(pt.partition_type.str.startswith('Open')) & (pt.gate_radius_m == 10)].leakage_rate_pct.iloc[0]:.2f}")
    put("leakDoorFour", f"{pt[(pt.partition_type.str.startswith('Open')) & (pt.gate_radius_m == 4)].leakage_rate_pct.iloc[0]:.2f}")
mp = pd.read_csv(os.path.join(RES, "c5_multiphone_rssi_variance.csv"))
with open(os.path.join(OUT, "tab_multiphone.tex"), "w") as f:
    last = None
    for _, r in mp.iterrows():
        name = r.device_category.split(" (")[0]
        if last is not None and name != last:
            f.write("\\midrule\n")
        last = name
        f.write(f"{name} & {r.hardware_offset_db} & {int(r.gate_radius_m)}\\,m & {r.frr_pct:.2f}\\% [{r.frr_ci_95_low:.2f}, {r.frr_ci_95_high:.2f}] \\\\\n")
g = lambda cat, R: mp[(mp.device_category.str.startswith(cat)) & (mp.gate_radius_m == R)].frr_pct.iloc[0]
put("frrRefTen", f"{g('Calibrated', 10):.2f}"); put("frrLowTen", f"{g('Low-Gain', 10):.2f}"); put("frrCaseTen", f"{g('Detuned', 10):.2f}")
put("frrCaseSix", f"{g('Detuned', 6):.2f}")

# ---------------- Exp 3 ----------------
E3 = J("exp3_metrics_summary.json")
st = {s["stage"]: s for s in E3["pipeline_microbenchmarks"]}
put("latKernelUs", f"{E3['vectorized_kernel_microseconds']:.1f}")
put("latSqlite", f"{E3['sqlite_commit_measured_ms']:.2f}"); put("latSqliteWal", f"{E3['sqlite_commit_wal_measured_ms']:.2f}")
put("latGateway", f"{st['Total Gateway In-Memory Latency (Excluding Disk Commit)']['median_ms']:.2f}")
put("latGatewayIQR", f"{st['Total Gateway In-Memory Latency (Excluding Disk Commit)']['iqr_ms']:.2f}")
put("latPAD", f"{st['Stage 2: 16-D Passive PAD Extraction + Inference']['median_ms']:.2f}")
put("latHMAC", f"{st['Stage 0: BLE HMAC Cryptographic Check']['median_ms']:.4f}")
cb = {c["cohort_size_N"]: c for c in E3["cohort_benchmarks"]}
put("latBatchTwoHundred", f"{cb[200]['vectorized_median_ms']:.2f}"); put("latSeqTwoHundred", f"{cb[200]['sequential_median_ms']:.2f}")
put("latSpeedup", f"{cb[200]['speedup_ratio_median']:.2f}")

# ---------------- Exp 4 ----------------
X = J("exp4_trimodal_summary.json")["rows"]
with open(os.path.join(OUT, "tab_exp4.tex"), "w") as f:
    for i, r in enumerate(X):
        bold = r["architecture"].startswith("Proposed")
        cells = [r["architecture"], pct(r["gar"]), pct(r["s2"]), pct(r["s3"]), pct(r["s4"]), pct(r["s5"]),
                 pct(r["s6"]), pct(r["mean_mitigation"]), f"{r['bsi']:.3f}", pct(r["s4_doorway"])]
        if i == 4:
            f.write("\\midrule\n")
        f.write(" & ".join((f"\\textbf{{{c}}}" if bold and j > 0 else c) for j, c in enumerate(cells)) + " \\\\\n")
prop = [r for r in X if r["architecture"].startswith("Proposed")]
put("casGAR", pct(prop[0]["gar"])); put("casMit", pct(prop[0]["mean_mitigation"])); put("casBSI", f"{prop[0]['bsi']:.3f}")
put("casGARs", pct(prop[1]["gar"])); put("casMits", pct(prop[1]["mean_mitigation"])); put("casBSIs", f"{prop[1]['bsi']:.3f}")
asm = pd.read_csv(os.path.join(RES, "exp4_assumptions.csv"))
with open(os.path.join(OUT, "tab_assumptions.tex"), "w") as f:
    for _, r in asm.iterrows():
        v = "--" if r.value == 1.0 and r.type == "A" else f"{100 * r.value:.2f}\\%"
        f.write(f"{r.input.replace('_', '-')} & {v} & {r.type} & {r.source.replace('_', '-').replace('<=', '$\\\\le$')} \\\\\n")

# ---------------- Exp 4 adaptive ----------------
for suf, tag in [("", "Ws"), ("_messaging", "Msg"), ("_structural", "Str")]:
    A = J(f"c4_adaptive_summary{suf}.json")
    for k in ["gar_c", "gar_i", "gar_ip", "mit_a10_i", "mit_a10_ip"]:
        key = k.replace("_", "").replace("garip", "garIP").replace("mita10ip", "aTenIP").replace("mita10i", "aTenI").replace("garc", "garC").replace("gari", "garI")
        put(f"ad{tag}{key}", pct(A[f"{k}_mean"])); put(f"ad{tag}{key}SD", pct(A[f"{k}_std"]))
    put(f"ad{tag}relayPass", pct(A["relay_rtt_pass_mean"])); put(f"ad{tag}genPass", pct(A["genuine_rtt_pass_mean"]))
    put(f"ad{tag}genMed", f"{A['genuine_rtt_median_mean']:.0f}"); put(f"ad{tag}relayMed", f"{A['relay_rtt_median_mean']:.0f}")
A = J("c4_adaptive_summary.json")
put("adDeadline", f"{A['parameters']['rtt_deadline_ms']:.0f}"); put("adPdet", f"{A['parameters']['p_coloc_detect']:.2f}")
put("adColocFP", pct(A["parameters"]["coloc_fp_rate"]))
put("adSeeds", str(A["num_seeds"])); put("adTrials", f"{A['num_seeds'] * A['trials_per_seed']:,}")
sens = pd.read_csv(os.path.join(RES, "c4_adaptive_sensitivity.csv"))
sv = lambda d, p, col: sens[(sens.rtt_deadline_ms == d) & (sens.coloc_detect_prob == p)][col].iloc[0]
for d in [200, 225, 250, 350]:
    put(f"sens{['Two','TwoTwoFive','TwoFifty','ThreeFifty'][[200,225,250,350].index(d)]}Gen", pct(sv(d, 0.65, "genuine_rtt_pass")))
    put(f"sens{['Two','TwoTwoFive','TwoFifty','ThreeFifty'][[200,225,250,350].index(d)]}Mit", pct(sv(d, 0.65, "a10_mitigation")))
    put(f"sens{['Two','TwoTwoFive','TwoFifty','ThreeFifty'][[200,225,250,350].index(d)]}MitNoCo", pct(sv(d, 0.0, "a10_mitigation")))
with open(os.path.join(OUT, "tab_deadline.tex"), "w") as f:
    for d in [175, 200, 225, 250, 300, 350]:
        f.write(f"{d} & {pct(sv(d, 0.65, 'genuine_rtt_pass'))}\\% & {pct(sv(d, 0.0, 'relay_rtt_pass'))}\\% & "
                f"{pct(sv(d, 0.0, 'a10_mitigation'))}\\% & {pct(sv(d, 0.65, 'a10_mitigation'))}\\% \\\\\n")

with open(os.path.join(OUT, "numbers.tex"), "w") as f:
    f.write("% AUTO-GENERATED by research/experiments/make_paper_numbers.py -- do not edit by hand\n")
    for k, v in sorted(M.items()):
        assert k.isalpha(), k
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(f"wrote {len(M)} macros and table files to {OUT}")
