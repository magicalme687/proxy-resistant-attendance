"""
Script: run_lfw_facenet_real_benchmark.py
Evaluates Biometric Verification on the official 6,000-pair LFW protocol
using real FaceNet (InceptionResnetV1 pretrained on VGGFace2).
Computes:
- ROC-AUC with 95% bootstrap CI (1,000 resamples)
- Equal Error Rate (EER) and EER threshold with 95% bootstrap CI
- TAR @ FAR = 0.1% and corresponding threshold
- Empirical Genuine Acceptance Rate (GAR) and Imposter False Acceptance Rate (FAR)
  at operational thresholds theta = 0.68, theta = 0.76, and theta = EER threshold
- Buddy-proxy imposter pass rate (FAR) for Table IV Scenario 5
Outputs:
- research/experiments/results/real_lfw_facenet_summary.json
- research/experiments/results/real_lfw_facenet_roc.csv
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_curve, auc

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LFW_BASE = os.path.join(ROOT_DIR, "datasets", "lfw", "lfw_funneled")
PAIRS_FILE = os.path.join(ROOT_DIR, "datasets", "lfw", "pairs.txt")
RESULTS_DIR = os.path.join(ROOT_DIR, "experiments", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_lfw_pairs(pairs_path, lfw_dir):
    with open(pairs_path, "r") as f:
        lines = [line.strip().split() for line in f.readlines() if line.strip()]
    header = lines[0]
    n_folds = int(header[0])
    n_pairs = int(header[1])

    pairs = []
    idx = 1
    for fold in range(n_folds):
        # Matched (genuine) pairs
        for _ in range(n_pairs):
            if idx >= len(lines):
                break
            row = lines[idx]
            idx += 1
            name = row[0]
            num1 = f"{int(row[1]):04d}"
            num2 = f"{int(row[2]):04d}"
            p1 = os.path.join(lfw_dir, name, f"{name}_{num1}.jpg")
            p2 = os.path.join(lfw_dir, name, f"{name}_{num2}.jpg")
            pairs.append((p1, p2, 1, fold, name, name))

        # Mismatched (imposter) pairs
        for _ in range(n_pairs):
            if idx >= len(lines):
                break
            row = lines[idx]
            idx += 1
            name1 = row[0]
            num1 = f"{int(row[1]):04d}"
            name2 = row[2]
            num2 = f"{int(row[3]):04d}"
            p1 = os.path.join(lfw_dir, name1, f"{name1}_{num1}.jpg")
            p2 = os.path.join(lfw_dir, name2, f"{name2}_{num2}.jpg")
            pairs.append((p1, p2, 0, fold, name1, name2))

    return pairs

_MTCNN = None
_ALIGN_STATS = {"detected": 0, "fallback_center_crop": 0}

def _get_mtcnn():
    global _MTCNN
    if _MTCNN is None:
        from facenet_pytorch import MTCNN
        # Standard facenet-pytorch evaluation setting: 160x160 crop, 14-px margin,
        # largest face kept, fixed image standardisation applied by post_process.
        _MTCNN = MTCNN(image_size=160, margin=14, keep_all=False,
                       select_largest=True, post_process=True, device="cpu")
    return _MTCNN

def preprocess_image(img_path):
    """MTCNN detection; when several faces are present, keep the one whose box
    centre is nearest the image centre (LFW targets are centred), then crop to
    160x160 with a 14-px margin. Falls back to a centre crop only if no face is
    found; the count is reported in the summary JSON."""
    img = Image.open(img_path).convert("RGB")
    m = _get_mtcnn()
    boxes, probs = m.detect(img)
    if boxes is not None and len(boxes):
        boxes = np.asarray(boxes, dtype=np.float32)
        w, h = img.size
        c = np.array([w / 2, h / 2])
        centres = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], 1)
        k = int(np.argmin(np.linalg.norm(centres - c, axis=1)))
        face = m.extract(img, boxes[k:k + 1], None)
        _ALIGN_STATS["detected"] += 1
        return face
    _ALIGN_STATS["fallback_center_crop"] += 1
    w, h = img.size
    img = img.crop(((w - 160) // 2, (h - 160) // 2, (w + 160) // 2, (h + 160) // 2))
    arr = (np.array(img, dtype=np.float32) - 127.5) / 128.0
    return torch.from_numpy(arr).permute(2, 0, 1).float()

def main():
    print("=" * 75)
    print("REAL LFW EVALUATION WITH INCEPTIONRESNETV1 (VGGFACE2 PRETRAINED)")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")

    pairs = load_lfw_pairs(PAIRS_FILE, LFW_BASE)
    print(f"Loaded {len(pairs)} pairs from {PAIRS_FILE}.")

    # Gather all unique image paths
    unique_paths = set()
    for p1, p2, _, _, _, _ in pairs:
        unique_paths.add(p1)
        unique_paths.add(p2)
    unique_paths = sorted(list(unique_paths))
    print(f"Found {len(unique_paths)} unique image files.")

    # Initialize model
    print("Loading InceptionResnetV1(pretrained='vggface2')...")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # Compute embeddings in batches
    batch_size = 64
    embeddings_dict = {}
    t0 = time.time()

    print("Extracting L2-normalized embeddings...")
    with torch.no_grad():
        for i in range(0, len(unique_paths), batch_size):
            batch_paths = unique_paths[i:i + batch_size]
            tensors = [preprocess_image(p) for p in batch_paths]
            batch_tensor = torch.stack(tensors).to(device)
            # InceptionResnetV1 returns 512-D L2-normalized embeddings
            feats = model(batch_tensor)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1).cpu().numpy()
            for path, feat in zip(batch_paths, feats):
                embeddings_dict[path] = feat

    emb_time = time.time() - t0
    print(f"Extracted {len(embeddings_dict)} embeddings in {emb_time:.2f}s ({len(unique_paths)/emb_time:.1f} img/s).")

    # Evaluate 6,000 pairs
    similarities = []
    labels = []
    for p1, p2, label, fold, _, _ in pairs:
        e1 = embeddings_dict[p1]
        e2 = embeddings_dict[p2]
        # Cosine similarity of L2-normalized vectors is dot product
        cos_sim = float(np.dot(e1, e2))
        similarities.append(cos_sim)
        labels.append(label)

    y_true = np.array(labels, dtype=int)
    y_scores = np.array(similarities, dtype=np.float32)

    gen_scores = y_scores[y_true == 1]
    imp_scores = y_scores[y_true == 0]

    # Compute ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = float(auc(fpr, tpr))

    # EER calculation
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # TAR @ FAR = 0.1% (0.001)
    target_far = 0.001
    far_idx = np.where(fpr <= target_far)[0]
    if len(far_idx) > 0:
        idx_01 = far_idx[-1]
        tar_at_01_far = float(tpr[idx_01])
        th_at_01_far = float(thresholds[idx_01])
        actual_far_at_01 = float(fpr[idx_01])
    else:
        idx_01 = 0
        tar_at_01_far = float(tpr[0])
        th_at_01_far = float(thresholds[0])
        actual_far_at_01 = float(fpr[0])

    # Metrics at theta = 0.68
    gar_at_068 = float(np.mean(gen_scores >= 0.68))
    far_at_068 = float(np.mean(imp_scores >= 0.68))

    # Metrics at theta = 0.76
    gar_at_076 = float(np.mean(gen_scores >= 0.76))
    far_at_076 = float(np.mean(imp_scores >= 0.76))

    # Metrics at theta = eer_threshold
    gar_at_eer = float(np.mean(gen_scores >= eer_threshold))
    far_at_eer = float(np.mean(imp_scores >= eer_threshold))

    # 1,000 Bootstrap Resamples for 95% Confidence Intervals
    print("Computing 1,000 bootstrap resamples for 95% CIs...")
    np.random.seed(42)
    boot_aucs = []
    boot_eers = []
    boot_tars_01 = []
    n_total = len(y_true)

    for _ in range(1000):
        b_idx = np.random.choice(n_total, n_total, replace=True)
        b_y = y_true[b_idx]
        b_s = y_scores[b_idx]
        if len(np.unique(b_y)) < 2:
            continue
        b_fpr, b_tpr, b_th = roc_curve(b_y, b_s)
        boot_aucs.append(float(auc(b_fpr, b_tpr)))
        b_fnr = 1.0 - b_tpr
        b_e_idx = int(np.nanargmin(np.abs(b_fpr - b_fnr)))
        boot_eers.append(float((b_fpr[b_e_idx] + b_fnr[b_e_idx]) / 2.0))
        b_far_idx = np.where(b_fpr <= target_far)[0]
        if len(b_far_idx) > 0:
            boot_tars_01.append(float(b_tpr[b_far_idx[-1]]))
        else:
            boot_tars_01.append(float(b_tpr[0]))

    ci_auc = [float(np.percentile(boot_aucs, 2.5)), float(np.percentile(boot_aucs, 97.5))]
    ci_eer = [float(np.percentile(boot_eers, 2.5)), float(np.percentile(boot_eers, 97.5))]
    ci_tar_01 = [float(np.percentile(boot_tars_01, 2.5)), float(np.percentile(boot_tars_01, 97.5))]

    results = {
        "model": "InceptionResnetV1 (pretrained='vggface2')",
        "preprocessing": "MTCNN detection, centre-nearest face, 160x160 crop (margin 14)",
        "mtcnn_detected": _ALIGN_STATS["detected"],
        "mtcnn_fallback_center_crop": _ALIGN_STATS["fallback_center_crop"],
        "dataset": "LFW (Labeled Faces in the Wild) 6,000 Pairs",
        "total_pairs": len(pairs),
        "genuine_pairs": int(np.sum(y_true == 1)),
        "imposter_pairs": int(np.sum(y_true == 0)),
        "genuine_sim_mean": round(float(np.mean(gen_scores)), 4),
        "genuine_sim_std": round(float(np.std(gen_scores)), 4),
        "genuine_sim_min": round(float(np.min(gen_scores)), 4),
        "genuine_sim_max": round(float(np.max(gen_scores)), 4),
        "imposter_sim_mean": round(float(np.mean(imp_scores)), 4),
        "imposter_sim_std": round(float(np.std(imp_scores)), 4),
        "imposter_sim_min": round(float(np.min(imp_scores)), 4),
        "imposter_sim_max": round(float(np.max(imp_scores)), 4),
        "roc_auc": round(roc_auc, 5),
        "roc_auc_ci_95": [round(ci_auc[0], 5), round(ci_auc[1], 5)],
        "eer": round(eer, 5),
        "eer_ci_95": [round(ci_eer[0], 5), round(ci_eer[1], 5)],
        "eer_threshold": round(eer_threshold, 4),
        "tar_at_far_01_percent": round(tar_at_01_far * 100.0, 3),
        "tar_at_far_01_ci_95": [round(ci_tar_01[0] * 100.0, 3), round(ci_tar_01[1] * 100.0, 3)],
        "threshold_at_far_01": round(th_at_01_far, 4),
        "actual_far_at_far_01": round(actual_far_at_01 * 100.0, 4),
        "theta_068_metrics": {
            "threshold": 0.68,
            "gar_percent": round(gar_at_068 * 100.0, 3),
            "frr_percent": round((1.0 - gar_at_068) * 100.0, 3),
            "far_percent": round(far_at_068 * 100.0, 3),
            "imposter_accept_count": int(np.sum(imp_scores >= 0.68)),
            "buddy_proxy_mitigation_percent": round((1.0 - far_at_068) * 100.0, 3)
        },
        "theta_076_metrics": {
            "threshold": 0.76,
            "gar_percent": round(gar_at_076 * 100.0, 3),
            "frr_percent": round((1.0 - gar_at_076) * 100.0, 3),
            "far_percent": round(far_at_076 * 100.0, 3),
            "imposter_accept_count": int(np.sum(imp_scores >= 0.76)),
            "buddy_proxy_mitigation_percent": round((1.0 - far_at_076) * 100.0, 3)
        },
        "theta_eer_metrics": {
            "threshold": round(eer_threshold, 4),
            "gar_percent": round(gar_at_eer * 100.0, 3),
            "frr_percent": round((1.0 - gar_at_eer) * 100.0, 3),
            "far_percent": round(far_at_eer * 100.0, 3),
            "imposter_accept_count": int(np.sum(imp_scores >= eer_threshold)),
            "buddy_proxy_mitigation_percent": round((1.0 - far_at_eer) * 100.0, 3)
        }
    }

    print("\n" + "=" * 75)
    print("SUMMARY OF EMPIRICAL LFW BENCHMARK WITH FACENET (VGGFACE2)")
    print("=" * 75)
    print(f"ROC-AUC:                {results['roc_auc']:.5f} (95% CI: [{results['roc_auc_ci_95'][0]:.5f}, {results['roc_auc_ci_95'][1]:.5f}])")
    print(f"Equal Error Rate (EER): {results['eer']*100:.3f}% at theta = {results['eer_threshold']} (95% CI: [{results['eer_ci_95'][0]*100:.3f}%, {results['eer_ci_95'][1]*100:.3f}%])")
    print(f"TAR @ FAR = 0.1%:       {results['tar_at_far_01_percent']:.3f}% at theta = {results['threshold_at_far_01']} (95% CI: [{results['tar_at_far_01_ci_95'][0]:.3f}%, {results['tar_at_far_01_ci_95'][1]:.3f}%])")
    print(f"Genuine Sim (Mean+-SD): {results['genuine_sim_mean']:.4f} +- {results['genuine_sim_std']:.4f}")
    print(f"Imposter Sim (Mean+-SD):{results['imposter_sim_mean']:.4f} +- {results['imposter_sim_std']:.4f}")
    print(f"\nOperating Point theta = 0.68:")
    print(f"  GAR: {results['theta_068_metrics']['gar_percent']:.3f}%, FAR: {results['theta_068_metrics']['far_percent']:.3f}% ({results['theta_068_metrics']['imposter_accept_count']}/3000), Buddy Mitigation: {results['theta_068_metrics']['buddy_proxy_mitigation_percent']:.3f}%")
    print(f"\nOperating Point theta = 0.76:")
    print(f"  GAR: {results['theta_076_metrics']['gar_percent']:.3f}%, FAR: {results['theta_076_metrics']['far_percent']:.3f}% ({results['theta_076_metrics']['imposter_accept_count']}/3000), Buddy Mitigation: {results['theta_076_metrics']['buddy_proxy_mitigation_percent']:.3f}%")
    print(f"\nOperating Point theta = EER ({results['eer_threshold']}):")
    print(f"  GAR: {results['theta_eer_metrics']['gar_percent']:.3f}%, FAR: {results['theta_eer_metrics']['far_percent']:.3f}% ({results['theta_eer_metrics']['imposter_accept_count']}/3000), Buddy Mitigation: {results['theta_eer_metrics']['buddy_proxy_mitigation_percent']:.3f}%")

    out_json = os.path.join(RESULTS_DIR, "real_lfw_facenet_summary.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics summary to: {out_json}")

    # Save ROC curve points and score distribution
    df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    out_roc = os.path.join(RESULTS_DIR, "real_lfw_facenet_roc.csv")
    df_roc.to_csv(out_roc, index=False)
    print(f"Saved ROC curve points to: {out_roc}")

    df_scores = pd.DataFrame({"label": y_true, "similarity": y_scores})
    out_scores = os.path.join(RESULTS_DIR, "real_lfw_facenet_scores.csv")
    df_scores.to_csv(out_scores, index=False)
    print(f"Saved pair similarities to: {out_scores}")

if __name__ == "__main__":
    main()
